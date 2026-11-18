import dataclasses
import os
import pathlib
from collections import OrderedDict
from collections.abc import Callable
from typing import TypeAlias, overload

import joblib
import numpy as np
import tdc
import yaml
from rdkit import Chem


@dataclasses.dataclass(frozen=True)
class MolBufferItem:
    score: float
    index: int

    def __iter__(self):
        return iter([self.score, self.index])

    def __getitem__(self, key: int):
        return [self.score, self.index][key]


_SmilesString: TypeAlias = str
MolBuffer: TypeAlias = OrderedDict[_SmilesString, MolBufferItem]


def top_auc(buffer: MolBuffer, top_n: int, finish: bool, freq_log: int, max_oracle_calls: int) -> float:
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


def _safe_mean(scorer, smis: list) -> float:
    """Call scorer and return mean, returning nan on any exception (e.g. TDC SA scorer division by zero)."""
    try:
        return float(np.mean(scorer(smis)))
    except Exception:
        return float("nan")


class Oracle:
    def __init__(
        self,
        evaluator: tdc.Evaluator | tdc.Oracle | Callable[[_SmilesString], float],
        output_dir: str | pathlib.Path,
        max_oracle_calls=10000,
        freq_log=100,
        mol_buffer: MolBuffer | None = None,
        save_last_n=5,
        n_cpu=12,
    ):
        self.name = None
        self.task_label = None
        self.evaluator = evaluator
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = freq_log

        # self.mol_buffer = mol_buffer or OrderedDict()
        self.mol_buffer = mol_buffer or {}
        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")

        self.last_log = 0
        self.output_dir = pathlib.Path(output_dir)
        self.save_last_n = save_last_n
        self.saved_paths: list[pathlib.Path] = []
        self.n_cpu = n_cpu

    @property
    def budget(self):
        return self.max_oracle_calls

    def sort_buffer(self):
        # self.mol_buffer = OrderedDict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1].score, reverse=True))
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_buffer(self):
        if len(self.saved_paths) >= self.save_last_n:
            path = self.saved_paths.pop(0)
            path.unlink()
        output_path = self.output_dir / f"oracle_{self.last_log}.yaml"
        self.sort_buffer()
        with open(output_path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
        self.saved_paths.append(output_path)

    def log_intermediate(self, mols=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[
                        : self.max_oracle_calls
                    ]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        metrics: dict[str, float] = {
            "avg_top1": np.max(scores),
            "avg_top10": np.mean(sorted(scores, reverse=True)[:10]),
            "avg_top100": np.mean(scores),
            "avg_sa_top100": _safe_mean(self.sa_scorer, smis),  # Top 100
            "div_top100": self.diversity_evaluator(smis),  # Top 100
        }
        if finish:
            metrics["auc_top1"] = top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls)
            metrics["auc_top10"] = top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls)
            metrics["auc_top100"] = top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls)
            metrics["n_oracle"] = float(n_calls)

        formatted = " | ".join(
            [f"{n_calls}/{self.max_oracle_calls}"] + [f"{key}: {value:.3f}" for key, value in metrics.items()]
        )
        print(formatted)

        with open(self.output_dir / "oracle_log.txt", "a") as f:
            f.write(formatted + "\n")

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi: _SmilesString) -> float:
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0.0

        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0.0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi not in self.mol_buffer:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer) + 1]
            return self.mol_buffer[smi][0]

    @overload
    def __call__(self, smiles_lst: list[_SmilesString]) -> list[float]:
        ...

    @overload
    def __call__(self, smiles_lst: _SmilesString) -> float:
        ...

    def __call__(self, smiles_lst: list[_SmilesString] | _SmilesString) -> list[float] | float:
        """
        Score
        """
        is_input_list = isinstance(smiles_lst, list)
        if not isinstance(smiles_lst, list):
            smiles_lst = [smiles_lst]
        score_list: list[float] = []

        if self.n_cpu > 1:
            score_list = joblib.Parallel(n_jobs=self.n_cpu)(joblib.delayed(self.evaluator)(smi) for smi in smiles_lst)

            for idx, smi in enumerate(smiles_lst):
                if smi not in self.mol_buffer:
                    self.mol_buffer[smi] = [score_list[idx], len(self.mol_buffer) + 1]

            if len(self.mol_buffer) > self.freq_log:
                self.sort_buffer()
                self.log_intermediate()
                self.save_buffer()
        else:
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_buffer()

        return score_list if is_input_list else score_list[0]

    @property
    def is_finished(self):
        return len(self.mol_buffer) >= self.max_oracle_calls

    def add_to_buffer(self, smi: _SmilesString, score: float):
        if smi not in self.mol_buffer:
            self.mol_buffer[smi] = [score, len(self.mol_buffer) + 1]
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_buffer()


def get_oracle_evaluator(name: str) -> tdc.Evaluator | tdc.Oracle | Callable[[_SmilesString], float]:
    # Split on first ":" only so reference_smiles containing ":" are handled correctly
    try:
        domain, task = name.split(":", 1)
    except ValueError:
        raise ValueError(f"Invalid oracle name: {name}")

    if domain == "tdc":
        return tdc.Oracle(task)

    # --- pretrained-embedding-distance (PED) and 3D-shape oracles ---
    # task = reference_smiles for molformer/geodiff, path to SDF for roshambo2

    elif domain == "molformer":
        # MolFormer embedding distance (euclidean, default BTK sigmoid params)
        from synformer.molopt.oracles import MolformerOracle
        return MolformerOracle(reference_smiles=task)

    elif domain == "geodiff":
        # GeoDiff concat_norm embedding (default, slowest but most informative)
        from synformer.molopt.oracles import GeodiffOracle
        return GeodiffOracle(reference_smiles=task, embedding_mode="concat_norm",
                             sigmoid_high=10.0, sigmoid_low=1.0, sigmoid_k=-0.25)

    elif domain == "geodiff2d":
        # GeoDiff 2D embedding (fastest, graph-topology only)
        from synformer.molopt.oracles import GeodiffOracle
        return GeodiffOracle(reference_smiles=task, embedding_mode="2D",
                             sigmoid_high=1.2, sigmoid_low=0.1, sigmoid_k=-0.25)

    elif domain == "geodiff3d":
        # GeoDiff 3D embedding (pure geometry)
        from synformer.molopt.oracles import GeodiffOracle
        return GeodiffOracle(reference_smiles=task, embedding_mode="3D",
                             sigmoid_high=10.0, sigmoid_low=1.0, sigmoid_k=-0.25)

    elif domain == "ecfp4":
        # ECFP4 (Morgan radius 2, 2048 bit) Tanimoto similarity; task = reference_smiles
        from synformer.molopt.oracles import ECFP4Oracle
        return ECFP4Oracle(reference_smiles=task)

    elif domain == "roshambo2":
        # Roshambo2 3D shape+pharmacophore similarity; task = path to reference SDF
        from synformer.molopt.oracles import Roshambo2Oracle
        return Roshambo2Oracle(supplier_path=task)

    else:
        raise ValueError(f"Unknown oracle domain: {domain}")
