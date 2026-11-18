"""
Scoring oracles for ``synformer.molopt``.

Four oracles are provided, one per reward used in the paper's comparison:

  - ECFP4Oracle        ECFP4 (Morgan r=2, 2048 bit) Tanimoto similarity to a
                       reference SMILES. Pure RDKit; no extra dependency.
  - MolformerOracle    "PED-MolFormer": distance between pretrained MolFormer
                       embeddings of the query and of a reference SMILES.
  - GeodiffOracle      "PED-GeoDiff": the same, with pretrained GeoDiff
                       embeddings (2D / 3D / concatenated).
  - Roshambo2Oracle     3D shape + pharmacophore overlap (Roshambo2) against a
                       reference ligand supplied as an SDF.

All four implement the same two methods, which is the whole interface the
optimisation loop relies on:

  - ``__call__(smiles: str) -> float``        single molecule
  - ``score_batch(smiles_list) -> list[float]``  one batch (preferred)

Every oracle returns a score in [0, 1], higher = better, and returns 0.0 for
an empty or unparseable SMILES.

Dependencies beyond the base environment
----------------------------------------
ECFP4 needs nothing but RDKit. The other three need the artefacts listed in
README.md ("Oracles and their dependencies"):

  - PED-MolFormer: ``ibm/MoLFormer-XL-both-10pct`` from the HuggingFace hub.
  - PED-GeoDiff:   the public GeoDiff ``drugs_default.pt`` checkpoint.
  - Roshambo2:      the Roshambo2 package and a reference ligand SDF.

The two PED oracles reach their pretrained encoders through ``reinvent_scoring``
scoring components (``MolformerDistance``, ``GeoDiffDistance``). Any object
exposing ``calculate_score(list[rdkit.Chem.Mol])`` and returning an object with
a ``total_score`` sequence can be substituted; see ``_get_scorer`` below for the
exact parameters used to produce the reported numbers.
"""

import gc
import sys
import logging
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from multiprocessing import Pool
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# reinvent_scoring imports optional native backends (OpenEye) at module import
# time. Mock the ones that are absent so the embedding components can be
# imported without those licensed libraries installed.
# ---------------------------------------------------------------------------

def _mock_unavailable_libs():
    from unittest.mock import MagicMock
    for mod in [
        "openeye", "openeye.oechem", "openeye.oeomega", "openeye.oeshape",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


# ---------------------------------------------------------------------------
# MolFormer Distance Oracle
# ---------------------------------------------------------------------------

class MolformerOracle:
    """
    Scores molecules by MolFormer embedding distance to a reference SMILES.

    Score: [0, 1], higher = more similar.
      distance ~sigmoid_low  → score ~1
      distance ~sigmoid_high → score ~0

    Sigmoid defaults (high=17, low=5, k=-0.25) are the ones used for the
    reported BTK numbers.
    """

    _scorer_cache: dict = {}

    def __init__(
        self,
        reference_smiles: str,
        distance_metric: str = "euclidean",
        sigmoid_high: float = 17.0,
        sigmoid_low: float = 5.0,
        sigmoid_k: float = -0.25,
    ):
        self.reference_smiles = reference_smiles
        self.distance_metric = distance_metric
        self.sigmoid_high = sigmoid_high
        self.sigmoid_low = sigmoid_low
        self.sigmoid_k = sigmoid_k
        self._cache_key = (reference_smiles, distance_metric, sigmoid_high, sigmoid_low, sigmoid_k)

    def _get_scorer(self):
        if self._cache_key not in MolformerOracle._scorer_cache:
            _mock_unavailable_libs()
            try:
                from reinvent_scoring.scoring.component_parameters import ComponentParameters
                from reinvent_scoring.scoring.score_components.embedding import MolformerDistance
            except ImportError:
                raise ImportError(
                    "reinvent_scoring is not installed. Run:\n"
                    "  pip install reinvent-scoring"
                )
            logger.info("Loading MolFormer model (one-time, ~10-30s)...")
            params = ComponentParameters(
                component_type="molformer_distance",
                name="MolformerDistance",
                weight=1.0,
                specific_parameters={
                    "reference_smiles": self.reference_smiles,
                    "distance_metric": self.distance_metric,
                    "transformation": {
                        "transformation_type": "sigmoid",
                        "high": self.sigmoid_high,
                        "k": self.sigmoid_k,
                        "low": self.sigmoid_low,
                    },
                },
            )
            MolformerOracle._scorer_cache[self._cache_key] = MolformerDistance(params)
            logger.info("MolFormer model loaded.")
        return MolformerOracle._scorer_cache[self._cache_key]

    def score_batch(self, smiles_list: List[str]) -> List[float]:
        """Score a batch of SMILES strings. Preferred over repeated single calls."""
        outputs = [0.0] * len(smiles_list)
        valid_mols, positions = [], []
        for i, smi in enumerate(smiles_list):
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_mols.append(mol)
                    positions.append(i)
        if valid_mols:
            scorer = self._get_scorer()
            summary = scorer.calculate_score(valid_mols)
            for pos, score in zip(positions, summary.total_score):
                outputs[pos] = float(score)
            del summary
            for mol in valid_mols:
                mol.RemoveAllConformers()
            del valid_mols
            gc.collect()
        return outputs

    def __call__(self, smiles: str) -> float:
        """Single-SMILES interface, compatible with synformer Oracle.evaluator."""
        return self.score_batch([smiles])[0]


# ---------------------------------------------------------------------------
# GeoDiff Distance Oracle
# ---------------------------------------------------------------------------

class GeodiffOracle:
    """
    Scores molecules by GeoDiff 3D embedding distance to a reference SMILES.

    embedding_mode:
      "2D"          - graph topology only, fastest
      "3D"          - pure 3D geometry encoding
      "concat"      - 2D + 3D concatenated
      "concat_norm" - 2D + 3D concatenated with normalization (recommended)

    Note: all modes require RDKit conformer generation as GNN input.
    The mode only affects which part of the embedding is used for the distance.

    Sigmoid defaults (high=10, low=1, k=-0.25) are the ones used for the
    reported BTK numbers in "3D" and "concat_norm" mode; "2D" uses
    high=1.2, low=0.1.
    """

    _scorer_cache: dict = {}

    def __init__(
        self,
        reference_smiles: str,
        distance_metric: str = "euclidean",
        embedding_mode: str = "concat_norm",
        sigmoid_high: float = 10.0,
        sigmoid_low: float = 1.0,
        sigmoid_k: float = -0.25,
        scoring_timeout: float = 600.0,
        num_workers: int = 16,
        batch_size: int = 128,
    ):
        self.reference_smiles = reference_smiles
        self.distance_metric = distance_metric
        self.embedding_mode = embedding_mode
        self.sigmoid_high = sigmoid_high
        self.sigmoid_low = sigmoid_low
        self.sigmoid_k = sigmoid_k
        self.scoring_timeout = scoring_timeout
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._cache_key = (
            reference_smiles, distance_metric, embedding_mode,
            sigmoid_high, sigmoid_low, sigmoid_k, num_workers, batch_size,
        )

    def _get_scorer(self):
        if self._cache_key not in GeodiffOracle._scorer_cache:
            _mock_unavailable_libs()
            try:
                from reinvent_scoring.scoring.component_parameters import ComponentParameters
                from reinvent_scoring.scoring.score_components.embedding import GeoDiffDistance
            except ImportError:
                raise ImportError(
                    "reinvent_scoring is not installed. Run:\n"
                    "  pip install reinvent-scoring"
                )
            logger.info("Loading GeoDiff model (one-time cost)...")
            params = ComponentParameters(
                component_type="geodiff_distance",
                name="GeoDiffDistance",
                weight=1.0,
                specific_parameters={
                    "reference_smiles": self.reference_smiles,
                    "distance_metric": self.distance_metric,
                    "embedding_mode": self.embedding_mode,
                    "num_workers": self.num_workers,
                    "batch_size": self.batch_size,
                    "transformation": {
                        "transformation_type": "sigmoid",
                        "high": self.sigmoid_high,
                        "k": self.sigmoid_k,
                        "low": self.sigmoid_low,
                    },
                },
            )
            GeodiffOracle._scorer_cache[self._cache_key] = GeoDiffDistance(params)
            logger.info("GeoDiff model loaded.")
        return GeodiffOracle._scorer_cache[self._cache_key]

    def score_batch(self, smiles_list: List[str]) -> List[float]:
        """Score a batch of SMILES strings with timeout protection."""
        outputs = [0.0] * len(smiles_list)
        valid_mols, positions = [], []
        for i, smi in enumerate(smiles_list):
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_mols.append(mol)
                    positions.append(i)
        if valid_mols:
            scorer = self._get_scorer()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(scorer.calculate_score, valid_mols)
                    summary = future.result(timeout=self.scoring_timeout)
                for pos, score in zip(positions, summary.total_score):
                    outputs[pos] = float(score)
                del summary
            except FuturesTimeoutError:
                logger.warning(
                    "GeoDiff scoring timed out after %.0fs for %d molecules — returning 0.0",
                    self.scoring_timeout, len(valid_mols),
                )
            finally:
                for mol in valid_mols:
                    mol.RemoveAllConformers()
                del valid_mols
                gc.collect()
        return outputs

    def __call__(self, smiles: str) -> float:
        return self.score_batch([smiles])[0]


# ---------------------------------------------------------------------------
# Roshambo2 3D Similarity Oracle
# ---------------------------------------------------------------------------

def _enumerate_stereoisomers(rdmol, num_isomers=4, random_seed=109838974):
    opts = StereoEnumerationOptions(
        maxIsomers=num_isomers, rand=random_seed, unique=True, onlyUnassigned=True
    )
    return list(EnumerateStereoisomers(rdmol, options=opts))


def _embed_and_optimize(rdmol, random_seed=109838974):
    rdmol = Chem.AddHs(rdmol)
    if AllChem.EmbedMolecule(rdmol, maxAttempts=100, randomSeed=random_seed, enforceChirality=True) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(rdmol)
    return rdmol


def _generate_confs(rdmol, num_confs=25, num_cpus=1, random_seed=109838974):
    args = rdDistGeom.ETKDGv3()
    args.randomSeed = random_seed
    args.clearConfs = True
    args.numThreads = num_cpus
    rdDistGeom.EmbedMultipleConfs(rdmol, numConfs=num_confs, params=args)
    return rdmol


def _prep_rdmol(rdmol, num_isomers=4, num_confs=25, num_cpus=1, random_seed=109838974):
    """Enumerate stereoisomers and generate conformers for one molecule."""
    if rdmol is None:
        return []
    name = rdmol.GetProp("_Name") if rdmol.HasProp("_Name") else "mol"
    conformers = []
    for iso_idx, isomer in enumerate(_enumerate_stereoisomers(rdmol, num_isomers, random_seed)):
        iso_mol = _embed_and_optimize(isomer, random_seed)
        if iso_mol is not None:
            iso_mol.SetProp("_Name", f"{name}_{iso_idx}")
            if num_confs:
                iso_mol = _generate_confs(iso_mol, num_confs=num_confs, num_cpus=num_cpus, random_seed=random_seed)
            conformers.append(iso_mol)
    return conformers


def _calculate_roshambo2_scores(query_mols, supplier_path, num_isomers=4, num_confs=25, num_cpus=40):
    """
    Core Roshambo2 scoring: enumerate conformers then run Roshambo2.
    Returns a numpy array of scores aligned with query_mols.
    """
    import pandas as pd
    from roshambo2 import Roshambo2

    reference_mol = Chem.SDMolSupplier(supplier_path)[0]
    if reference_mol is None:
        raise ValueError(f"Could not load reference molecule from {supplier_path}")

    # Tag each molecule with its original index
    dataset_mols = []
    for idx, mol in enumerate(query_mols):
        if mol is not None:
            mol.SetProp("_Name", str(idx))
            dataset_mols.append(mol)

    if not dataset_mols:
        return np.zeros(len(query_mols))

    # Generate conformers in parallel
    input_data = [(mol, num_isomers, num_confs, 1) for mol in dataset_mols]
    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(_prep_rdmol, input_data)

    mols_with_confs = [mol for sublist in results for mol in sublist]
    del results  # free worker-returned conformer objects immediately
    gc.collect()

    if not mols_with_confs:
        del dataset_mols, mols_with_confs
        return np.zeros(len(query_mols))

    # Run Roshambo2. max_results defaults to 10,000, which silently truncates a
    # batch of 128 molecules at 4 stereoisomers and 25 conformers each.
    use_cuda = torch.cuda.is_available()
    calculator = Roshambo2(reference_mol, mols_with_confs, color=True, data_mode="in_memory",
                           verbosity=0, n_cpus_prepare=num_cpus,
                           conformers_have_unique_names=False)
    backend = "cuda" if use_cuda else "cpp"
    scores_dict = calculator.compute(
        backend=backend,
        reduce_over_conformers=True,
        optim_mode="combination",
        write_scores=False,
        combination_param=0.5,
        max_results=len(mols_with_confs),
    )

    # Align scores back to original molecule indices
    df = next(iter(scores_dict.values()))
    df["ID"] = df["name"].str.split("_").str[0].astype(int)
    df = df.sort_values(by=["ID", "tanimoto_combination"], ascending=[True, False])
    df = df.drop_duplicates(subset="ID", keep="first")
    result = pd.merge(
        pd.DataFrame({"ID": range(len(query_mols))}),
        df[["ID", "tanimoto_combination"]],
        on="ID",
        how="left",
    ).fillna(0)
    scores = result["tanimoto_combination"].to_numpy().copy()

    # Explicitly free all conformer-heavy objects before returning
    for mol in mols_with_confs:
        mol.RemoveAllConformers()
    del calculator, mols_with_confs, dataset_mols, scores_dict, df, result
    gc.collect()

    return scores


class Roshambo2Oracle:
    """
    Scores molecules by 3D shape + pharmacophore similarity (Roshambo2) to a reference
    molecule loaded from an SDF file (typically an experimental X-ray structure).

    Score: [0, 1] (tanimoto_combination of shape + pharmacophore overlap).

    Requires the Roshambo2 package to be importable (see README.md).

    supplier_path: path to SDF file with the reference molecule (1 molecule).
    num_confs    : conformers per stereoisomer (default 25, more = better coverage, slower).
    num_cpus     : parallel processes for conformer generation (default 40).
    """

    def __init__(
        self,
        supplier_path: str,
        num_confs: int = 25,
        num_cpus: int = 40,
    ):
        self.supplier_path = supplier_path
        self.num_confs = num_confs
        self.num_cpus = num_cpus

    def score_batch(self, smiles_list: List[str]) -> List[float]:
        """Score a batch. Strongly preferred over single-SMILES calls for Roshambo2."""
        outputs = [0.0] * len(smiles_list)
        valid_mols, positions = [], []
        for i, smi in enumerate(smiles_list):
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_mols.append(Chem.RemoveHs(mol))
                    positions.append(i)
        if valid_mols:
            try:
                scores = _calculate_roshambo2_scores(
                    valid_mols,
                    self.supplier_path,
                    num_confs=self.num_confs,
                    num_cpus=self.num_cpus,
                )
                for pos, score in zip(positions, scores):
                    outputs[pos] = float(score)
            except ImportError:
                raise ImportError(
                    "The Roshambo2 package is not importable; see README.md "
                    "(\"Oracles and their dependencies\")."
                )
        return outputs

    def __call__(self, smiles: str) -> float:
        return self.score_batch([smiles])[0]


# ---------------------------------------------------------------------------
# ECFP4 Tanimoto Similarity Oracle
# ---------------------------------------------------------------------------

class ECFP4Oracle:
    """
    Scores molecules by ECFP4 Tanimoto similarity to a reference SMILES.

    ECFP4 == Morgan fingerprint of radius 2 (diameter 4), 2048 bits by default.
    Note that `radius=2` is what makes this ECFP4; radius 3 would be ECFP6.

    Score: [0, 1], higher = more similar. 1.0 means an identical fingerprint
    (so the reference molecule itself scores exactly 1.0). No sigmoid or other
    transformation is applied -- raw Tanimoto is already bounded in [0, 1],
    which keeps it on the same scale as Roshambo2Oracle.

    Invalid / empty SMILES score 0.0, matching the other oracles here.

    The oracle is pure RDKit: no model loading, no conformers, ~1e5 mol/s, so
    `score_batch` exists only for interface compatibility with the molopt loop.
    """

    def __init__(
        self,
        reference_smiles: str,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
    ):
        from rdkit.Chem import rdFingerprintGenerator

        self.reference_smiles = reference_smiles
        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality

        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if ref_mol is None:
            raise ValueError(f"Could not parse reference SMILES: {reference_smiles!r}")

        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=use_chirality
        )
        self._ref_fp = self._gen.GetFingerprint(ref_mol)

    def score_batch(self, smiles_list: List[str]) -> List[float]:
        from rdkit import DataStructs

        outputs = [0.0] * len(smiles_list)
        fps, positions = [], []
        for i, smi in enumerate(smiles_list):
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fps.append(self._gen.GetFingerprint(mol))
            positions.append(i)
        if fps:
            sims = DataStructs.BulkTanimotoSimilarity(self._ref_fp, fps)
            for pos, sim in zip(positions, sims):
                outputs[pos] = float(sim)
        return outputs

    def __call__(self, smiles: str) -> float:
        return self.score_batch([smiles])[0]
