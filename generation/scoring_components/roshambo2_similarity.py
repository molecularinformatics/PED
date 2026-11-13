"""REINVENT scoring component: 3D shape and pharmacophore similarity via Roshambo2."""

from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class Roshambo2Similarity(BaseScoreComponent):
    """Score each generated molecule by its best combined shape-and-color Tanimoto
    against a reference ligand, maximised over conformers and stereoisomers.

    Specific parameters: ``reference_path`` (SDF with 3D coordinates), ``num_confs``,
    ``num_isomers``, ``num_processes``, ``random_seed``, ``embed_method``,
    ``combination_param``, ``backend``.
    """

    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        p = self.parameters.specific_parameters
        self._reference_path = p["reference_path"]
        self._num_confs = p.get("num_confs", 25)
        self._num_isomers = p.get("num_isomers", 4)
        self._num_processes = p.get("num_processes", 16)
        self._seed = p.get("random_seed", 42)
        self._embed_method = p.get("embed_method", "ETKDGv3")
        self._combination_param = p.get("combination_param", 0.5)
        self._backend = p.get("backend", "cpp")

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        score = self._calculate_similarity(molecules)
        return ComponentSummary(total_score=score, parameters=self.parameters)

    def _conformers_for(self, mol, index: int) -> List:
        """3D conformers for one molecule, named ``{index}_{isomer}`` for grouping."""
        options = StereoEnumerationOptions(maxIsomers=self._num_isomers,
                                           rand=self._seed, unique=True,
                                           onlyUnassigned=True)
        out = []
        for isomer_index, isomer in enumerate(EnumerateStereoisomers(mol, options=options)):
            isomer = Chem.AddHs(isomer)
            if AllChem.EmbedMolecule(isomer, maxAttempts=100, randomSeed=self._seed,
                                     enforceChirality=True) != 0:
                continue
            AllChem.MMFFOptimizeMolecule(isomer)
            isomer.SetProp("_Name", f"{index}_{isomer_index}")
            if self._num_confs:
                params = getattr(rdDistGeom, self._embed_method)()
                params.randomSeed = self._seed
                params.clearConfs = True
                params.numThreads = 1
                rdDistGeom.EmbedMultipleConfs(isomer, numConfs=self._num_confs, params=params)
            out.append(isomer)
        return out

    def _calculate_similarity(self, query_mols: List) -> np.ndarray:
        from roshambo2 import Roshambo2

        reference = Chem.SDMolSupplier(self._reference_path)[0]
        if reference is None:
            raise ValueError(f"could not read a reference molecule from {self._reference_path}")

        dataset = []
        for index, mol in enumerate(query_mols):
            if mol is not None:
                dataset.extend(self._conformers_for(mol, index))
        if not dataset:
            return np.zeros(len(query_mols), dtype=float)

        calculator = Roshambo2(reference, dataset, color=True, data_mode="in_memory",
                               verbosity=0, n_cpus_prepare=self._num_processes,
                               conformers_have_unique_names=False)
        scores = calculator.compute(backend=self._backend, optim_mode="combination",
                                    combination_param=self._combination_param,
                                    reduce_over_conformers=True, write_scores=False,
                                    max_results=len(dataset))

        df = next(iter(scores.values()))
        df["query_index"] = df["name"].str.split("_").str[0].astype(int)
        best = (df.sort_values(["query_index", "tanimoto_combination"],
                               ascending=[True, False])
                  .drop_duplicates(subset="query_index", keep="first"))

        # Molecules whose embedding or scoring failed score 0 rather than drop out.
        full = pd.DataFrame({"query_index": range(len(query_mols))})
        merged = full.merge(best[["query_index", "tanimoto_combination"]],
                            on="query_index", how="left").fillna(0.0)
        return merged["tanimoto_combination"].to_numpy(dtype=float)
