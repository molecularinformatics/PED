import logging
import os
from typing import List, Optional

import numpy as np
import torch

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary

from .geodiff_utils import (
    canonicalize_smiles,
    cosine_distance,
    euclidean_distance,
    geodiff_embed_batch,
    load_geodiff_model_and_transforms,
    smiles_to_mol,
    sigmoid_transform
)

logger = logging.getLogger(__name__)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CKPT_PATH = os.path.join(_CURRENT_DIR, "geodiff/log/model/checkpoints/drugs_default.pt")

class _GeoDiffBackend:
    _instance: Optional["_GeoDiffBackend"] = None

    def __init__(self, ckpt_path: str, device: torch.device):
        logger.info("Loading GeoDiff checkpoint '%s' onto %s (one-time cost).", ckpt_path, device)
        self.ckpt_path = ckpt_path
        self.device = device
        self.model, self.transforms = load_geodiff_model_and_transforms(
            ckpt_path=ckpt_path,
            device=device,
        )
        logger.info("GeoDiff model loaded and cached in memory.")

    @classmethod
    def get(cls, ckpt_path: str, device: torch.device) -> "_GeoDiffBackend":
        if cls._instance is None:
            cls._instance = cls(ckpt_path=ckpt_path, device=device)
        else:
            if cls._instance.ckpt_path != ckpt_path:
                logger.warning(
                    "GeoDiff singleton already loaded from '%s'; ignoring requested checkpoint '%s'.",
                    cls._instance.ckpt_path,
                    ckpt_path,
                )
            if cls._instance.device != device:
                logger.warning(
                    "GeoDiff singleton already on %s; ignoring requested device %s.",
                    cls._instance.device,
                    device,
                )
        return cls._instance


class GeoDiffDistance(BaseScoreComponent):
    """
    Scoring component that computes raw embedding-space distance between each
    query molecule and a reference SMILES using a pretrained GeoDiff model.
    """

    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

        specific = self.parameters.specific_parameters
        self._ckpt_path: str = str(specific.get("ckpt_path", _DEFAULT_CKPT_PATH))
        self._reference_smiles: str = str(specific["reference_smiles"])
        self._embedding_mode: str = str(specific.get("embedding_mode", "concat"))
        self._distance_metric: str = str(specific.get("distance_metric", "cosine")).lower()
        self._concat_weight: float = float(specific.get("concat_weight", 0.5))
        self._batch_size: int = int(specific.get("batch_size", 64))
        self._seed: int = int(specific.get("seed", 42))

        device_str: str = str(specific.get("device", "auto"))
        if device_str == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device_str)

        if self._embedding_mode not in ("2D", "3D", "concat", "concat_norm", "weight_concat"):
            raise ValueError(
                f"embedding_mode must be '2D', '3D', 'concat', 'concat_norm', or 'weight_concat', got '{self._embedding_mode}'"
            )
        if self._distance_metric not in ("cosine", "euclidean"):
            raise ValueError(
                f"distance_metric must be 'cosine' or 'euclidean', got '{self._distance_metric}'"
            )
        if self._batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self._batch_size}")

        reference_smiles_canonical = canonicalize_smiles(self._reference_smiles)
        if reference_smiles_canonical is None:
            raise ValueError(f"Invalid reference_smiles: {self._reference_smiles!r}")
        self._reference_smiles = reference_smiles_canonical

        self._backend = _GeoDiffBackend.get(
            ckpt_path=self._ckpt_path,
            device=self._device,
        )

        self._reference_embedding = self._embed_smiles([self._reference_smiles])[0]

    # Public API
    def calculate_score(self, molecules: List) -> ComponentSummary:
        raw_distances = self._calculate_distances(molecules)

        specific = self.parameters.specific_parameters
        transform_params = specific.get(ComponentSpecificParametersEnum.TRANSFORMATION, {})
        if transform_params:
            from reinvent_scoring.scoring.score_transformations import TransformationFactory

            transform_fn = TransformationFactory().get_transformation_function(transform_params)
            scores = transform_fn(raw_distances, transform_params)
        else:
            scores = raw_distances

        return ComponentSummary(
            total_score=scores,
            raw_score=raw_distances,
            parameters=self.parameters,
        )

    @staticmethod
    def _mol_to_smiles(mol) -> Optional[str]:
        if mol is None:
            return None
        from rdkit import Chem

        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None
        return smi if smi else None

    def _embed_smiles(self, smiles_list: List[Optional[str]]) -> np.ndarray:
        valid_positions = []
        valid_smiles = []
        valid_mols = []

        for i, smiles in enumerate(smiles_list):
            if smiles is None:
                continue
            smi = canonicalize_smiles(smiles)
            if smi is None:
                continue
            try:
                mol = smiles_to_mol(smi, seed=self._seed)
            except Exception as e:
                logger.warning("Failed to build conformer for SMILES '%s': %s", smiles, e)
                mol = None
            if mol is None:
                continue

            valid_positions.append(i)
            valid_smiles.append(smi)
            valid_mols.append(mol)

        if not valid_positions:
            raise ValueError("No valid SMILES available for embedding.")

        all_embeddings = []
        for start in range(0, len(valid_mols), self._batch_size):
            mol_chunk = valid_mols[start:start + self._batch_size]
            smiles_chunk = valid_smiles[start:start + self._batch_size]
            emb_chunk = geodiff_embed_batch(
                model=self._backend.model,
                transforms=self._backend.transforms,
                mols=mol_chunk,
                smiles_list=smiles_chunk,
                device=self._backend.device,
                mode=self._embedding_mode,
            )
            all_embeddings.append(emb_chunk)

        valid_embeddings = np.vstack(all_embeddings).astype(np.float32)
        emb_dim = valid_embeddings.shape[1]
        result = np.zeros((len(smiles_list), emb_dim), dtype=np.float32)
        for pos, emb in zip(valid_positions, valid_embeddings):
            result[pos] = emb
        return result

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self._distance_metric == "cosine":
            return cosine_distance(a, b)
        return euclidean_distance(a, b)

    def _calculate_distances(self, query_mols: List) -> np.ndarray:
        n = len(query_mols)
        smiles_list = [self._mol_to_smiles(mol) for mol in query_mols]

        valid_indices = [i for i, s in enumerate(smiles_list) if s is not None]
        if not valid_indices:
            return np.zeros(n, dtype=np.float32)

        valid_smiles = [smiles_list[i] for i in valid_indices]
        try:
            valid_embeddings = self._embed_smiles(valid_smiles)
        except Exception as e:
            logger.warning("Failed to embed query batch: %s", e)
            return np.zeros(n, dtype=np.float32)

        if self._embedding_mode == "weight_concat":
            distances_2d = np.array(
                [self._distance(emb[:len(emb) // 2], self._reference_embedding[:len(self._reference_embedding) // 2]) for emb in valid_embeddings],
                dtype=np.float32,
            )
            distances_2d = sigmoid_transform(distances_2d, low=0.1, high=1.2, k=-0.25)
            distances_3d = np.array(
                [self._distance(emb[len(emb) // 2:], self._reference_embedding[len(self._reference_embedding) // 2:]) for emb in valid_embeddings],
                dtype=np.float32,
            )
            distances_3d = sigmoid_transform(distances_3d, low=1, high=10, k=-0.25)
            distances = (1 - self._concat_weight) * distances_2d + self._concat_weight * distances_3d
        else:
            distances = np.array(
                [self._distance(emb, self._reference_embedding) for emb in valid_embeddings],
                dtype=np.float32,
            )

        result = np.zeros(n, dtype=np.float32)
        for idx, dist in zip(valid_indices, distances):
            result[idx] = dist
        return result
