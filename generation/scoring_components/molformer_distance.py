import logging
import numpy as np
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModel

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
logger = logging.getLogger(__name__)

_MOLFORMER_MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"


# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, reused for the entire process lifetime
# ---------------------------------------------------------------------------

class _MolformerBackend:
    """
    Holds the tokenizer and model as a process-wide singleton.

    The first call to ``get()`` loads the model onto the target device; every
    subsequent call returns the same cached instance regardless of how many
    ``MolformerDistance`` objects are created.  This ensures GPU memory is
    allocated exactly once for the full RL run.
    """

    _instance: Optional["_MolformerBackend"] = None

    def __init__(self, model_name: str, device: torch.device):
        logger.info(
            "Loading MoLFormer model '%s' onto %s (one-time cost).", model_name, device
        )
        self.model_name = model_name
        self.device = device

        # MoLFormer's modeling code calls torch.linalg.qr, which was added in
        # PyTorch 1.9.  On older builds the attribute is missing, so patch it
        # onto the linalg namespace before the model file is imported.
        if not hasattr(torch.linalg, "qr"):
            logger.warning(
                "torch.linalg.qr not found (torch %s); patching with torch.qr.",
                torch.__version__,
            )
            torch.linalg.qr = lambda x, mode="reduced": torch.qr(x)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=True)
            .to(device)
            .eval()
        )
        logger.info("MoLFormer model loaded and cached in memory.")

    @classmethod
    def get(cls, model_name: str, device: torch.device) -> "_MolformerBackend":
        """
        Return the singleton, creating it on the first call.

        If a different ``model_name`` or ``device`` is requested after the
        singleton is already initialised, a warning is logged and the existing
        instance is returned unchanged — reloading mid-run would defeat the
        purpose of the singleton.
        """
        if cls._instance is None:
            cls._instance = cls(model_name, device)
        else:
            if cls._instance.model_name != model_name:
                logger.warning(
                    "MoLFormer singleton already loaded as '%s'; "
                    "ignoring requested model '%s'.",
                    cls._instance.model_name,
                    model_name,
                )
            if cls._instance.device != device:
                logger.warning(
                    "MoLFormer singleton already on %s; "
                    "ignoring requested device %s.",
                    cls._instance.device,
                    device,
                )
        return cls._instance


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine_similarity) between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1-D vectors."""
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Scoring component
# ---------------------------------------------------------------------------

class MolformerDistance(BaseScoreComponent):
    """
    Scoring component that computes raw embedding-space distance between each
    query molecule and a reference SMILES using IBM's MoLFormer model.

    The MoLFormer model is loaded **once** for the entire process and kept
    resident in GPU/CPU memory for the full RL run.  Constructing multiple
    ``MolformerDistance`` instances (e.g. for different references) does NOT
    reload the model.

    Returns RAW DISTANCE (no transformation)
    ----------------------------------------
    The "score" returned is actually the **raw distance** value:
      - Cosine distance: [0, 2]  where 0=identical, 2=opposite direction
      - Euclidean distance: [0, ∞)  where 0=identical

    This is NOT a [0,1] score. Apply transformations (exp(-d/scale), sigmoid,
    etc.) downstream in your scoring function aggregation if needed.

    Specific parameters
    -------------------
    reference_smiles : str
        SMILES string of the reference molecule.
    distance_metric : str, optional
        "cosine" (default) or "euclidean".
    model_name : str, optional
        HuggingFace model identifier
        (default: "ibm/MoLFormer-XL-both-10pct").
    batch_size : int, optional
        SMILES per forward pass (default: 64).
    device : str, optional
        "cuda", "cpu", or "auto" (default; selects CUDA when available).
    """

    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

        specific = self.parameters.specific_parameters

        self._reference_smiles: str = specific["reference_smiles"]
        self._distance_metric: str = specific.get("distance_metric", "cosine").lower()
        self._batch_size: int = int(specific.get("batch_size", 64))

        model_name: str = specific.get("model_name", _MOLFORMER_MODEL_NAME)

        device_str: str = specific.get("device", "auto")
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        if self._distance_metric not in ("cosine", "euclidean"):
            raise ValueError(
                f"distance_metric must be 'cosine' or 'euclidean', "
                f"got '{self._distance_metric}'"
            )

        # Acquire the singleton backend — loads the model only on first call
        self._backend: _MolformerBackend = _MolformerBackend.get(model_name, device)

        # Pre-compute the reference embedding once at construction time
        self._reference_embedding: np.ndarray = self._embed_smiles(
            [self._reference_smiles]
        )[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        return ComponentSummary(total_score=scores, raw_score=raw_distances, parameters=self.parameters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mol_to_smiles(mol) -> Optional[str]:
        """Convert an RDKit mol to canonical SMILES, or None on failure."""
        if mol is None:
            return None
        from rdkit import Chem
        smi = Chem.MolToSmiles(mol, canonical=True)
        return smi if smi else None

    def _embed_smiles(self, smiles_list: List[Optional[str]]) -> np.ndarray:
        """
        Embed SMILES strings using the cached MoLFormer model.

        Returns an (N, hidden_dim) float32 array of mean-pooled embeddings.
        None entries are replaced by zero vectors.
        """
        tokenizer = self._backend.tokenizer
        model = self._backend.model
        device = self._backend.device

        embeddings = []

        for start in range(0, len(smiles_list), self._batch_size):
            batch = smiles_list[start: start + self._batch_size]

            # Replace None with a dummy so the tokenizer doesn't fail;
            # those slots are zeroed out afterwards.
            valid_mask = [s is not None for s in batch]
            safe_batch = [s if s is not None else "C" for s in batch]

            tokens = tokenizer(
                safe_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**tokens)

            # Mean-pool over token dimension weighted by attention mask
            attention_mask = tokens["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            token_embeddings = outputs.last_hidden_state                     # (B, T, H)
            summed = (token_embeddings * attention_mask).sum(dim=1)          # (B, H)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)               # (B, 1)
            batch_embeddings = (summed / counts).cpu().numpy()                # (B, H)

            # Zero out slots that had no valid SMILES
            for i, valid in enumerate(valid_mask):
                if not valid:
                    batch_embeddings[i] = 0.0

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings).astype(np.float32)

    def _calculate_distances(self, query_mols: List) -> np.ndarray:
        """
        Compute raw distances (no transformation).

        Returns:
          np.ndarray of distances, where 0 = failures/invalid molecules.
        """
        n = len(query_mols)
        smiles_list = [self._mol_to_smiles(mol) for mol in query_mols]

        valid_indices = [i for i, s in enumerate(smiles_list) if s is not None]
        if not valid_indices:
            return np.zeros(n, dtype=np.float32)

        valid_smiles = [smiles_list[i] for i in valid_indices]
        valid_embeddings = self._embed_smiles(valid_smiles)  # (V, H)

        ref = self._reference_embedding  # (H,)

        if self._distance_metric == "cosine":
            distances = np.array(
                [_cosine_distance(emb, ref) for emb in valid_embeddings],
                dtype=np.float32,
            )
        else:  # euclidean
            distances = np.array(
                [_euclidean_distance(emb, ref) for emb in valid_embeddings],
                dtype=np.float32,
            )

        # Place distances back into the full-length array (0.0 for invalid mols)
        result = np.zeros(n, dtype=np.float32)
        for idx, dist in zip(valid_indices, distances):
            result[idx] = dist

        return result