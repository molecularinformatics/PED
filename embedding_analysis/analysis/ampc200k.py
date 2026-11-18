"""
Loader for the AmpC 200k correlation study (the data behind Figure 2).

Everything the geometry and scale analyses need lives on one set of 199,770
molecules with one reference ligand, so every embedding space is measured on
exactly the same chemistry.  That is what makes the spaces comparable.

Three row keyings have to be reconciled:

  * ``similarity_distance_200k.csv``  -- keyed by the original (non-canonical)
    AmpC SMILES, holding the ROCS and ECFP4 similarities.
  * ``library/embeddings/*/smiles.npy`` -- keyed by RDKit-CANONICAL SMILES,
    199,770 rows.  ``geodiff_2D`` and ``geodiff_3D`` are stored in a different
    row order from the other folders, so every space is indexed by name, never
    by position.
  * ``raw_canonical_distances.csv`` -- carries both spellings and is the join key.

``aligned()`` returns a frame of the traditional similarities plus, for each
space, the row index into that space's ``emb.npy``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Repo-local scratch for pipeline outputs; see .gitignore.
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = str(REPO_ROOT / "results" / "ampc200k")

# Traditional similarity columns in similarity_distance_200k.csv.
SIMILARITY_COLS = {
    "ROCS combination": "3d_similarity_ROCS",
    "ROCS shape": "ShapeTanimoto",
    "ROCS color": "ColorTanimoto",
    "ECFP4 Tanimoto": "ecfp_tanimoto",
}

# (label, folder(s)).  A tuple of two folders means the paper's "concat" mode:
# each branch L2-normalised, then concatenated.
SPACES: Dict[str, object] = {
    "GeoDiff 2D":        "geodiff_2D",
    "GeoDiff 3D":        "geodiff_3D",
    "GeoDiff concat":    ("geodiff_2D", "geodiff_3D"),
    "MoLFormer":         "molformer",
    "MoleculeSDE 2D":    "sde_2D",
    "MoleculeSDE 3D":    "sde_3D",
    "MoleculeSDE concat": ("sde_2D", "sde_3D"),
    "Uni-Mol2 164m CLS":  "unimol_164m_cls",
    "Uni-Mol2 164m atom": "unimol_164m_atom",
    "Uni-Mol2 84m CLS":   "unimol_84m_cls",
    "Uni-Mol2 84m atom":  "unimol_84m_atom",
    "Uni-Mol2 1.1B":      "unimol_1.1B",
}

# Which spaces the manuscript actually reports, for marking in the output.
#
# Uni-Mol2 is the 164M checkpoint with ATOM-wise mean pooling of the last hidden
# states (manuscript Section 2 and Appendix "setting"), not the CLS token.  The two
# are not interchangeable: atom-pooled reproduces the Figure 2 column exactly
# (r = -0.027 against ROCS shape), whereas the CLS
# variant gives +0.180 on the same molecules.
REPORTED = {"GeoDiff 2D", "GeoDiff 3D", "GeoDiff concat", "MoLFormer",
            "MoleculeSDE 2D", "MoleculeSDE 3D", "Uni-Mol2 164m atom"}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


@functools.lru_cache(maxsize=None)
def _space_smiles(folder: str) -> np.ndarray:
    p = os.path.join(ROOT, "library", "embeddings", folder, "smiles.npy")
    return np.array([str(v) for v in np.load(p, allow_pickle=True)])


@functools.lru_cache(maxsize=1)
def aligned() -> pd.DataFrame:
    """Traditional similarities plus a row index into each embedding folder."""
    sim = pd.read_csv(os.path.join(ROOT, "similarity_distance_200k.csv"),
                      usecols=["smiles"] + list(SIMILARITY_COLS.values()))
    key = pd.read_csv(os.path.join(ROOT, "raw_canonical_distances.csv"),
                      usecols=["raw_smiles", "canonical_smiles"])

    # The similarity table has 200,000 rows for 199,770 distinct molecules: 230
    # SMILES appear twice.  Keep the first occurrence so the join is one-to-one and
    # no molecule is silently up-weighted in a correlation.
    sim = sim.drop_duplicates(subset="smiles", keep="first")

    df = sim.merge(key, left_on="smiles", right_on="raw_smiles", how="inner")
    if len(df) != len(key):
        raise RuntimeError(f"join lost rows: {len(df)} vs {len(key)}")

    folders = sorted({f for v in SPACES.values()
                      for f in ((v,) if isinstance(v, str) else v)} | {"ecfp"})
    for folder in folders:
        pos = pd.Series(np.arange(len(_space_smiles(folder))),
                        index=_space_smiles(folder))
        df[f"idx__{folder}"] = pos.reindex(df.canonical_smiles).to_numpy()
        if df[f"idx__{folder}"].isna().any():
            raise RuntimeError(f"{folder}: some canonical SMILES missing")
        df[f"idx__{folder}"] = df[f"idx__{folder}"].astype(int)
    return df.reset_index(drop=True)


def subsample(n: int, seed: int = 0) -> np.ndarray:
    """Row positions into ``aligned()``; the same draw for every space."""
    df = aligned()
    rng = np.random.default_rng(seed)
    if n >= len(df):
        return np.arange(len(df))
    return np.sort(rng.choice(len(df), size=n, replace=False))


def load_library(space: str, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Library embeddings for ``space``, in ``aligned()`` row order."""
    df = aligned()
    rows = np.arange(len(df)) if rows is None else rows
    spec = SPACES[space]
    parts = (spec,) if isinstance(spec, str) else spec
    out = []
    for folder in parts:
        arr = np.load(os.path.join(ROOT, "library", "embeddings", folder, "emb.npy"),
                      mmap_mode="r")
        idx = df[f"idx__{folder}"].to_numpy()[rows]
        block = np.asarray(arr[idx], dtype=np.float32)
        out.append(l2_normalize(block) if len(parts) > 1 else block)
    return out[0] if len(out) == 1 else np.concatenate(out, axis=1).astype(np.float32)


def load_reference(space: str) -> np.ndarray:
    """The single reference ligand's embedding, built the same way."""
    spec = SPACES[space]
    parts = (spec,) if isinstance(spec, str) else spec
    out = []
    for folder in parts:
        v = np.load(os.path.join(ROOT, "refs", "embeddings", folder, "emb.npy")).astype(np.float32)
        out.append(l2_normalize(v) if len(parts) > 1 else v)
    v = out[0] if len(out) == 1 else np.concatenate(out, axis=1)
    return np.asarray(v, dtype=np.float32).reshape(-1)


def load_ecfp(rows: Optional[np.ndarray] = None) -> np.ndarray:
    df = aligned()
    rows = np.arange(len(df)) if rows is None else rows
    arr = np.load(os.path.join(ROOT, "library", "embeddings", "ecfp", "emb.npy"),
                  mmap_mode="r")
    return np.asarray(arr[df["idx__ecfp"].to_numpy()[rows]], dtype=np.float32)
