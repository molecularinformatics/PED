import os
import re
import json
import pickle
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REFS_CSV = os.path.join(REPO_ROOT, "data", "lit-pcba-reference.csv")


def load_reference_table(refs_csv: str, target: str) -> pd.DataFrame:
    """Reference ligands for one target from the curated SMILES table."""
    table = pd.read_csv(refs_csv)
    missing = {"target", "pdb_id", "ligand_id", "smiles"} - set(table.columns)
    if missing:
        raise ValueError(f"{refs_csv} is missing columns {sorted(missing)}")
    rows = table[table["target"].astype(str).str.strip() == target].copy()
    if len(rows) == 0:
        raise ValueError(f"{refs_csv} has no rows for target {target!r}")
    rows["ref_name"] = rows["pdb_id"].astype(str).str.strip() + "_ligand"
    rows["raw_smiles"] = rows["smiles"]
    rows["smiles"] = rows["smiles"].map(canonicalize_smiles)
    if rows["smiles"].isna().any():
        bad = rows.loc[rows["smiles"].isna(), "pdb_id"].tolist()
        raise ValueError(f"{refs_csv}: canonicalization failed for {bad}")
    rows["mol2_path"] = ""
    rows["input_type"] = "refs_csv"
    return rows[["ref_name", "pdb_id", "smiles", "raw_smiles", "mol2_path", "input_type"]]


def sanitize_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def read_smi_first_column(path: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 0:
                continue
            smiles = parts[0]
            sid = parts[1] if len(parts) > 1 else None
            rows.append(
                {
                    "raw_smiles": smiles,
                    "sid": sid,
                    "line_idx": line_idx,
                }
            )
    return pd.DataFrame(rows)


def mol2_to_smiles(mol2_path: str) -> str:
    # mol = Chem.MolFromMol2File(mol2_path)
    mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to parse MOL2 file: {mol2_path}")

    # Return raw SMILES here
    smiles = Chem.MolToSmiles(mol, canonical=False)
    if not smiles:
        raise ValueError(f"Failed to convert MOL2 to SMILES: {mol2_path}")
    return smiles


def find_reference_ligands(target_dir: str) -> List[str]:
    paths = sorted(glob(os.path.join(target_dir, "*_ligand.mol2")))
    return paths


def prepare_target_inputs(
    target_dir: str,
    out_dir: str,
    refs_csv: Optional[str] = DEFAULT_REFS_CSV,
    target: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Create refs.csv, library.csv and prep_meta.json under out_dir.

    References come from the curated SMILES table by default; pass refs_csv=None to
    read them from the target folder's .mol2 files instead.
    """
    os.makedirs(out_dir, exist_ok=True)
    target = target or Path(os.path.abspath(target_dir)).name
    reference_source = refs_csv or "mol2"

    active_path = os.path.join(target_dir, "actives.smi")
    inactive_path = os.path.join(target_dir, "inactives.smi")
    if not os.path.exists(active_path):
        raise FileNotFoundError(f"Missing actives.smi under {target_dir}")
    if not os.path.exists(inactive_path):
        raise FileNotFoundError(f"Missing inactives.smi under {target_dir}")

    ref_rows: List[Dict[str, object]] = []
    failed_refs: List[Dict[str, str]] = []
    refs_from_csv = None
    if refs_csv:
        refs_from_csv = load_reference_table(refs_csv, target)
        ref_rows = refs_from_csv.to_dict("records")
        ligand_paths = []
    else:
        ligand_paths = find_reference_ligands(target_dir)
        if len(ligand_paths) == 0:
            raise FileNotFoundError(f"No *_ligand.mol2 files found under {target_dir}")

    for lig_path in ligand_paths:
        ref_name = Path(lig_path).stem
        pdb_id = ref_name.replace("_ligand", "")
        try:
            raw_smiles = mol2_to_smiles(lig_path)
            smiles = canonicalize_smiles(raw_smiles)
            if smiles is None:
                raise ValueError("Canonicalization failed")
            ref_rows.append(
                {
                    "ref_name": ref_name,
                    "pdb_id": pdb_id,
                    "mol2_path": os.path.abspath(lig_path),
                    "raw_smiles": raw_smiles,
                    "smiles": smiles,
                    "input_type": "mol2",
                }
            )
        except Exception as exc:
            failed_refs.append({"mol2_path": os.path.abspath(lig_path), "error": str(exc)})

    if len(ref_rows) == 0:
        raise RuntimeError(f"All reference ligands failed for {target_dir}")

    act_df = read_smi_first_column(active_path)
    act_df["label"] = 1
    act_df["source"] = "active"

    inact_df = read_smi_first_column(inactive_path)
    inact_df["label"] = 0
    inact_df["source"] = "inactive"

    lib_df = pd.concat([act_df, inact_df], axis=0, ignore_index=True)
    lib_df["smiles"] = lib_df["raw_smiles"].map(canonicalize_smiles)
    lib_df["row_id"] = np.arange(len(lib_df), dtype=np.int64)

    failed_lib_df = lib_df[lib_df["smiles"].isna()].copy()
    lib_df = lib_df[lib_df["smiles"].notna()].copy()
    lib_df["sid"] = lib_df["sid"].astype(str)
    lib_df = lib_df[["row_id", "smiles", "raw_smiles", "sid", "label", "source", "line_idx"]]

    refs_df = pd.DataFrame(ref_rows)
    refs_df = refs_df[["ref_name", "pdb_id", "smiles", "raw_smiles", "mol2_path", "input_type"]]

    # Drop duplicates by smiles (keep first occurrence)
    refs_df_orig_len = len(refs_df)
    refs_df = refs_df.drop_duplicates(subset=["smiles"], keep="first").copy()
    refs_df["ref_idx"] = np.arange(len(refs_df), dtype=np.int64)

    lib_df_orig_len = len(lib_df)
    lib_df = lib_df.drop_duplicates(subset=["smiles"], keep="first").copy()
    lib_df["row_id"] = np.arange(len(lib_df), dtype=np.int64)
    lib_df["input_type"] = "smi"
    lib_df = lib_df[["row_id", "smiles", "raw_smiles", "sid", "label", "source", "line_idx", "input_type"]]

    refs_df = refs_df[["ref_idx", "ref_name", "pdb_id", "smiles", "raw_smiles", "mol2_path", "input_type"]]

    refs_csv = os.path.join(out_dir, "refs.csv")
    lib_csv = os.path.join(out_dir, "library.csv")
    refs_df.to_csv(refs_csv, index=False)
    lib_df.to_csv(lib_csv, index=False)

    meta = {
        "target": target,
        "target_dir": os.path.abspath(target_dir),
        "reference_source": reference_source,
        "n_reference_ligands_before_dedup": int(refs_df_orig_len),
        "n_reference_ligands": int(len(refs_df)),
        "n_reference_ligands_duplicates_removed": int(refs_df_orig_len - len(refs_df)),
        "n_library_rows_before_dedup": int(lib_df_orig_len),
        "n_library_rows_kept": int(len(lib_df)),
        "n_library_rows_duplicates_removed": int(lib_df_orig_len - len(lib_df)),
        "n_library_rows_failed_smiles": int(len(failed_lib_df)),
        "n_actives": int((lib_df["label"] == 1).sum()),
        "n_inactives": int((lib_df["label"] == 0).sum()),
        "failed_reference_ligands": failed_refs,
    }
    with open(os.path.join(out_dir, "prep_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if len(failed_lib_df) > 0:
        failed_lib_df.to_csv(os.path.join(out_dir, "library_failed_smiles.csv"), index=False)
    if len(failed_refs) > 0:
        pd.DataFrame(failed_refs).to_csv(os.path.join(out_dir, "refs_failed.csv"), index=False)

    return refs_csv, lib_csv, os.path.join(out_dir, "prep_meta.json")


def smiles_to_conformers(smiles: str, num_confs: int, seed: int):
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return []

    mol0 = Chem.AddHs(mol0)

    def set_embed_attempts(params, n):
        if hasattr(params, "maxIterations"):
            params.maxIterations = n
        elif hasattr(params, "maxAttempts"):
            params.maxAttempts = n

    def try_embed(params):
        params.randomSeed = int(seed) & 0xFFFFFFFF
        params.numThreads = 1
        conf_ids = list(AllChem.EmbedMultipleConfs(mol0, numConfs=num_confs, params=params))
        if len(conf_ids) == 0:
            return None
        return conf_ids

    conf_ids = try_embed(AllChem.ETKDGv3())

    if conf_ids is None:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        set_embed_attempts(params, 1000)
        conf_ids = try_embed(params)

    if conf_ids is None:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        set_embed_attempts(params, 2000)
        params.enforceChirality = False
        conf_ids = try_embed(params)

    if conf_ids is None:
        params = AllChem.ETKDGv2()
        params.useRandomCoords = True
        set_embed_attempts(params, 2000)
        conf_ids = try_embed(params)

    if conf_ids is None:
        return []

    for cid in conf_ids:
        try:
            AllChem.UFFOptimizeMolecule(mol0, confId=cid, maxIters=200)
        except Exception:
            pass

    out = []
    for cid in conf_ids:
        m = Chem.Mol(mol0)
        conf = Chem.Conformer(mol0.GetConformer(cid))
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        out.append(m)
    return out


def mol2_to_mols(mol2_path: str, smiles: str = None):
    mol = None

    try:
        # mol = Chem.MolFromMol2File(mol2_path, sanitize=True, removeHs=False)
        mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
    except Exception:
        mol = None

    if mol is None:
        try:
            mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
            if mol is not None:
                Chem.SanitizeMol(mol)
        except Exception:
            mol = None

    if mol is None:
        print(f"MolFromMol2File failed: {mol2_path}")
        return []

    if mol.GetNumConformers() == 0:
        print(f"No conformer found in MOL2: {mol2_path}")
        return []

    if mol.GetNumAtoms() == 0:
        print(f"Empty molecule in MOL2: {mol2_path}")
        return []

    # Keep the original 3D conformer from MOL2.
    # Only add hydrogens with coordinates if the file does not already contain H atoms.
    has_h = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    if not has_h:
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass

    # Store a single-conformer copy to match the output style of smiles_to_conformers
    out = []
    for cid in range(mol.GetNumConformers()):
        m = Chem.Mol(mol)
        conf = Chem.Conformer(mol.GetConformer(cid))
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        out.append(m)

    return out


def row_to_mols(row, ref_confs: int, seed: int):
    input_type = row["input_type"]

    smi = str(row["smiles"]).strip()
    return smiles_to_conformers(smi, num_confs=ref_confs, seed=seed)

    # if input_type == "smi":
    #     smi = str(row["smiles"]).strip()
    #     return smiles_to_conformers(smi, num_confs=ref_confs, seed=seed)

    # if input_type == "mol2":
    #     mol2_path = str(row["mol2_path"]).strip()
    #     smi = str(row["smiles"]).strip()
    #     return mol2_to_mols(mol2_path, smiles=smi)

    raise ValueError(f"Unsupported input_type: {input_type}")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, eps, None)


def load_smiles_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    return np.array([str(x).strip() for x in arr], dtype=object)


def load_emb_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D array, but got shape {arr.shape}")
    return arr.astype(np.float32)


def cosine_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_n = l2_normalize(x)
    y_n = l2_normalize(y)
    sim = x_n @ y_n.T
    return 1.0 - sim


def euclidean_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    sq = np.clip(x2 + y2 - 2.0 * (x @ y.T), 0.0, None)
    return np.sqrt(sq, dtype=np.float32)


def concat_normalized_embeddings(emb2d: np.ndarray, emb3d: np.ndarray) -> np.ndarray:
    return np.concatenate([l2_normalize(emb2d), l2_normalize(emb3d)], axis=1).astype(np.float32)


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_pickle(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
