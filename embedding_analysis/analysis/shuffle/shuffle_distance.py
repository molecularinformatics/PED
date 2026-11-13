import os
import math
import json
import pickle
import random
import argparse
from glob import glob
from typing import List, Optional, Tuple

import yaml
import torch
import numpy as np
import pandas as pd
from easydict import EasyDict
from tqdm import tqdm
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from transformers import AutoModel, AutoTokenizer

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

# ===== GeoDiff imports =====
from geodiff.utils.datasets import rdmol_to_data
from geodiff.utils.misc import seed_all
from geodiff.models.epsnet import get_model
from geodiff.utils.transforms import Compose, CountNodesPerGraph, AddHigherOrderEdges

from pathlib import Path

# Repo-local scratch for pipeline outputs; see .gitignore.
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"


# =========================================================
# basic utils
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def randomize_smiles_once(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        can = Chem.MolToSmiles(mol, canonical=True)
        mol_can = Chem.MolFromSmiles(can)
        if mol_can is None:
            return None

        n_atoms = mol_can.GetNumAtoms()
        atom_order = list(range(n_atoms))
        random.shuffle(atom_order)

        mol_perm = rdmolops.RenumberAtoms(mol_can, atom_order)

        # non-canonical, so this really tests equivalent SMILES enumeration
        shuffled = Chem.MolToSmiles(mol_perm, canonical=False)
        return shuffled
    except Exception:
        return None


def randomize_smiles_n_times(smiles: str, shuffle_time: int) -> Optional[str]:
    if shuffle_time < 0:
        raise ValueError("shuffle_time must be >= 0")

    current = canonicalize_smiles(smiles)
    if current is None:
        return None

    if shuffle_time == 0:
        return current

    for _ in range(shuffle_time):
        current = randomize_smiles_once(current)
        if current is None:
            return None
    return current


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, eps, None)


def cosine_distance_matrixwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_safe = l2_normalize(a)
    b_safe = l2_normalize(b)
    cos_sim = np.sum(a_safe * b_safe, axis=1)
    return (1.0 - cos_sim).astype(np.float32)


def euclidean_distance_matrixwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1).astype(np.float32)


def mean_pool_last_hidden(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    mean = summed / lengths
    return mean.detach().cpu().numpy().astype(np.float32)


# =========================================================
# sampling
# =========================================================

def uniform_bin_sample(
    df: pd.DataFrame,
    score_col: str,
    total_n: int = 10000,
    bin_width: float = 0.2,
    seed: int = 42
) -> pd.DataFrame:
    df = df.copy()
    valid_mask = df[score_col].notna() & np.isfinite(df[score_col].values)
    df = df.loc[valid_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows found for {score_col}")

    min_val = math.floor(df[score_col].min() / bin_width) * bin_width
    max_val = math.ceil(df[score_col].max() / bin_width) * bin_width

    if min_val == max_val:
        return df.sample(n=min(total_n, len(df)), random_state=seed).reset_index(drop=True)

    bins = np.arange(min_val, max_val + bin_width + 1e-8, bin_width)
    df["bin"] = pd.cut(df[score_col], bins=bins, right=False, include_lowest=True)

    bin_groups = {b: g for b, g in df.groupby("bin") if len(g) > 0}
    non_empty_bins = list(bin_groups.keys())
    if len(non_empty_bins) == 0:
        raise ValueError("No non-empty bins after binning")

    rng = np.random.RandomState(seed)
    base_quota = total_n // len(non_empty_bins)

    sampled_parts = []
    remaining_pool_parts = []

    for b in non_empty_bins:
        g = bin_groups[b]
        take = min(base_quota, len(g))
        idx = rng.choice(g.index.values, size=take, replace=False)
        sampled_parts.append(df.loc[idx])

        remaining = g.drop(index=idx)
        if len(remaining) > 0:
            remaining_pool_parts.append(remaining)

    sampled_df = pd.concat(sampled_parts, axis=0).copy()
    leftover = total_n - len(sampled_df)

    if leftover > 0 and remaining_pool_parts:
        remaining_pool = pd.concat(remaining_pool_parts, axis=0).copy()
        if len(remaining_pool) <= leftover:
            sampled_df = pd.concat([sampled_df, remaining_pool], axis=0)
        else:
            extra_idx = rng.choice(remaining_pool.index.values, size=leftover, replace=False)
            sampled_df = pd.concat([sampled_df, remaining_pool.loc[extra_idx]], axis=0)

    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled_df = sampled_df.drop(columns=["bin"], errors="ignore")

    if len(sampled_df) > total_n:
        sampled_df = sampled_df.iloc[:total_n].copy().reset_index(drop=True)

    return sampled_df


def prepare_sampled_dataframe(
    input_csv: str,
    smiles_col: str = "smiles",
    score_col: str = "3d_similarity_ROCS",
    sample_n: int = 10000,
    bin_width: float = 0.2,
    seed: int = 42
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required_cols = [smiles_col, score_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[[smiles_col, score_col]].copy()
    df["original_smiles_raw"] = df[smiles_col].astype(str)

    sampled = uniform_bin_sample(
        df=df,
        score_col=score_col,
        total_n=sample_n,
        bin_width=bin_width,
        seed=seed
    )

    sampled["original_smiles_canonical"] = sampled["original_smiles_raw"].apply(canonicalize_smiles)
    sampled = sampled[sampled["original_smiles_canonical"].notna()].reset_index(drop=True)

    return sampled


# =========================================================
# MolFormer
# =========================================================

def get_pretrained_transformer(model_name_key: str = "molformer"):
    if model_name_key == "molformer":
        model_name = "ibm/MoLFormer-XL-both-10pct"
        model = AutoModel.from_pretrained(
            model_name,
            deterministic_eval=True,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    elif model_name_key == "chemberta":
        model_name = "DeepChem/ChemBERTa-100M-MLM"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model_name_key: {model_name_key}")

    model.eval()
    return tokenizer, model


def embed_smiles_list(
    smiles_list: List[str],
    tokenizer,
    model,
    batch_size: int = 128,
    device: str = "cuda"
) -> np.ndarray:
    all_embs = []
    model = model.to(device)
    model.eval()

    for start in tqdm(range(0, len(smiles_list), batch_size), desc="MolFormer embedding"):
        batch_smiles = smiles_list[start:start + batch_size]
        with torch.no_grad():
            enc = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = mean_pool_last_hidden(out.last_hidden_state, enc["attention_mask"])
            all_embs.append(emb)

    return np.concatenate(all_embs, axis=0).astype(np.float32)


def save_molformer_outputs(
    setting_dir: str,
    original_smiles_raw: List[str],
    setting_smiles: List[str],
    setting_emb: np.ndarray,
    original_canonical_emb: np.ndarray
):
    out_dir = os.path.join(setting_dir, "molformer")
    ensure_dir(out_dir)

    euclid = euclidean_distance_matrixwise(setting_emb, original_canonical_emb)
    cosine = cosine_distance_matrixwise(setting_emb, original_canonical_emb)

    np.save(os.path.join(out_dir, "smiles.npy"), np.array(setting_smiles, dtype=object))
    np.save(os.path.join(out_dir, "emb.npy"), setting_emb.astype(np.float32))

    results_df = pd.DataFrame({
        "original_smiles_from_csv": original_smiles_raw,
        "smiles_under_this_setting": setting_smiles,
        "molformer_euclidean": euclid,
        "molformer_cosine": cosine,
    })
    results_df.to_csv(os.path.join(out_dir, "molformer_results.csv"), index=False)


# =========================================================
# GeoDiff
# =========================================================

def row_to_mols(row, ref_confs: int, seed: int):
    smi = str(row["smiles"]).strip()
    return smiles_to_conformers(smi, num_confs=ref_confs, seed=seed)


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


@torch.no_grad()
def dual_mol_embedding(model, atom_type, pos, bond_index, bond_type, batch, num_graphs):
    device = pos.device
    time_step = torch.zeros(num_graphs, dtype=torch.long, device=device)

    _, _, edge_index, edge_type, edge_length, local_edge_mask = model(
        atom_type=atom_type,
        pos=pos,
        bond_index=bond_index,
        bond_type=bond_type,
        batch=batch,
        time_step=time_step,
        return_edges=True,
        extend_order=False,
        extend_radius=True,
    )

    edge_attr_global = model.edge_encoder_global(
        edge_length=edge_length,
        edge_type=edge_type
    )

    node_attr_global = model.encoder_global(
        z=atom_type,
        edge_index=edge_index,
        edge_length=edge_length,
        edge_attr=edge_attr_global,
    )
    mol_emb_3d = scatter_mean(node_attr_global, batch, dim=0)

    edge_attr_local = model.edge_encoder_global(
        edge_length=edge_length,
        edge_type=edge_type,
    )
    local_edge_index = edge_index[:, local_edge_mask]
    local_edge_attr = edge_attr_local[local_edge_mask]

    node_attr_local = model.encoder_local(
        z=atom_type,
        edge_index=local_edge_index,
        edge_attr=local_edge_attr,
    )
    mol_emb_2d = scatter_mean(node_attr_local, batch, dim=0)

    return mol_emb_2d, mol_emb_3d


def load_geodiff_model_and_transforms(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
    ymls = glob(os.path.join(ckpt_dir, "*.yml"))
    if len(ymls) == 0:
        raise FileNotFoundError(f"No .yml found in {ckpt_dir}")

    with open(ymls[0], "r", encoding="utf-8") as f:
        config = EasyDict(yaml.safe_load(f))

    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order),
    ])

    model = get_model(ckpt["config"].model)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return model, transforms


def pick_pos_ref(data, which_ref: int = 0):
    num_nodes = int(data.num_nodes)
    total = int(data.pos_ref.size(0))
    num_refs = total // num_nodes
    if num_refs <= 0:
        raise ValueError("pos_ref is empty")
    which = which_ref % num_refs
    pos = data.pos_ref.view(num_refs, num_nodes, 3)[which]
    return pos


def preprocess_one_data(data, transforms, which_ref: int = 0):
    data = transforms(data)
    smiles = getattr(data, "smiles", "")
    if smiles is None:
        smiles = ""

    if hasattr(data, "pos_ref") and data.pos_ref is not None:
        pos = pick_pos_ref(data, which_ref=which_ref)
        data_input = data.clone()
        data_input["pos_ref"] = None
        data_input.pos = pos
    else:
        data_input = data

    return data_input, smiles


def geodiff_pack_serial_from_csv(
    csv_path: str,
    out_pkl: str,
    ref_confs: int = 1,
    seed: int = 42,
    smiles_col: str = "smiles",
):
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"{smiles_col} not found in {csv_path}")

    packed = []
    failed_rows = []

    records = df.to_dict(orient="records")
    for i, row in enumerate(tqdm(records, desc=f"GeoDiff pack: {os.path.basename(csv_path)}")):
        try:
            mols = row_to_mols(
                row=row,
                ref_confs=ref_confs,
                seed=seed + i,
            )
        except Exception:
            mols = None

        if not mols:
            failed_rows.append(i)
            continue

        smiles = str(row[smiles_col]).strip()
        for m in mols:
            d = rdmol_to_data(m, smiles=smiles)
            packed.append(d)

    with open(out_pkl, "wb") as f:
        pickle.dump(packed, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "csv_path": csv_path,
        "out_pkl": out_pkl,
        "n_input_rows": len(df),
        "n_failed_rows": len(failed_rows),
        "n_packed_graphs": len(packed),
        "failed_row_indices": failed_rows[:1000],  # avoid huge json
    }
    with open(out_pkl + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


@torch.no_grad()
def geodiff_embed_from_packed_serial(
    packed_pkl: str,
    out_dir: str,
    ckpt_path: str,
    device: str = "cuda:0",
    batch_size: int = 128,
    which_ref: int = 0,
    seed: int = 2021,
):
    ensure_dir(out_dir)
    seed_all(seed)

    with open(packed_pkl, "rb") as f:
        data_list = pickle.load(f)

    total_n = len(data_list)
    if total_n == 0:
        empty2 = np.zeros((0, 128), dtype=np.float32)
        empty3 = np.zeros((0, 128), dtype=np.float32)
        save_geodiff_embedding_outputs(out_dir, empty2, empty3, [])
        return

    model, transforms = load_geodiff_model_and_transforms(ckpt_path, device)

    emb2d_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "_tmp_emb2d.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(total_n, 128),
    )
    emb3d_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "_tmp_emb3d.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(total_n, 128),
    )

    smiles_out = []
    batch_graphs = []
    batch_smiles = []
    write_idx = 0

    def flush_batch():
        nonlocal batch_graphs, batch_smiles, write_idx
        if len(batch_graphs) == 0:
            return

        pyg_batch = Batch.from_data_list(batch_graphs).to(device)
        mol_emb_2d, mol_emb_3d = dual_mol_embedding(
            model=model,
            atom_type=pyg_batch.atom_type,
            pos=pyg_batch.pos,
            bond_index=pyg_batch.edge_index,
            bond_type=pyg_batch.edge_type,
            batch=pyg_batch.batch,
            num_graphs=pyg_batch.num_graphs,
        )

        emb2d_np = mol_emb_2d.detach().cpu().numpy().astype(np.float32)
        emb3d_np = mol_emb_3d.detach().cpu().numpy().astype(np.float32)
        n = emb2d_np.shape[0]

        emb2d_mm[write_idx:write_idx + n] = emb2d_np
        emb3d_mm[write_idx:write_idx + n] = emb3d_np
        smiles_out.extend(batch_smiles)

        write_idx += n
        batch_graphs = []
        batch_smiles = []

    for data in tqdm(data_list, desc=f"GeoDiff embed: {os.path.basename(out_dir)}"):
        data_input, smiles = preprocess_one_data(data, transforms, which_ref=which_ref)
        batch_graphs.append(data_input)
        batch_smiles.append(smiles)

        if len(batch_graphs) >= batch_size:
            flush_batch()

    flush_batch()

    emb2d = np.array(emb2d_mm)
    emb3d = np.array(emb3d_mm)
    save_geodiff_embedding_outputs(out_dir, emb2d, emb3d, smiles_out)

    for tmp_name in ["_tmp_emb2d.npy", "_tmp_emb3d.npy"]:
        tmp_path = os.path.join(out_dir, tmp_name)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_geodiff_embedding_outputs(out_dir: str, emb2d: np.ndarray, emb3d: np.ndarray, smiles_list: List[str]):
    out_2d = os.path.join(out_dir, "geodiff_2D")
    out_3d = os.path.join(out_dir, "geodiff_3D")
    out_concat = os.path.join(out_dir, "geodiff_concat_norm")

    ensure_dir(out_2d)
    ensure_dir(out_3d)
    ensure_dir(out_concat)

    smiles_arr = np.array(smiles_list, dtype=object)
    concat_norm = np.concatenate([l2_normalize(emb2d), l2_normalize(emb3d)], axis=1).astype(np.float32)

    np.save(os.path.join(out_2d, "emb.npy"), emb2d.astype(np.float32))
    np.save(os.path.join(out_2d, "smiles.npy"), smiles_arr)

    np.save(os.path.join(out_3d, "emb.npy"), emb3d.astype(np.float32))
    np.save(os.path.join(out_3d, "smiles.npy"), smiles_arr)

    np.save(os.path.join(out_concat, "emb.npy"), concat_norm)
    np.save(os.path.join(out_concat, "smiles.npy"), smiles_arr)


def save_geodiff_distance_results(
    setting_dir: str,
    original_smiles_raw: List[str],
    setting_smiles: List[str],
    setting_emb2d: np.ndarray,
    setting_emb3d: np.ndarray,
    original_emb2d: np.ndarray,
    original_emb3d: np.ndarray,
):
    setting_concat = np.concatenate([l2_normalize(setting_emb2d), l2_normalize(setting_emb3d)], axis=1).astype(np.float32)
    original_concat = np.concatenate([l2_normalize(original_emb2d), l2_normalize(original_emb3d)], axis=1).astype(np.float32)

    results_df = pd.DataFrame({
        "original_smiles_from_csv": original_smiles_raw,
        "smiles_under_this_setting": setting_smiles,

        "geodiff_2D_euclidean": euclidean_distance_matrixwise(setting_emb2d, original_emb2d),
        "geodiff_2D_cosine": cosine_distance_matrixwise(setting_emb2d, original_emb2d),

        "geodiff_3D_euclidean": euclidean_distance_matrixwise(setting_emb3d, original_emb3d),
        "geodiff_3D_cosine": cosine_distance_matrixwise(setting_emb3d, original_emb3d),

        "geodiff_concat_norm_euclidean": euclidean_distance_matrixwise(setting_concat, original_concat),
        "geodiff_concat_norm_cosine": cosine_distance_matrixwise(setting_concat, original_concat),
    })
    results_df.to_csv(os.path.join(setting_dir, "geodiff_results.csv"), index=False)


def load_geodiff_embs(setting_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    emb2d = np.load(os.path.join(setting_dir, "geodiff_2D", "emb.npy"), allow_pickle=True)
    emb3d = np.load(os.path.join(setting_dir, "geodiff_3D", "emb.npy"), allow_pickle=True)
    smiles = np.load(os.path.join(setting_dir, "geodiff_2D", "smiles.npy"), allow_pickle=True)
    smiles = [str(x) for x in smiles]
    return emb2d, emb3d, smiles


# =========================================================
# whole pipeline
# =========================================================

def build_setting_smiles(original_raw_smiles: List[str], setting_name: str) -> Tuple[List[str], np.ndarray]:
    if setting_name == "original_canonical":
        smiles = [canonicalize_smiles(s) for s in original_raw_smiles]
    elif setting_name == "original_raw":
        smiles = original_raw_smiles
    elif setting_name == "shuffle_1":
        smiles = [randomize_smiles_n_times(s, 1) for s in tqdm(original_raw_smiles, desc=setting_name)]
    elif setting_name == "shuffle_5":
        smiles = [randomize_smiles_n_times(s, 5) for s in tqdm(original_raw_smiles, desc=setting_name)]
    elif setting_name == "shuffle_10":
        smiles = [randomize_smiles_n_times(s, 10) for s in tqdm(original_raw_smiles, desc=setting_name)]
    else:
        raise ValueError(f"Unknown setting_name: {setting_name}")

    valid_mask = np.array([s is not None for s in smiles], dtype=bool)
    valid_smiles = [s for s in smiles if s is not None]
    return valid_smiles, valid_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(RESULTS / "ampc200k" / "similarity_distance_200k_all_metrics_with2d.csv")
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(RESULTS / "shuffle")
    )
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--score_col", type=str, default="3d_similarity_ROCS")
    parser.add_argument("--sample_n", type=int, default=10000)
    parser.add_argument("--bin_width", type=float, default=0.2)

    parser.add_argument("--run_molformer", action="store_true")
    parser.add_argument("--run_geodiff", action="store_true")

    parser.add_argument("--molformer_batch_size", type=int, default=256)
    parser.add_argument("--molformer_device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="molformer", choices=["molformer", "chemberta"])

    parser.add_argument("--geodiff_ckpt", type=str, default="./geodiff/log/model/checkpoints/drugs_default.pt")
    parser.add_argument("--geodiff_batch_size", type=int, default=256)
    parser.add_argument("--geodiff_device", type=str, default="cuda:0")
    parser.add_argument("--geodiff_ref_confs", type=int, default=1)
    parser.add_argument("--geodiff_which_ref", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.run_molformer and not args.run_geodiff:
        raise ValueError("At least one of --run_molformer or --run_geodiff must be set")

    if args.run_geodiff and not args.geodiff_ckpt:
        raise ValueError("--geodiff_ckpt is required when --run_geodiff is set")

    set_seed(args.seed)
    ensure_dir(args.out_root)

    sampled_csv = os.path.join(args.out_root, "sampled_10k_base.csv")
    if os.path.exists(sampled_csv):
        sampled = pd.read_csv(sampled_csv)
    else:
        sampled = prepare_sampled_dataframe(
            input_csv=args.input_csv,
            smiles_col=args.smiles_col,
            score_col=args.score_col,
            sample_n=args.sample_n,
            bin_width=args.bin_width,
            seed=args.seed
        )
        sampled.to_csv(sampled_csv, index=False)

    print(f"Using sampled dataframe with n={len(sampled)}")

    original_raw_smiles = sampled["original_smiles_raw"].tolist()
    original_canonical_smiles = sampled["original_smiles_canonical"].tolist()

    # ---------------- original canonical references ----------------
    original_canonical_molformer_emb = None
    original_canonical_geodiff_2d = None
    original_canonical_geodiff_3d = None

    if args.run_molformer:
        tokenizer, model = get_pretrained_transformer(model_name_key=args.model_name)

        canonical_setting_dir = os.path.join(args.out_root, "original_canonical")
        ensure_dir(canonical_setting_dir)

        print("MolFormer: original_canonical")
        original_canonical_molformer_emb = embed_smiles_list(
            original_canonical_smiles,
            tokenizer=tokenizer,
            model=model,
            batch_size=args.molformer_batch_size,
            device=args.molformer_device
        )
        save_molformer_outputs(
            canonical_setting_dir,
            original_smiles_raw=original_raw_smiles,
            setting_smiles=original_canonical_smiles,
            setting_emb=original_canonical_molformer_emb,
            original_canonical_emb=original_canonical_molformer_emb
        )

    if args.run_geodiff:
        # canonical_setting_dir = os.path.join(args.out_root, "original_canonical_conf")
        canonical_setting_dir = os.path.join(args.out_root, "original_canonical")
        ensure_dir(canonical_setting_dir)

        canonical_csv = os.path.join(canonical_setting_dir, "_geodiff_input.csv")
        pd.DataFrame({"smiles": original_canonical_smiles}).to_csv(canonical_csv, index=False)

        canonical_pkl = os.path.join(canonical_setting_dir, "_packed.pkl")
        print("GeoDiff pack: original_canonical")
        geodiff_pack_serial_from_csv(
            csv_path=canonical_csv,
            out_pkl=canonical_pkl,
            ref_confs=args.geodiff_ref_confs,
            seed=args.seed,
            # seed=10000,
            smiles_col="smiles",
        )

        print("GeoDiff embed: original_canonical")
        geodiff_embed_from_packed_serial(
            packed_pkl=canonical_pkl,
            out_dir=canonical_setting_dir,
            ckpt_path=args.geodiff_ckpt,
            device=args.geodiff_device,
            batch_size=args.geodiff_batch_size,
            which_ref=args.geodiff_which_ref,
            seed=args.seed,
        )

        # real_canonical_setting_dir = os.path.join(args.out_root, "original_canonical")
        # original_canonical_geodiff_2d, original_canonical_geodiff_3d, _ = load_geodiff_embs(real_canonical_setting_dir)
        # new_canonical_geodiff_2d, new_canonical_geodiff_3d, _ = load_geodiff_embs(canonical_setting_dir)

        # save_geodiff_distance_results(
        #     setting_dir=canonical_setting_dir,
        #     original_smiles_raw=original_raw_smiles,
        #     setting_smiles=original_canonical_smiles,
        #     setting_emb2d=new_canonical_geodiff_2d,
        #     setting_emb3d=new_canonical_geodiff_3d,
        #     original_emb2d=original_canonical_geodiff_2d,
        #     original_emb3d=original_canonical_geodiff_3d,
        # )

        original_canonical_geodiff_2d, original_canonical_geodiff_3d, _ = load_geodiff_embs(canonical_setting_dir)

        save_geodiff_distance_results(
            setting_dir=canonical_setting_dir,
            original_smiles_raw=original_raw_smiles,
            setting_smiles=original_canonical_smiles,
            setting_emb2d=original_canonical_geodiff_2d,
            setting_emb3d=original_canonical_geodiff_3d,
            original_emb2d=original_canonical_geodiff_2d,
            original_emb3d=original_canonical_geodiff_3d,
        )

    # ---------------- other settings ----------------
    setting_names = ["original_raw", "shuffle_1", "shuffle_5", "shuffle_10"]

    for setting_name in setting_names:
        print(f"\n===== Processing {setting_name} =====")
        setting_dir = os.path.join(args.out_root, setting_name)
        ensure_dir(setting_dir)

        valid_smiles, valid_mask = build_setting_smiles(original_raw_smiles, setting_name)
        kept_idx = np.where(valid_mask)[0]

        if len(kept_idx) == 0:
            print(f"[Warning] No valid molecules for {setting_name}, skip.")
            continue

        original_raw_valid = [original_raw_smiles[i] for i in kept_idx]
        original_canonical_smiles_valid = [original_canonical_smiles[i] for i in kept_idx]

        if args.run_molformer:
            if original_canonical_molformer_emb is None:
                raise RuntimeError("original_canonical MolFormer embedding is missing")

            print(f"MolFormer: {setting_name}")
            tokenizer, model = get_pretrained_transformer(model_name_key=args.model_name)
            setting_emb = embed_smiles_list(
                valid_smiles,
                tokenizer=tokenizer,
                model=model,
                batch_size=args.molformer_batch_size,
                device=args.molformer_device
            )
            save_molformer_outputs(
                setting_dir,
                original_smiles_raw=original_raw_valid,
                setting_smiles=valid_smiles,
                setting_emb=setting_emb,
                original_canonical_emb=original_canonical_molformer_emb[kept_idx]
            )

        if args.run_geodiff:
            if original_canonical_geodiff_2d is None or original_canonical_geodiff_3d is None:
                raise RuntimeError("original_canonical GeoDiff embeddings are missing")

            setting_csv = os.path.join(setting_dir, "_geodiff_input.csv")
            pd.DataFrame({"smiles": valid_smiles}).to_csv(setting_csv, index=False)

            setting_pkl = os.path.join(setting_dir, "_packed.pkl")
            print(f"GeoDiff pack: {setting_name}")
            geodiff_pack_serial_from_csv(
                csv_path=setting_csv,
                out_pkl=setting_pkl,
                ref_confs=args.geodiff_ref_confs,
                seed=args.seed,
                smiles_col="smiles",
            )

            print(f"GeoDiff embed: {setting_name}")
            geodiff_embed_from_packed_serial(
                packed_pkl=setting_pkl,
                out_dir=setting_dir,
                ckpt_path=args.geodiff_ckpt,
                device=args.geodiff_device,
                batch_size=args.geodiff_batch_size,
                which_ref=args.geodiff_which_ref,
                seed=args.seed,
            )

            setting_emb2d, setting_emb3d, _ = load_geodiff_embs(setting_dir)

            save_geodiff_distance_results(
                setting_dir=setting_dir,
                original_smiles_raw=original_raw_valid,
                setting_smiles=valid_smiles,
                setting_emb2d=setting_emb2d,
                setting_emb3d=setting_emb3d,
                original_emb2d=original_canonical_geodiff_2d[kept_idx],
                original_emb3d=original_canonical_geodiff_3d[kept_idx],
            )


if __name__ == "__main__":
    main()
