import logging
import os
import sys
from glob import glob
from typing import Optional, Sequence

import numpy as np
import torch
import yaml
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

from .geodiff.models.epsnet import get_model
from .geodiff.utils.datasets import rdmol_to_data
from .geodiff.utils.transforms import Compose, CountNodesPerGraph, AddHigherOrderEdges

logger = logging.getLogger(__name__)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_GEODIFF_ROOT = os.path.join(_CURRENT_DIR, "geodiff")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_conformer(smiles: str, seed: int = 42) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    def _embed(params) -> bool:
        params.randomSeed = int(seed) & 0xFFFFFFFF
        params.numThreads = 1
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params))
        return len(conf_ids) > 0

    ok = _embed(AllChem.ETKDGv3())
    if not ok:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        if hasattr(params, "maxIterations"):
            params.maxIterations = 1000
        elif hasattr(params, "maxAttempts"):
            params.maxAttempts = 1000
        ok = _embed(params)

    if not ok:
        params = AllChem.ETKDGv2()
        params.useRandomCoords = True
        if hasattr(params, "maxIterations"):
            params.maxIterations = 2000
        elif hasattr(params, "maxAttempts"):
            params.maxAttempts = 2000
        ok = _embed(params)

    if not ok or mol.GetNumConformers() == 0:
        return None

    try:
        AllChem.UFFOptimizeMolecule(mol, confId=0, maxIters=200)
    except Exception:
        pass

    out = Chem.Mol(mol)
    conf = Chem.Conformer(mol.GetConformer(0))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def smiles_to_mol(smiles: str, seed: int = 42) -> Optional[Chem.Mol]:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    return smiles_to_conformer(canonical, seed=seed)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, eps, None)


def concat_embeddings(emb2d: np.ndarray, emb3d: np.ndarray) -> np.ndarray:
    return np.concatenate([emb2d, emb3d], axis=1).astype(np.float32)


def concat_normalized_embeddings(emb2d: np.ndarray, emb3d: np.ndarray) -> np.ndarray:
    return np.concatenate([l2_normalize(emb2d), l2_normalize(emb3d)], axis=1).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def sigmoid_transform(value, low, high, k=-0.25):
    return 1 / (1 + 10 ** (10 * k * (value - (low + high) * 0.5) / (low - high)))


def resolve_config_yml_from_ckpt(ckpt_path: str) -> str:
    ckpt_path = os.path.abspath(ckpt_path)
    ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
    ymls = sorted(glob(os.path.join(ckpt_dir, "*.yml")))
    if not ymls:
        raise FileNotFoundError(f"No .yml found in {ckpt_dir}.")
    return ymls[0]


def load_geodiff_model_and_transforms(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    yml_path = resolve_config_yml_from_ckpt(ckpt_path)
    with open(yml_path, "r", encoding="utf-8") as f:
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
        edge_type=edge_type,
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


def prepare_geodiff_data(mol, smiles: str, transforms):
    data = rdmol_to_data(mol=mol, smiles=smiles)
    data = transforms(data)

    if hasattr(data, "pos_ref") and data.pos_ref is not None:
        pos = data.pos_ref[: int(data.num_nodes)]
        data_input = data.clone()
        data_input["pos_ref"] = None
        data_input.pos = pos
        return data_input
    return data


@torch.no_grad()
def geodiff_embed_batch(
    model,
    transforms,
    mols: Sequence,
    smiles_list: Sequence[str],
    device: torch.device,
    mode: str = "concat",
) -> np.ndarray:
    if len(mols) != len(smiles_list):
        raise ValueError("mols and smiles_list must have the same length.")
    if len(mols) == 0:
        raise ValueError("Empty input to geodiff_embed_batch.")

    data_list = [
        prepare_geodiff_data(mol=mol, smiles=smiles, transforms=transforms)
        for mol, smiles in zip(mols, smiles_list)
    ]

    batch = Batch.from_data_list(data_list).to(device)
    mol_emb_2d, mol_emb_3d = dual_mol_embedding(
        model=model,
        atom_type=batch.atom_type,
        pos=batch.pos,
        bond_index=batch.edge_index,
        bond_type=batch.edge_type,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
    )

    emb2d = mol_emb_2d.detach().cpu().numpy().astype(np.float32)
    emb3d = mol_emb_3d.detach().cpu().numpy().astype(np.float32)

    mode = str(mode).strip()
    if mode == "2D":
        return emb2d
    if mode == "3D":
        return emb3d
    if mode in ["concat", "weight_concat"]:
        return concat_embeddings(emb2d, emb3d)
    if mode == "concat_norm":
        return concat_normalized_embeddings(emb2d, emb3d)      
    raise ValueError(f"embedding_mode must be one of ['2D', '3D', 'concat', 'concat_norm', 'weight_concat'], got {mode!r}")
