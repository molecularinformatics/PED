import os
import argparse
import pickle
from glob import glob

import yaml
import torch
import numpy as np
from easydict import EasyDict
from tqdm.auto import tqdm
from torch_scatter import scatter_mean

from geodiff.utils.misc import seed_all
from geodiff.models.epsnet import get_model
from geodiff.utils.transforms import Compose, CountNodesPerGraph, AddHigherOrderEdges


@torch.no_grad()
def dual_mol_embedding(model, atom_type, pos, bond_index, bond_type, batch, num_graphs):
    """
    Generate both 2D (GIN-based local) and 3D (SchNet-based global) embeddings
    in a single forward pass.
    
    Returns:
        mol_emb_2d: GIN local embedding (N, 128)
        mol_emb_3d: SchNet global embedding (N, 128)
    """
    device = pos.device
    time_step = torch.zeros(num_graphs, dtype=torch.long, device=device)

    # Single forward pass to get edges
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

    # Global edge attributes (used for both encoders)
    edge_attr_global = model.edge_encoder_global(
        edge_length=edge_length,
        edge_type=edge_type
    )

    # 3D embedding (SchNet global encoder)
    node_attr_global = model.encoder_global(
        z=atom_type,
        edge_index=edge_index,
        edge_length=edge_length,
        edge_attr=edge_attr_global,
    )
    mol_emb_3d = scatter_mean(node_attr_global, batch, dim=0)

    # 2D embedding (GIN local encoder)
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


def pick_pos_ref(data, which_ref: int = 0):
    num_nodes = int(data.num_nodes)
    total = int(data.pos_ref.size(0))
    num_refs = total // num_nodes
    if num_refs <= 0:
        raise ValueError("pos_ref is empty")
    which = which_ref % num_refs
    pos = data.pos_ref.view(num_refs, num_nodes, 3)[which]
    return pos


def iter_shard_data(packed_dir: str, start_idx: int = 0, end_idx: int = 10**18):
    shard_paths = sorted(glob(os.path.join(packed_dir, "worker*_shard*.pkl")))
    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No worker*_shard*.pkl found under: {packed_dir}")

    seen = 0
    for p in shard_paths:
        with open(p, "rb") as f:
            lst = pickle.load(f)
        for d in lst:
            if seen < start_idx:
                seen += 1
                continue
            if seen >= end_idx:
                return
            yield d
            seen += 1


def count_total(packed_dir: str, start_idx: int = 0, end_idx: int = 10**18):
    shard_paths = sorted(glob(os.path.join(packed_dir, "worker*_shard*.pkl")))
    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No worker*_shard*.pkl found under: {packed_dir}")

    total = 0
    seen = 0
    for p in shard_paths:
        with open(p, "rb") as f:
            lst = pickle.load(f)
        n = len(lst)
        if seen + n <= start_idx:
            seen += n
            continue
        s = max(0, start_idx - seen)
        e = min(n, end_idx - seen) if end_idx < 10**18 else n
        if e > s:
            total += e - s
        seen += n
        if seen >= end_idx:
            break
    return total


def load_model_and_transforms(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
    ymls = glob(os.path.join(ckpt_dir, "*.yml"))
    if len(ymls) == 0:
        raise FileNotFoundError(f"No .yml found in {ckpt_dir}. Put the config yml there.")

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


def save_outputs(out_dir: str, emb2d: np.ndarray, emb3d: np.ndarray, smiles_list):
    out_2d = os.path.join(out_dir, "2D")
    out_3d = os.path.join(out_dir, "3D")
    os.makedirs(out_2d, exist_ok=True)
    os.makedirs(out_3d, exist_ok=True)

    smiles_arr = np.array(smiles_list, dtype=object)
    np.save(os.path.join(out_2d, "emb.npy"), emb2d)
    np.save(os.path.join(out_2d, "smiles.npy"), smiles_arr)
    np.save(os.path.join(out_3d, "emb.npy"), emb3d)
    np.save(os.path.join(out_3d, "smiles.npy"), smiles_arr)


def main():
    parser = argparse.ArgumentParser(description="Generate 2D and 3D embeddings together from packed shard pkls")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--packed_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10**18)
    parser.add_argument("--which_ref", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seed_all(args.seed)
    model, transforms = load_model_and_transforms(os.path.abspath(args.ckpt), args.device)

    total_n = count_total(args.packed_dir, args.start_idx, args.end_idx)
    emb2d_mm = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "_tmp_emb2d.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(total_n, 128),
    )
    emb3d_mm = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "_tmp_emb3d.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(total_n, 128),
    )
    smiles_out = []

    j = 0
    for data in tqdm(iter_shard_data(args.packed_dir, args.start_idx, args.end_idx), total=total_n):
        data = transforms(data)
        smiles = getattr(data, "smiles", "")
        if smiles is None:
            smiles = ""

        if hasattr(data, "pos_ref") and data.pos_ref is not None:
            pos = pick_pos_ref(data, which_ref=args.which_ref)
            data_input = data.clone()
            data_input["pos_ref"] = None
            data_input.pos = pos
        else:
            data_input = data

        batch = repeat_data(data_input, 1).to(args.device)

        mol_emb_2d, mol_emb_3d = dual_mol_embedding(
            model=model,
            atom_type=batch.atom_type,
            pos=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_graphs=batch.num_graphs,
        )

        emb2d_mm[j] = mol_emb_2d[0].detach().cpu().numpy().astype(np.float32)
        emb3d_mm[j] = mol_emb_3d[0].detach().cpu().numpy().astype(np.float32)
        smiles_out.append(smiles)
        j += 1

    emb2d = np.array(emb2d_mm)
    emb3d = np.array(emb3d_mm)
    os.makedirs(args.out_dir, exist_ok=True)
    save_outputs(args.out_dir, emb2d=emb2d, emb3d=emb3d, smiles_list=smiles_out)

    tmp2d = os.path.join(args.out_dir, "_tmp_emb2d.npy")
    tmp3d = os.path.join(args.out_dir, "_tmp_emb3d.npy")
    if os.path.exists(tmp2d):
        os.remove(tmp2d)
    if os.path.exists(tmp3d):
        os.remove(tmp3d)

    print(f"Saved {len(smiles_out)} embeddings to {args.out_dir}")


if __name__ == "__main__":
    main()
