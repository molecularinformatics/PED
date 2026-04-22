import os
import math
import argparse
import pickle
import subprocess
import sys
from glob import glob

import yaml
import torch
import numpy as np
from easydict import EasyDict
from tqdm.auto import tqdm
from torch_scatter import scatter_mean
from torch_geometric.data import Batch

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


@torch.no_grad()
def process_range_worker(args):
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    model, transforms = load_model_and_transforms(os.path.abspath(args.ckpt), args.device)

    total_n = count_total(args.packed_dir, args.start_idx, args.end_idx)
    if total_n == 0:
        empty = np.zeros((0, 128), dtype=np.float32)
        save_outputs(args.out_dir, emb2d=empty, emb3d=empty, smiles_list=[])
        print(f"Worker on {args.device}: empty range [{args.start_idx}, {args.end_idx})")
        return

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

    batch_graphs = []
    batch_smiles = []
    write_idx = 0

    iterator = iter_shard_data(args.packed_dir, args.start_idx, args.end_idx)
    pbar = tqdm(iterator, total=total_n, desc=f"embed {args.device} [{args.start_idx}:{args.end_idx})")

    def flush_batch():
        nonlocal batch_graphs, batch_smiles, write_idx
        if len(batch_graphs) == 0:
            return
        pyg_batch = Batch.from_data_list(batch_graphs).to(args.device)
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

    for data in pbar:
        data_input, smiles = preprocess_one_data(data, transforms, which_ref=args.which_ref)
        batch_graphs.append(data_input)
        batch_smiles.append(smiles)
        if len(batch_graphs) >= args.batch_size:
            flush_batch()
    flush_batch()

    emb2d = np.array(emb2d_mm)
    emb3d = np.array(emb3d_mm)
    save_outputs(args.out_dir, emb2d=emb2d, emb3d=emb3d, smiles_list=smiles_out)

    for tmp_name in ["_tmp_emb2d.npy", "_tmp_emb3d.npy"]:
        tmp_path = os.path.join(args.out_dir, tmp_name)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"Worker on {args.device}: saved {len(smiles_out)} embeddings to {args.out_dir}")


def split_ranges(total_n, n_parts):
    if total_n < 0:
        raise ValueError("total_n must be >= 0")
    if n_parts <= 0:
        raise ValueError("n_parts must be > 0")
    chunk = int(math.ceil(total_n / n_parts)) if total_n > 0 else 0
    ranges = []
    for i in range(n_parts):
        start = i * chunk
        end = min(total_n, (i + 1) * chunk)
        if start >= end:
            break
        ranges.append((start, end))
    return ranges


def merge_worker_outputs(parts_root: str, final_out_dir: str):
    part_dirs = sorted(glob(os.path.join(parts_root, "part_*")))
    if len(part_dirs) == 0:
        raise FileNotFoundError(f"No part_* directories found under: {parts_root}")

    emb2d_list, emb3d_list, smiles_list = [], [], []
    for part_dir in part_dirs:
        emb2d = np.load(os.path.join(part_dir, "2D", "emb.npy"), allow_pickle=True)
        emb3d = np.load(os.path.join(part_dir, "3D", "emb.npy"), allow_pickle=True)
        smiles = np.load(os.path.join(part_dir, "2D", "smiles.npy"), allow_pickle=True)
        emb2d_list.append(emb2d)
        emb3d_list.append(emb3d)
        smiles_list.extend([str(x) for x in smiles])

    if len(emb2d_list) == 0:
        final_2d = np.zeros((0, 128), dtype=np.float32)
        final_3d = np.zeros((0, 128), dtype=np.float32)
    else:
        final_2d = np.concatenate(emb2d_list, axis=0).astype(np.float32, copy=False)
        final_3d = np.concatenate(emb3d_list, axis=0).astype(np.float32, copy=False)

    save_outputs(final_out_dir, emb2d=final_2d, emb3d=final_3d, smiles_list=smiles_list)
    print(f"Merged {len(part_dirs)} worker outputs into {final_out_dir}")


def launch_multi_gpu(args):
    total_n = count_total(args.packed_dir, args.start_idx, args.end_idx)
    devices = args.devices if args.devices else [args.device]
    ranges = split_ranges(total_n, len(devices))

    if len(ranges) == 0:
        empty = np.zeros((0, 128), dtype=np.float32)
        save_outputs(args.out_dir, emb2d=empty, emb3d=empty, smiles_list=[])
        print(f"No molecules found under {args.packed_dir}")
        return

    parts_root = os.path.join(args.out_dir, "_parts")
    os.makedirs(parts_root, exist_ok=True)

    procs = []
    for worker_idx, ((local_start, local_end), device) in enumerate(zip(ranges, devices)):
        worker_out = os.path.join(parts_root, f"part_{worker_idx:03d}")
        os.makedirs(worker_out, exist_ok=True)

        gpu_id = device.split(":")[1]

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--ckpt", os.path.abspath(args.ckpt),
            "--packed_dir", os.path.abspath(args.packed_dir),
            "--out_dir", worker_out,

            # cuda:0 for subprocess
            "--device", "cuda:0",

            "--start_idx", str(args.start_idx + local_start),
            "--end_idx", str(args.start_idx + local_end),
            "--which_ref", str(args.which_ref),
            "--seed", str(args.seed + worker_idx),
            "--batch_size", str(args.batch_size),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        print("Launching:", " ".join(cmd), " GPU:", gpu_id)

        procs.append(
            subprocess.Popen(
                cmd,
                env=env
            )
        )

    failed = False
    for proc in procs:
        ret = proc.wait()
        if ret != 0:
            failed = True
    if failed:
        raise RuntimeError("At least one embedding worker failed.")

    merge_worker_outputs(parts_root=parts_root, final_out_dir=args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D and 3D embeddings with batched multi-GPU execution")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--packed_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None, help="Multiple devices, e.g. cuda:0 cuda:1 cuda:2")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10**18)
    parser.add_argument("--which_ref", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--worker", action="store_true", help="Internal flag used by multi-GPU launcher")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.worker:
        process_range_worker(args)
    else:
        devices = args.devices if args.devices else [args.device]
        if len(devices) == 1:
            args.device = devices[0]
            process_range_worker(args)
        else:
            launch_multi_gpu(args)


if __name__ == "__main__":
    main()
