import argparse
import json
import os
import pickle
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from MoleculeSDE.Geom3D.models.molecule_gnn_model import GNN
from MoleculeSDE.Geom3D.models.schnet import SchNet


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_numpy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def normalize_gpu_devices(devices: Sequence[str]) -> List[str]:
    out = []
    for d in devices:
        if not d.startswith("cuda:"):
            raise ValueError(f"Expected device like cuda:0, got {d}")
        idx = d.split(":", 1)[1]
        if not idx.isdigit():
            raise ValueError(f"Invalid CUDA device: {d}")
        out.append(d)
    return out


def build_models(device: torch.device, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_2d = GNN(
        num_layer=5,
        emb_dim=300,
        JK="last",
        drop_ratio=0,
        gnn_type="GIN",
    )
    model_3d = SchNet(
        hidden_channels=300,
        num_filters=128,
        num_interactions=6,
        num_gaussians=51,
        cutoff=10.0,
        readout="mean",
        node_class=119,
    )

    if "model_2D" not in ckpt or "model_3D" not in ckpt:
        raise ValueError("Checkpoint must contain keys 'model_2D' and 'model_3D'")

    model_2d.load_state_dict(ckpt["model_2D"], strict=True)
    model_3d.load_state_dict(ckpt["model_3D"], strict=True)

    model_2d = model_2d.to(device).eval()
    model_3d = model_3d.to(device).eval()
    return model_2d, model_3d


@torch.no_grad()
def embed_records(
    records: List[Dict],
    model_2d,
    model_3d,
    device: torch.device,
    batch_size: int,
):
    rows: List[int] = []
    smiles_list: List[str] = []
    emb_2d_list: List[np.ndarray] = []
    emb_3d_list: List[np.ndarray] = []

    for start in tqdm(range(0, len(records), batch_size), desc="Embedding batches", leave=False):
        batch_records = records[start : start + batch_size]

        batch_2d = Batch.from_data_list([r["data_2d"] for r in batch_records]).to(device)
        batch_3d = Batch.from_data_list([r["data_3d"] for r in batch_records]).to(device)

        node_repr_2d = model_2d(batch_2d.x, batch_2d.edge_index, batch_2d.edge_attr)
        graph_repr_2d = global_mean_pool(node_repr_2d, batch_2d.batch)

        # SchNet takes atomic numbers and positions
        _, node_repr_3d = model_3d(
            batch_3d.x[:, 0],
            batch_3d.positions,
            batch_3d.batch,
            return_latent=True,
        )
        graph_repr_3d = global_mean_pool(node_repr_3d, batch_3d.batch)

        emb_2d = graph_repr_2d.detach().cpu().numpy().astype(np.float32)
        emb_3d = graph_repr_3d.detach().cpu().numpy().astype(np.float32)

        for rec, e2, e3 in zip(batch_records, emb_2d, emb_3d):
            rows.append(int(rec["row_id"]))
            smiles_list.append(str(rec["smiles"]).strip())
            emb_2d_list.append(e2)
            emb_3d_list.append(e3)

    if len(emb_2d_list) == 0 or len(emb_3d_list) == 0:
        raise RuntimeError("No embeddings were generated for this shard split")

    return (
        rows,
        smiles_list,
        np.stack(emb_2d_list, axis=0),
        np.stack(emb_3d_list, axis=0),
    )


def run_worker_on_shards(
    worker_name: str,
    visible_gpu_id: str,
    shard_paths: List[str],
    ckpt_path: str,
    out_dir: str,
    batch_size: int,
):
    # Important: set CUDA_VISIBLE_DEVICES before touching CUDA in this process
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu_id
    device = torch.device("cuda:0")

    model_2d, model_3d = build_models(device=device, ckpt_path=ckpt_path)

    per_shard_dir = os.path.join(out_dir, "per_shard")
    os.makedirs(per_shard_dir, exist_ok=True)

    for shard_path in shard_paths:
        records = load_pickle(shard_path)
        rows, smiles, emb_2d, emb_3d = embed_records(
            records=records,
            model_2d=model_2d,
            model_3d=model_3d,
            device=device,
            batch_size=batch_size,
        )
        base = Path(shard_path).stem
        with open(os.path.join(per_shard_dir, f"{base}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "row_id": np.asarray(rows, dtype=np.int64),
                    "smiles": np.asarray(smiles, dtype=object),
                    "emb_2d": emb_2d,
                    "emb_3d": emb_3d,
                    "worker": worker_name,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def merge_outputs(out_dir: str):
    per_shard_dir = os.path.join(out_dir, "per_shard")
    shard_files = sorted(Path(per_shard_dir).glob("*.pkl"))
    if len(shard_files) == 0:
        raise RuntimeError(f"No per-shard outputs found under {per_shard_dir}")

    row_ids = []
    smiles_parts = []
    emb2_parts = []
    emb3_parts = []

    for p in shard_files:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        row_ids.append(obj["row_id"])
        smiles_parts.append(obj["smiles"])
        emb2_parts.append(obj["emb_2d"])
        emb3_parts.append(obj["emb_3d"])

    row_ids = np.concatenate(row_ids, axis=0)
    smiles = np.concatenate(smiles_parts, axis=0)
    emb2 = np.concatenate(emb2_parts, axis=0)
    emb3 = np.concatenate(emb3_parts, axis=0)

    order = np.argsort(row_ids, kind="stable")
    row_ids = row_ids[order]
    smiles = smiles[order]
    emb2 = emb2[order]
    emb3 = emb3[order]

    save_numpy(os.path.join(out_dir, "sde_2D", "smiles.npy"), smiles)
    save_numpy(os.path.join(out_dir, "sde_2D", "emb.npy"), emb2)
    save_numpy(os.path.join(out_dir, "sde_3D", "smiles.npy"), smiles)
    save_numpy(os.path.join(out_dir, "sde_3D", "emb.npy"), emb3)

    pd.DataFrame(
        {
            "row_id": row_ids,
            "smiles": smiles,
        }
    ).to_csv(os.path.join(out_dir, "embedding_index.csv"), index=False)

    meta = {
        "n_rows": int(len(row_ids)),
        "emb_dim_2d": int(emb2.shape[1]),
        "emb_dim_3d": int(emb3.shape[1]),
    }
    with open(os.path.join(out_dir, "embedding_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def split_list_round_robin(items: List[str], n: int) -> List[List[str]]:
    out = [[] for _ in range(n)]
    for i, item in enumerate(items):
        out[i % n].append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract MoleculeSDE 2D and 3D graph embeddings")
    parser.add_argument("--ckpt", required=True, help="Path to model_complete.pth")
    parser.add_argument("--prepared_dir", required=True, help="Output directory from prepare_sde_inputs.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--devices", nargs="+", default=None)
    args = parser.parse_args()

    prepared_dir = os.path.abspath(args.prepared_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(prepared_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    shard_paths = manifest["shards"]
    if len(shard_paths) == 0:
        raise RuntimeError(f"No shards found in manifest: {prepared_dir}")

    devices = args.devices if args.devices else ([args.device] if args.device else ["cuda:0"])
    devices = normalize_gpu_devices(devices)

    # Single GPU: do not use multiprocessing. This avoids CUDA initialization errors.
    if len(devices) == 1:
        device_str = devices[0]
        visible_gpu_id = device_str.split(":", 1)[1]
        run_worker_on_shards(
            worker_name="single_gpu",
            visible_gpu_id=visible_gpu_id,
            shard_paths=shard_paths,
            ckpt_path=os.path.abspath(args.ckpt),
            out_dir=out_dir,
            batch_size=args.batch_size,
        )
    else:
        shard_splits = split_list_round_robin(shard_paths, len(devices))
        ctx = get_context("spawn")
        procs = []

        for rank, (device_str, my_shards) in enumerate(zip(devices, shard_splits)):
            if len(my_shards) == 0:
                continue
            visible_gpu_id = device_str.split(":", 1)[1]
            p = ctx.Process(
                target=run_worker_on_shards,
                args=(
                    f"gpu_worker_{rank}",
                    visible_gpu_id,
                    my_shards,
                    os.path.abspath(args.ckpt),
                    out_dir,
                    args.batch_size,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Embedding worker exited with code {p.exitcode}")

    merge_outputs(out_dir)
    print(f"Saved embeddings under: {out_dir}")


if __name__ == "__main__":
    main()
