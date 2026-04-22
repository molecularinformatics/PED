import argparse
import os
import pickle
import shutil
from functools import partial
from multiprocessing import get_context
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

from pipline_utils import smiles_to_conformers, write_json
from MoleculeSDE.Geom3D.datasets.dataset_utils import (
    mol_to_graph_data_obj_simple_2D,
    mol_to_graph_data_obj_simple_3D,
)


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def build_record(
    row: Dict,
    smiles_col: str,
    num_confs: int,
    seed: int,
) -> Dict:
    row_id = _safe_int(row.get("row_id", row.get("input_row_id", row.get("index", 0))))
    smiles = str(row[smiles_col]).strip()

    if not smiles:
        return {
            "ok": False,
            "row_id": row_id,
            "smiles": smiles,
            "error": "empty_smiles",
        }

    mol_2d = Chem.MolFromSmiles(smiles)
    if mol_2d is None:
        return {
            "ok": False,
            "row_id": row_id,
            "smiles": smiles,
            "error": "rdkit_mol_from_smiles_failed",
        }

    try:
        data_2d = mol_to_graph_data_obj_simple_2D(mol_2d)
    except Exception as exc:
        return {
            "ok": False,
            "row_id": row_id,
            "smiles": smiles,
            "error": f"build_2d_failed: {exc}",
        }

    conf_mols = smiles_to_conformers(smiles, num_confs=num_confs, seed=seed + (row_id or 0))
    if len(conf_mols) == 0:
        return {
            "ok": False,
            "row_id": row_id,
            "smiles": smiles,
            "error": "conformer_generation_failed",
        }

    try:
        data_3d, atom_count = mol_to_graph_data_obj_simple_3D(conf_mols[0])
    except Exception as exc:
        return {
            "ok": False,
            "row_id": row_id,
            "smiles": smiles,
            "error": f"build_3d_failed: {exc}",
        }

    data_2d.id = row_id
    data_3d.id = row_id

    return {
        "ok": True,
        "row_id": row_id,
        "smiles": smiles,
        "data_2d": data_2d,
        "data_3d": data_3d,
        "num_atoms": int(data_2d.x.size(0)),
        "num_bonds": int(data_2d.edge_index.size(1) // 2),
        "atom_count": dict(atom_count),
    }


assert Data is not None


def iter_chunks(rows: List[Dict], chunk_size: int):
    for start in range(0, len(rows), chunk_size):
        yield start, rows[start : start + chunk_size]


def process_chunk_and_write(
    chunk_idx: int,
    chunk_rows: List[Dict],
    smiles_col: str,
    num_confs: int,
    seed: int,
    tmp_dir: str,
) -> Dict:
    worker_records: List[Dict] = []
    success_meta: List[Dict] = []
    failures: List[Dict] = []

    for row in tqdm(chunk_rows, desc=f"Chunk {chunk_idx}", leave=False):
        rec = build_record(
            row=row,
            smiles_col=smiles_col,
            num_confs=num_confs,
            seed=seed,
        )
        if rec["ok"]:
            worker_records.append(rec)
            success_meta.append(
                {
                    "row_id": rec["row_id"],
                    "smiles": rec["smiles"],
                    "num_atoms": rec["num_atoms"],
                    "num_bonds": rec["num_bonds"],
                }
            )
        else:
            failures.append(
                {
                    "row_id": rec.get("row_id"),
                    "smiles": rec.get("smiles"),
                    "error": rec.get("error"),
                }
            )

    os.makedirs(tmp_dir, exist_ok=True)
    chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_idx:06d}.pkl")
    with open(chunk_path, "wb") as f:
        pickle.dump(worker_records, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "chunk_idx": chunk_idx,
        "chunk_path": chunk_path,
        "n_success": len(worker_records),
        "success_meta": success_meta,
        "failures": failures,
    }


def merge_chunk_files_to_shards(
    chunk_results: List[Dict],
    final_shard_dir: str,
    shard_size: int,
) -> List[str]:
    os.makedirs(final_shard_dir, exist_ok=True)

    shard_paths: List[str] = []
    current_records: List[Dict] = []
    shard_idx = 0

    for info in sorted(chunk_results, key=lambda x: x["chunk_idx"]):
        chunk_path = info["chunk_path"]
        with open(chunk_path, "rb") as f:
            chunk_records = pickle.load(f)

        for rec in chunk_records:
            current_records.append(rec)
            if len(current_records) >= shard_size:
                shard_path = os.path.join(final_shard_dir, f"shard_{shard_idx:06d}.pkl")
                with open(shard_path, "wb") as fout:
                    pickle.dump(current_records, fout, protocol=pickle.HIGHEST_PROTOCOL)
                shard_paths.append(shard_path)
                current_records = []
                shard_idx += 1

    if current_records:
        shard_path = os.path.join(final_shard_dir, f"shard_{shard_idx:06d}.pkl")
        with open(shard_path, "wb") as fout:
            pickle.dump(current_records, fout, protocol=pickle.HIGHEST_PROTOCOL)
        shard_paths.append(shard_path)

    return shard_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare sharded SDE inputs from a CSV of canonical SMILES")
    parser.add_argument("--csv", required=True, help="Input CSV. SMILES are assumed already canonical.")
    parser.add_argument("--out_dir", required=True, help="Output directory for sharded prepared data")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=128, help="Number of rows per worker task")
    parser.add_argument("--shard_size", type=int, default=5000, help="Final output shard size")
    parser.add_argument("--num_confs", type=int, default=1, help="How many ETKDG conformers to try per molecule")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tmp_chunk_dir = os.path.join(out_dir, "tmp_chunks")
    final_shard_dir = os.path.join(out_dir, "shards")

    if os.path.exists(tmp_chunk_dir):
        shutil.rmtree(tmp_chunk_dir)
    os.makedirs(tmp_chunk_dir, exist_ok=True)
    os.makedirs(final_shard_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.csv}")

    if "row_id" not in df.columns:
        df = df.copy()
        df["row_id"] = np.arange(len(df), dtype=np.int64)

    rows = df.to_dict("records")
    chunked_rows = list(iter_chunks(rows, args.chunksize))

    worker_fn = partial(
        process_chunk_and_write,
        smiles_col=args.smiles_col,
        num_confs=args.num_confs,
        seed=args.seed,
        tmp_dir=tmp_chunk_dir,
    )

    chunk_results: List[Dict] = []

    if args.n_workers <= 1:
        iterator = map(lambda x: worker_fn(chunk_idx=x[0], chunk_rows=x[1]), chunked_rows)
        for result in iterator:
            chunk_results.append(result)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.n_workers) as pool:
            iterator = pool.starmap_async(
                worker_fn,
                [(chunk_idx, chunk_rows) for chunk_idx, chunk_rows in chunked_rows],
            )
            for result in iterator.get():
                chunk_results.append(result)

    shard_paths = merge_chunk_files_to_shards(
        chunk_results=chunk_results,
        final_shard_dir=final_shard_dir,
        shard_size=args.shard_size,
    )

    success_meta: List[Dict] = []
    failures: List[Dict] = []

    for info in sorted(chunk_results, key=lambda x: x["chunk_idx"]):
        success_meta.extend(info["success_meta"])
        failures.extend(info["failures"])

    success_df = pd.DataFrame(success_meta)
    failure_df = pd.DataFrame(failures)

    success_df.to_csv(os.path.join(out_dir, "prepared_success.csv"), index=False)
    if len(failure_df) > 0:
        failure_df.to_csv(os.path.join(out_dir, "prepared_failed.csv"), index=False)

    manifest = {
        "input_csv": os.path.abspath(args.csv),
        "smiles_col": args.smiles_col,
        "n_input_rows": int(len(df)),
        "n_success": int(len(success_meta)),
        "n_failed": int(len(failures)),
        "n_chunks": int(len(chunked_rows)),
        "n_shards": int(len(shard_paths)),
        "worker_chunk_size": int(args.chunksize),
        "shard_size": int(args.shard_size),
        "num_confs": int(args.num_confs),
        "seed": int(args.seed),
        "shards": [os.path.abspath(p) for p in shard_paths],
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    # Optional cleanup. Keep tmp chunks if you want debugging.
    shutil.rmtree(tmp_chunk_dir, ignore_errors=True)

    print(f"Saved manifest: {os.path.join(out_dir, 'manifest.json')}")
    print(f"Prepared molecules: {len(success_meta)}")
    print(f"Failed molecules: {len(failures)}")
    print(f"Final shards: {len(shard_paths)}")


if __name__ == "__main__":
    main()
