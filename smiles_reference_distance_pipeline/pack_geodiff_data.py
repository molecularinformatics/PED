import os
import json
import pickle
import atexit
import argparse
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

from pipline_utils import row_to_mols
from geodiff.utils.datasets import rdmol_to_data as _rdmol_to_data


_G = None
rdmol_to_data = None


def _data_shard_path(worker_id: int, shard_id: int, out_dir: str):
    return os.path.join(out_dir, f"worker{worker_id:03d}_shard{shard_id:07d}.pkl")


def _failed_shard_path(worker_id: int, shard_id: int, out_dir: str):
    return os.path.join(out_dir, f"failed_worker{worker_id:03d}_shard{shard_id:07d}.pkl")


def _atomic_write_pickle(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _flush_data():
    global _G
    if _G is None or not _G["buf"]:
        return 0
    path = _data_shard_path(_G["worker_id"], _G["shard_id"], _G["out_dir"])
    _atomic_write_pickle(path, _G["buf"])
    n = len(_G["buf"])
    _G["buf"].clear()
    _G["shard_id"] += 1
    _G["n_added"] += n
    return n


def _flush_failed():
    global _G
    if _G is None or not _G["failed_buf"]:
        return 0
    path = _failed_shard_path(_G["worker_id"], _G["failed_shard_id"], _G["out_dir"])
    _atomic_write_pickle(path, _G["failed_buf"])
    n = len(_G["failed_buf"])
    _G["failed_buf"].clear()
    _G["failed_shard_id"] += 1
    _G["n_failed_written"] += n
    return n


def _finalize_worker():
    try:
        _flush_data()
    except Exception:
        pass
    try:
        _flush_failed()
    except Exception:
        pass


def _init_worker(out_dir: str, shard_size: int, failed_shard_size: int, base_seed: int, ref_confs: int):
    global _G, rdmol_to_data
    os.makedirs(out_dir, exist_ok=True)
    
    rdmol_to_data = _rdmol_to_data
    worker_id = os.getpid()
    _G = {
        "out_dir": out_dir,
        "shard_size": int(shard_size),
        "failed_shard_size": int(failed_shard_size),
        "base_seed": int(base_seed),
        "ref_confs": int(ref_confs),
        "worker_id": worker_id,
        "buf": [],
        "shard_id": 0,
        "n_added": 0,
        "failed_buf": [],
        "failed_shard_id": 0,
        "n_failed_seen": 0,
        "n_failed_written": 0,
    }
    atexit.register(_finalize_worker)


def _worker(task):
    global _G, rdmol_to_data
    i, row = task

    # Process ligand mol2 and library smi separately
    mols = row_to_mols(
        row=row,
        ref_confs=_G["ref_confs"],
        seed=_G["base_seed"] + i,
    )

    if not mols:
        _G["n_failed_seen"] += 1
        _G["failed_buf"].append((int(i), row))
        if len(_G["failed_buf"]) >= _G["failed_shard_size"]:
            _flush_failed()
        return 0, 1

    smiles = str(row["smiles"]).strip()

    added = 0
    for m in mols:
        d = rdmol_to_data(m, smiles=smiles)
        _G["buf"].append(d)
        added += 1
        if len(_G["buf"]) >= _G["shard_size"]:
            _flush_data()

    return added, 0


def main():
    parser = argparse.ArgumentParser(description="Pack smiles into GeoDiff shard pkl files")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ref_confs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=45)
    parser.add_argument("--chunksize", type=int, default=256)
    parser.add_argument("--shard_size", type=int, default=100000)
    parser.add_argument("--failed_shard_size", type=int, default=5000)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()


    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    records = df.to_dict(orient="records")
    n_tasks = len(records)
    n_workers = args.n_workers if args.n_workers and args.n_workers > 0 else (os.cpu_count() or 1)
    os.makedirs(args.out_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            os.path.abspath(args.out_dir),
            args.shard_size,
            args.failed_shard_size,
            args.seed,
            args.ref_confs,
        ),
        maxtasksperchild=20000,
    )

    total_added = 0
    total_failed = 0
    try:
        tasks = [(i, row) for i, row in enumerate(records)]
        it = pool.imap_unordered(_worker, tasks, chunksize=args.chunksize)
        for added, failed in tqdm(it, total=n_tasks):
            total_added += int(added)
            total_failed += int(failed)
    finally:
        pool.close()
        pool.join()

    meta = {
        "csv": os.path.abspath(args.csv),
        "smiles_col": args.smiles_col,
        "out_dir": os.path.abspath(args.out_dir),
        "ref_confs": args.ref_confs,
        "seed": args.seed,
        "n_workers": n_workers,
        "chunksize": args.chunksize,
        "shard_size": args.shard_size,
        "failed_shard_size": args.failed_shard_size,
        "n_smiles": n_tasks,
        "n_data_added_reported": total_added,
        "n_failed_smiles_reported": total_failed,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
