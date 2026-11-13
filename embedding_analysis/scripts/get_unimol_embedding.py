import os
import argparse
import numpy as np
import pandas as pd

from unimol_tools import UniMolRepr

from pathlib import Path

# Repo-local scratch for pipeline outputs; see .gitignore.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(RESULTS / "ampc200k" / "similarity_distance_200k_with_all_distances.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(RESULTS / "ampc200k" / "embeddings" / "unimol"),
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="Column name in the CSV containing SMILES.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="Whether to use UniMol built-in DDP.",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
    )
    parser.add_argument(
        "--remove_hs",
        action="store_true",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unimolv2",
        choices=["unimolv1", "unimolv2"],
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="164m",
        help="Used when model_name=unimolv2.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str,
        default="all",
        help='GPU selection string passed to UniMolRepr, e.g. "all", "0", "0,1,2,3".',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Reading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if args.smiles_col not in df.columns:
        raise ValueError(
            f"Column '{args.smiles_col}' not found. Available columns: {list(df.columns)}"
        )

    smiles = df[args.smiles_col].astype(str).tolist()
    print(f"[INFO] Number of SMILES: {len(smiles)}")
    print(f"[INFO] DDP enabled: {args.use_ddp}")

    clf = UniMolRepr(
        data_type="molecule",
        batch_size=args.batch_size,
        remove_hs=args.remove_hs,
        model_name=args.model_name,
        model_size=args.model_size,
        use_cuda=args.use_cuda,
        use_ddp=args.use_ddp,
        use_gpu=args.use_gpu,
    )

    unimol_repr = clf.get_repr(smiles, return_atomic_reprs=True)

    atomic_reprs = unimol_repr["atomic_reprs"]  # list of N * (num_atoms, D)
    emb = np.stack([np.mean(arr, axis=0).astype(np.float32) for arr in atomic_reprs], axis=0)  # (N, D)

    # cls_reprs = unimol_repr["cls_repr"]  # shape: (N, D)
    # emb = np.asarray(cls_reprs, dtype=np.float32)

    smiles_path = os.path.join(args.output_dir, "smiles.npy")
    emb_path = os.path.join(args.output_dir, "emb.npy")

    np.save(smiles_path, np.array(smiles, dtype=object), allow_pickle=True)
    np.save(emb_path, emb)

    print(f"[INFO] Saved: {smiles_path}")
    print(f"[INFO] Saved: {emb_path}")
    print(f"[INFO] emb shape: {emb.shape}")


if __name__ == "__main__":
    main()
