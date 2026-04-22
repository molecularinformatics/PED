import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pipline_utils import canonicalize_smiles


def main():
    parser = argparse.ArgumentParser(
        description="Prepare refs.csv and library.csv from one library CSV and one reference SMILES"
    )
    parser.add_argument("--library_csv", required=True, help="Input CSV that contains a SMILES column")
    parser.add_argument("--reference_smiles", required=True, help="Single reference SMILES string")
    parser.add_argument("--out_dir", required=True, help="Output folder for prepared CSV files")
    parser.add_argument("--smiles_col", default="Smiles", help="SMILES column name in the input CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ref_smiles = canonicalize_smiles(args.reference_smiles)
    if ref_smiles is None:
        raise ValueError("reference_smiles is invalid and could not be canonicalized")

    refs_df = pd.DataFrame(
        [{
            "ref_idx": 0,
            "ref_name": "reference_smiles",
            "smiles": ref_smiles,
            "raw_smiles": str(args.reference_smiles).strip(),
        }]
    )

    df = pd.read_csv(args.library_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.library_csv}")

    lib_df = df.copy()
    lib_df["raw_smiles"] = lib_df[args.smiles_col].astype(str).str.strip()
    lib_df["smiles"] = lib_df["raw_smiles"].map(canonicalize_smiles)
    lib_df["input_row_id"] = np.arange(len(lib_df), dtype=np.int64)

    failed_df = lib_df[lib_df["smiles"].isna()].copy()
    lib_df = lib_df[lib_df["smiles"].notna()].copy()

    n_before_dedup = len(lib_df)
    lib_df = lib_df.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)
    lib_df["row_id"] = np.arange(len(lib_df), dtype=np.int64)

    # Keep smiles first for downstream scripts, but also preserve all original columns.
    base_cols = ["row_id", "smiles", "raw_smiles", "input_row_id"]
    # other_cols = [c for c in lib_df.columns if c not in base_cols]
    # lib_df = lib_df[base_cols + other_cols]
    lib_df = lib_df[base_cols]

    refs_csv = os.path.join(args.out_dir, "refs.csv")
    library_csv = os.path.join(args.out_dir, "library.csv")
    refs_df.to_csv(refs_csv, index=False)
    lib_df.to_csv(library_csv, index=False)

    meta = {
        "library_csv": os.path.abspath(args.library_csv),
        "smiles_col": args.smiles_col,
        "reference_smiles_raw": str(args.reference_smiles).strip(),
        "reference_smiles_canonical": ref_smiles,
        "n_library_rows_before_filter": int(len(df)),
        "n_library_rows_failed_smiles": int(len(failed_df)),
        "n_library_rows_before_dedup": int(n_before_dedup),
        "n_library_rows_kept": int(len(lib_df)),
        "n_library_rows_duplicates_removed": int(n_before_dedup - len(lib_df)),
    }
    with open(os.path.join(args.out_dir, "prep_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if len(failed_df) > 0:
        failed_df.to_csv(os.path.join(args.out_dir, "library_failed_smiles.csv"), index=False)

    print(f"Saved refs csv: {refs_csv}")
    print(f"Saved library csv: {library_csv}")
    print(f"Saved metadata: {os.path.join(args.out_dir, 'prep_meta.json')}")


if __name__ == "__main__":
    main()
