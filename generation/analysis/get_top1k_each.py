import os
import sys

import pandas as pd
from paths import RESULTS_ROOT


TOP_K = 1000
# `python get_top1k_each.py <reinvent|libinvent|synformer>` overrides this default
MODEL_TITLE = sys.argv[1] if len(sys.argv) > 1 else "libinvent"
assert MODEL_TITLE in ("reinvent", "libinvent", "synformer"), MODEL_TITLE

BASE_DIR = str(RESULTS_ROOT / MODEL_TITLE / "scaffold_memory")

FILE_MAP = {
    "ROSHAMBO2": f"{MODEL_TITLE}_roshambo2.csv",
    "ECFP4": f"{MODEL_TITLE}_ecfp4.csv",
    "GeoDiff 2D": f"{MODEL_TITLE}_geodiff_2d.csv",
    "GeoDiff 3D": f"{MODEL_TITLE}_geodiff_3d.csv",
    "GeoDiff Concat": f"{MODEL_TITLE}_geodiff_concat.csv",
    "MoLFormer": f"{MODEL_TITLE}_molformer.csv",
}
print(FILE_MAP.keys())

def load_topk_with_method(base_dir, file_map, top_k=1000):
    dfs = []

    for method, filename in file_map.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"[skip] {method}: file not found: {path}")
            continue

        df = pd.read_csv(path)

        if "smiles" in df.columns:
            df.rename(columns={"smiles": "SMILES"}, inplace=True)

        df.drop_duplicates(subset=["SMILES"], inplace=True)

        sub = (
            df.sort_values("total_score", ascending=False)
              .head(top_k)
              .copy()
        )

        sub["Method"] = method
        dfs.append(sub)

    combined = pd.concat(dfs, axis=0, ignore_index=True)

    combined["Method"] = pd.Categorical(
        combined["Method"],
        categories=list(file_map.keys()),
        ordered=True,
    )

    return combined


if __name__ == "__main__":
    combined_df = load_topk_with_method(BASE_DIR, FILE_MAP, TOP_K)

    print(combined_df.head())
    print("\nShape:", combined_df.shape)

    # `python get_top1k_each.py <model> [out_name]`. The default name is the one the
    # published Boltz-2 run was built from, so refuse to clobber it silently: pass a
    # new name (e.g. top1000_all_methods_6arms.csv) when adding an arm.
    out_name = sys.argv[2] if len(sys.argv) > 2 else "top1000_all_methods.csv"
    out_path = os.path.join(BASE_DIR, out_name)
    if os.path.exists(out_path) and len(sys.argv) <= 2:
        raise SystemExit(
            f"{out_path} already exists; pass an output name as the second argument "
            "to keep the file the published pIC50 analysis was built from."
        )
    combined_df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
