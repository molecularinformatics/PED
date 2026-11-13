import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from pathlib import Path

# Repo-local scratch for pipeline outputs; see .gitignore.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def resolve_columns(dist_type: str):
    dist_type = dist_type.lower()
    if dist_type not in {"cosine", "euclidean"}:
        raise ValueError("dist_type must be either 'cosine' or 'euclidean'")

    return {
        "ROCS Combination": "3d_similarity_ROCS",
        "ROCS Color": "ColorTanimoto",
        "ROCS Shape": "ShapeTanimoto",
        "ECFP Tanimoto": "ecfp_tanimoto",
        "GeoDiff Concat": f"geodiff_concat_norm_{dist_type}",
        "GeoDiff 2D": f"geodiff_2D_{dist_type}",
        "GeoDiff 3D": f"geodiff_3D_{dist_type}",
        "MoLFormer": f"molformer_{dist_type}",

        "MoleculeSDE Concat": f"sde_concat_norm_{dist_type}",
        "MoleculeSDE 2D": f"sde_2D_{dist_type}",
        "MoleculeSDE 3D": f"sde_3D_{dist_type}",
        # "UniMol2-84M": f"unimol_84_{dist_type}",
        "UniMol2-164M": f"unimol_164_{dist_type}",
        # "UniMol2-1.1B": f"unimol_1.1_{dist_type}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RESULTS / "ampc200k" / "similarity_distance_200k.csv"),
    )
    parser.add_argument(
        "--dist_type",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PDF path. If None, save next to csv.",
    )
    args = parser.parse_args()

    col_map = resolve_columns(args.dist_type)

    df = pd.read_csv(args.csv)

    if "smiles" not in df.columns:
        raise ValueError("Column 'smiles' not found in CSV.")

    missing_cols = [col for col in col_map.values() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # keep first row for each unique smiles
    df = df.drop_duplicates(subset="smiles", keep="first").copy()

    plot_df = df[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})

    # Pearson correlation
    corr = plot_df.corr(method="pearson")

    # plot
    # fig, ax = plt.subplots(figsize=(9, 8))
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(corr.values, cmap="coolwarm_r",vmin=-1, vmax=1)

    labels = corr.columns.tolist()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)

    # annotate numbers
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            text_color = "black" if val < 0 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("Pearson correlation", rotation=90)

    plt.tight_layout()

    out_path = f"./figs/pearson_corr_{args.dist_type}_all.pdf"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved PDF to: {out_path}")
    print(f"Number of unique smiles used: {len(df)}")


if __name__ == "__main__":
    main()
