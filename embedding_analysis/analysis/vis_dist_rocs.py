import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from pathlib import Path

# Repo-local scratch for pipeline outputs; see .gitignore.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def build_rocs_bins_from_data(values, step=0.1):
    """
    Generate bins from actual min/max.
    Keep labels like '0.4-0.5'.
    """
    values = pd.Series(values).dropna().astype(float)
    if len(values) == 0:
        raise ValueError("No valid ROCS values found.")

    vmin = values.min()
    vmax = values.max()

    left = math.floor(vmin / step) * step
    right = math.ceil(vmax / step) * step

    if np.isclose(left, right):
        right = left + step

    # include last edge
    bins = np.arange(left, right + step * 0.5, step)
    bins = np.round(bins, 10)

    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    return bins, labels


def add_rocs_group(df, rocs_col="3d_similarity_ROCS", step=0.1):
    bins, labels = build_rocs_bins_from_data(df[rocs_col], step=step)

    out = df.copy()
    out["ROCS_Group"] = pd.cut(
        out[rocs_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    # include possible exact max at right boundary
    max_val = out[rocs_col].max()
    if np.isclose(max_val, bins[-1]):
        out.loc[out[rocs_col] == max_val, "ROCS_Group"] = labels[-1]

    return out, labels, bins


def plot_boxpanels(
    df,
    cosine_col,
    euclidean_col,
    output_path,
    rocs_col,
    rocs_step=0.1,
    xlabel="ROCS 3D Similarity Group",
    ylabel_model="MoLFormer",
    logy=False,
):
    df_plot, group_order, bins = add_rocs_group(df, rocs_col=rocs_col, step=rocs_step)

    def corr_pair(col):
        """Pearson and Spearman against the ROCS column, on rows where both exist."""
        m = df[[col, rocs_col]].dropna()
        return (stats.pearsonr(m[col], m[rocs_col])[0],
                stats.spearmanr(m[col], m[rocs_col])[0])

    corr_cosine, rho_cosine = corr_pair(cosine_col)
    corr_euclidean, rho_euclidean = corr_pair(euclidean_col)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    style = dict(
        patch_artist=True,
        boxprops=dict(facecolor="#5B8DB8", alpha=0.8, edgecolor="black"),
        medianprops=dict(color="red", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(
            marker="o",
            markersize=4,
            markerfacecolor="gray",
            markeredgecolor="gray",
            alpha=0.35,
            markeredgewidth=0.0,
        ),
    )

    cosine_data = [
        df_plot.loc[df_plot["ROCS_Group"] == g, cosine_col].dropna().values
        for g in group_order
    ]
    euclidean_data = [
        df_plot.loc[df_plot["ROCS_Group"] == g, euclidean_col].dropna().values
        for g in group_order
    ]

    axes[0].boxplot(cosine_data, labels=group_order, **style)
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel(f"{ylabel_model} Cosine Distance", fontsize=12)
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].text(
        0.98, 0.97,
        f"Pearson $r$ = {corr_cosine:.2f}\nSpearman $\\rho$ = {rho_cosine:.2f}",
        transform=axes[0].transAxes,
        fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="none"),
    )

    axes[1].boxplot(euclidean_data, labels=group_order, **style)
    if logy:
        # A few spaces put a small subpopulation three orders of magnitude from the
        # bulk; on a linear axis that collapses every box onto the baseline.
        for a in axes:
            a.set_yscale("log")
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].set_ylabel(f"{ylabel_model} Euclidean Distance", fontsize=12)
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].text(
        0.98, 0.97,
        f"Pearson $r$ = {corr_euclidean:.2f}\nSpearman $\\rho$ = {rho_euclidean:.2f}",
        transform=axes[1].transAxes,
        fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {output_path}")
    print(f"ROCS bin edges: {bins}")
    print(f"cosine    Pearson {corr_cosine:+.3f}  Spearman {rho_cosine:+.3f}")
    print(f"euclidean Pearson {corr_euclidean:+.3f}  Spearman {rho_euclidean:+.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize distance vs ROCS similarity")
    parser.add_argument("--model", type=str, default="molformer", choices=["geodiff", "molformer", "unimol", "sde"])
    parser.add_argument("--mode", type=str, default="concat_norm", choices=["2D", "3D", "concat", "concat_norm"])
    parser.add_argument("--rocs_col", type=str, default="3d_similarity_ROCS", choices=["3d_similarity_ROCS", "ShapeTanimoto", "ColorTanimoto"])
    # The reported Uni-Mol2 space is the 164M checkpoint with atom-wise mean pooling.
    parser.add_argument("--unimol_size", type=str, default="164", choices=["164", "84", "1.1"])
    parser.add_argument("--outdir", type=str, default="./figs")
    parser.add_argument("--logy", action="store_true", help="log distance axis; use when a subpopulation sits far from the bulk")
    args = parser.parse_args()

    dir_name = str(RESULTS / "ampc200k")
    df = pd.read_csv(os.path.join(dir_name, "similarity_distance_200k.csv"))
    df = df.drop_duplicates(subset="smiles")
    print(df.shape)

    os.makedirs(args.outdir, exist_ok=True)
    suffix = "_log" if args.logy else ""

    if args.model in ["molformer", "unimol"]:
        stem = args.model if args.model == "molformer" else f"{args.model}_{args.unimol_size}"
        cosine_col = f"{stem}_cosine"
        euclidean_col = f"{stem}_euclidean"
        output_path = os.path.join(args.outdir, f"{stem}_dist_{args.rocs_col}{suffix}.pdf")
        if args.model == "molformer":
            ylabel_model = "MoLFormer"
        else:
            ylabel_model = f"Uni-Mol2 {args.unimol_size}M"
    else:
        cosine_col = f"{args.model}_{args.mode}_cosine"
        euclidean_col = f"{args.model}_{args.mode}_euclidean"
        output_path = os.path.join(args.outdir, f"{args.model}_{args.mode}_dist_{args.rocs_col}{suffix}.pdf")
        if args.mode in ["2D", "3D"]:
            if args.model == "geodiff":
                ylabel_model = f"GeoDiff {args.mode}"
            else:
                ylabel_model = f"MoleculeSDE {args.mode}"
        else:
            if args.model == "geodiff":
                ylabel_model = "GeoDiff Concat"
            else:
                ylabel_model = "MoleculeSDE Concat"

    if args.rocs_col == "3d_similarity_ROCS":
        xlabel = "ROCS 3D Similarity Group"
    elif args.rocs_col == "ShapeTanimoto":
        xlabel = "ROCS Shape Similarity Group"
    elif args.rocs_col == "ColorTanimoto":
        xlabel = "ROCS Color Similarity Group"

    plot_boxpanels(
        df,
        cosine_col=cosine_col,
        euclidean_col=euclidean_col,
        output_path=output_path,
        rocs_col=args.rocs_col,
        rocs_step=0.1,
        xlabel=xlabel,
        ylabel_model=ylabel_model,
        logy=args.logy,
    )
