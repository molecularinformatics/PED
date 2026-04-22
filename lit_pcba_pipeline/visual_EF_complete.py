#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


targets_a = ['ADRB2', 'ESR1_ant', 'ESR1_ago', 'MAPK1', 'MTORC1', 'PKM2', 'PPARG', 'TP53']
targets_b = ['ALDH1', 'FEN1', 'GBA', 'IDH1', 'KAT2A', 'OPRK1', 'VDR']
targets = targets_a + targets_b

# 7 methods in plotting order
method_order = [
    "Shape only",
    "Color only",
    "Combination",
    "GeoDiff 3D",
    "GeoDiff 2D",
    "GeoDiff Concat",
    "MoLFormer",
]

# color design: Roshambo red shades, GeoDiff blue shades, MoLFormer green
method_colors = {
    "Shape only": "#bfedc1",      # light green
    "Color only": "#53b058",      # medium green
    "Combination": "#15751a",     # dark green

    "GeoDiff 3D": "#8ac0f2",      # light blue
    "GeoDiff 2D": "#438cd0",      # medium blue
    "GeoDiff Concat": "#235787",  # dark blue

    "MoLFormer": "#d15353",       # dark red
}

# label mapping for Roshambo sheets
roshambo_sheet_map = {
    "Shape only": "litpcba_shape_EF1_results",
    "Color only": "litpcba_color_EF1_results",
    "Combination": "litpcba_combination_EF1_results",
}


def parse_list_cell(v):
    """
    Parse a cell that may be:
    - a string representation of a list
    - a scalar float/int
    - NaN
    Return a Python list of floats.
    """
    if v is None or pd.isna(v):
        return []

    if isinstance(v, list):
        return [float(x) for x in v]

    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "nan":
            return []
        if s.startswith("[") and s.endswith("]"):
            parsed = ast.literal_eval(s)
            return [float(x) for x in parsed]
        # fallback: treat as scalar string
        return [float(s)]

    # scalar numeric
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [float(v)]

    raise ValueError(f"Unsupported cell type: {type(v)} | value={v}")


def load_roshambo_data(xlsx_path):
    """
    Return:
      per_ref_data[method][target] = list of ef1
      pooled_data[method][target] = float
    """
    per_ref_data = {m: {} for m in roshambo_sheet_map}
    pooled_data = {m: {} for m in roshambo_sheet_map}

    for method, sheet_name in roshambo_sheet_map.items():
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

        if not {"Target", "max_pooled_EF1", "individual_EF1s"}.issubset(df.columns):
            raise ValueError(
                f"Sheet {sheet_name} does not have expected columns. "
                f"Found: {df.columns.tolist()}"
            )

        for _, row in df.iterrows():
            target = str(row["Target"]).strip()
            per_ref_data[method][target] = parse_list_cell(row["individual_EF1s"])
            pooled_data[method][target] = float(row["max_pooled_EF1"])

    return per_ref_data, pooled_data


def load_geodiff_data(dist_type, geodiff_dir):
    """
    per_reference columns expected: target, mode, reference_idx, ef1
    pooled columns expected: target, mode, pooled_ef1
    modes used: 3D, 2D, concat
    """
    per_ref_path = os.path.join(
        geodiff_dir, f"lit_pcba_geodiff_{dist_type}_per_reference.csv"
    )
    pooled_path = os.path.join(
        geodiff_dir, f"lit_pcba_geodiff_{dist_type}_pooled.csv"
    )

    df_ref = pd.read_csv(per_ref_path)
    df_pool = pd.read_csv(pooled_path)

    mode_to_method = {
        "3D": "GeoDiff 3D",
        "2D": "GeoDiff 2D",
        "concat": "GeoDiff Concat",
    }

    per_ref_data = {m: {} for m in mode_to_method.values()}
    pooled_data = {m: {} for m in mode_to_method.values()}

    for mode, method in mode_to_method.items():
        sub = df_ref[df_ref["mode"] == mode].copy()
        grouped = sub.groupby("target")["ef1"].apply(list)
        for target, vals in grouped.items():
            per_ref_data[method][str(target).strip()] = [float(x) for x in vals]

        subp = df_pool[df_pool["mode"] == mode].copy()
        for _, row in subp.iterrows():
            pooled_data[method][str(row["target"]).strip()] = float(row["pooled_ef1"])

    return per_ref_data, pooled_data


def load_molformer_data(dist_type, molformer_dir):
    """
    per_reference columns expected: target, mode, reference_idx, ef1
    pooled columns expected: target, mode, pooled_ef1

    mode is expected to be 'molformer', but if there is only one mode
    present, we still use all rows.
    """
    per_ref_path = os.path.join(
        molformer_dir, f"lit_pcba_molformer_{dist_type}_per_reference.csv"
    )
    pooled_path = os.path.join(
        molformer_dir, f"lit_pcba_molformer_{dist_type}_pooled.csv"
    )

    df_ref = pd.read_csv(per_ref_path)
    df_pool = pd.read_csv(pooled_path)

    if "mode" in df_ref.columns and "molformer" in set(df_ref["mode"].astype(str)):
        df_ref = df_ref[df_ref["mode"].astype(str) == "molformer"].copy()

    if "mode" in df_pool.columns and "molformer" in set(df_pool["mode"].astype(str)):
        df_pool = df_pool[df_pool["mode"].astype(str) == "molformer"].copy()

    per_ref_data = {"MoLFormer": {}}
    pooled_data = {"MoLFormer": {}}

    grouped = df_ref.groupby("target")["ef1"].apply(list)
    for target, vals in grouped.items():
        per_ref_data["MoLFormer"][str(target).strip()] = [float(x) for x in vals]

    for _, row in df_pool.iterrows():
        pooled_data["MoLFormer"][str(row["target"]).strip()] = float(row["pooled_ef1"])

    return per_ref_data, pooled_data


def merge_all_data(xlsx_path, dist_type, geodiff_dir, molformer_dir):
    roshambo_ref, roshambo_pool = load_roshambo_data(xlsx_path)
    geodiff_ref, geodiff_pool = load_geodiff_data(dist_type, geodiff_dir)
    molformer_ref, molformer_pool = load_molformer_data(dist_type, molformer_dir)

    all_ref = {}
    all_pool = {}

    for method in method_order:
        all_ref[method] = {}
        all_pool[method] = {}

    for src in [roshambo_ref, geodiff_ref, molformer_ref]:
        for method, dd in src.items():
            all_ref[method].update(dd)

    for src in [roshambo_pool, geodiff_pool, molformer_pool]:
        for method, dd in src.items():
            all_pool[method].update(dd)

    return all_ref, all_pool


def sanity_check(all_ref, all_pool):
    for method in method_order:
        for t in targets:
            if t not in all_ref[method]:
                print(f"[WARN] missing per-reference data: method={method}, target={t}")
                all_ref[method][t] = []

            if t not in all_pool[method]:
                print(f"[WARN] missing pooled data: method={method}, target={t}")
                all_pool[method][t] = np.nan


def make_plot(all_ref, all_pool, dist_type, out_path):
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    x_base = np.arange(1, len(targets) + 1)

    # 7 boxplots per target
    offsets = np.array([-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30])
    width = 0.09

    ax = plt.gca()

    # background for targets_a
    ax.axvspan(0, len(targets_a) + 0.5, color='lightblue', alpha=0.3, zorder=0)

    # horizontal guide lines
    ax.axhline(1, color='red', linestyle='--', zorder=1)
    ax.axhline(2, color='blue', linestyle='--', zorder=1)

    for i, method in enumerate(method_order):
        positions = x_base + offsets[i]
        data = [all_ref[method][t] for t in targets]
        pooled_vals = [all_pool[method][t] for t in targets]
        color = method_colors[method]

        bp = plt.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            zorder=2,
            flierprops=dict(
                marker='x',
                color=color,
                markeredgecolor=color,
                markersize=6,
            ),
            manage_ticks=False,
        )

        for box in bp['boxes']:
            box.set_facecolor(color)
            box.set_edgecolor('black')
            box.set_linewidth(1.0)

        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.0)

        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.0)

        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.2)

        # pooled marker
        plt.scatter(
            positions,
            pooled_vals,
            marker='o',
            c=color,
            edgecolor='black',
            s=30,
            zorder=10
        )

    # plt.ylim(*ylim)
    plt.xlim(left=0.4, right=len(targets) + 0.6)
    plt.xticks(x_base, labels=targets, rotation=90)
    plt.xlabel('Target')
    plt.ylabel('Enrichment Factor 1%')

    # custom legend
    legend_handles = [
        Patch(facecolor=method_colors[m], edgecolor='black', label=m)
        for m in method_order
    ]
    legend_handles.append(
        Line2D(
            [0], [0],
            marker='o',
            color='white',
            markerfacecolor='gray',
            markeredgecolor='black',
            markersize=6,
            linestyle='None',
            label='Best-pooled'
        )
    )

    plt.legend(
        handles=legend_handles,
        loc='upper right',
        ncol=3,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx",
        type=str,
        default="./roshambo2.xlsx",
        help="Path to Roshambo xlsx file."
    )
    parser.add_argument(
        "--geodiff_dir",
        type=str,
        default="/mnt/depts/dept01/misd/molml/embeddings/figures_geodiff",
        help="Directory containing GeoDiff per_reference and pooled csv files."
    )
    parser.add_argument(
        "--molformer_dir",
        type=str,
        default="/mnt/depts/dept01/misd/molml/embeddings/figures_molformer",
        help="Directory containing MoLFormer per_reference and pooled csv files."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./figs",
        help="Output directory."
    )
    parser.add_argument(
        "--dist_types",
        nargs="+",
        default=["cosine", "euclidean"],
        help="Distance types to plot."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for dist_type in args.dist_types:
        all_ref, all_pool = merge_all_data(
            xlsx_path=args.xlsx,
            dist_type=dist_type,
            geodiff_dir=args.geodiff_dir,
            molformer_dir=args.molformer_dir,
        )
        sanity_check(all_ref, all_pool)

        out_path = os.path.join(
            args.out_dir,
            f"litpcba_7methods_{dist_type}.pdf"
        )

        make_plot(
            all_ref=all_ref,
            all_pool=all_pool,
            dist_type=dist_type,
            out_path=out_path,
        )
        print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
