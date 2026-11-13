import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def ef1_from_distances(distances: np.ndarray, labels: np.ndarray, top_frac: float = 0.01) -> float:
    valid = np.isfinite(distances)
    distances = distances[valid]
    labels = labels[valid]

    n = len(labels)
    if n == 0:
        return np.nan

    n_top = max(1, int(np.ceil(n * top_frac)))
    order = np.argsort(distances)
    top_labels = labels[order[:n_top]]

    n_actives = int(labels.sum())
    if n_actives == 0:
        return np.nan

    hit_rate_top = top_labels.mean()
    hit_rate_all = labels.mean()
    if hit_rate_all == 0:
        return np.nan

    return float(hit_rate_top / hit_rate_all)


def find_distance_csvs(out_root: str, filename_pattern: str = r".*\.csv$") -> Dict[str, str]:
    csvs = {}
    pat = re.compile(filename_pattern)

    for root, _, files in os.walk(out_root):
        if os.path.basename(root) != "distance":
            continue
        for fn in files:
            if not pat.match(fn):
                continue
            path = os.path.join(root, fn)
            target = os.path.basename(os.path.dirname(root))
            csvs[target] = path
    return csvs


def infer_reference_columns(df: pd.DataFrame, mode: str, distance_type: str) -> List[str]:
    prefix = f"{mode}_{distance_type}_dist_"
    pooled = f"{mode}_{distance_type}_dist_min"
    cols = [c for c in df.columns if c.startswith(prefix) and c != pooled]
    return sorted(cols)


def compute_target_results(
    csv_path: str,
    modes: List[str],
    distance_type: str,
    label_col: str = "label",
    top_frac: float = 0.01,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")

    # # Debug
    # print(f"CSV: {csv_path}")
    # print(f"Columns: {df.columns.tolist()}")
    # print(f"Looking for distance_type: {distance_type}")

    labels = df[label_col].to_numpy(dtype=int)

    per_mode_ref_ef1 = {}
    per_mode_pooled_ef1 = {}

    for mode in modes:
        ref_cols = infer_reference_columns(df, mode, distance_type)
        pooled_col = f"{mode}_{distance_type}_dist_min"

        ef1s = []
        for col in ref_cols:
            ef1 = ef1_from_distances(df[col].to_numpy(dtype=float), labels, top_frac=top_frac)
            if np.isfinite(ef1):
                ef1s.append(ef1)
        per_mode_ref_ef1[mode] = ef1s

        if pooled_col in df.columns:
            pooled_ef1 = ef1_from_distances(df[pooled_col].to_numpy(dtype=float), labels, top_frac=top_frac)
        else:
            pooled_ef1 = np.nan
        per_mode_pooled_ef1[mode] = pooled_ef1

    return per_mode_ref_ef1, per_mode_pooled_ef1


def build_plot_inputs(
    csv_map: Dict[str, str],
    modes: List[str],
    distance_type: str,
    label_col: str,
    top_frac: float,
) -> Tuple[List[str], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]]]:
    figure3_order = [
        "ADRB2", "ESR1_ant", "ESR1_ago", "MAPK1", "MTORC1",
        "PKM2", "PPARG", "TP53",
        "ALDH1", "FEN1", "GBA", "IDH1", "KAT2A", "OPRK1", "VDR"
    ]
    targets = [t for t in figure3_order if t in csv_map]

    all_ref = {}
    all_pooled = {}

    for target in targets:
        ref_res, pooled_res = compute_target_results(
            csv_map[target],
            modes=modes,
            distance_type=distance_type,
            label_col=label_col,
            top_frac=top_frac,
        )
        all_ref[target] = ref_res
        all_pooled[target] = pooled_res

    return targets, all_ref, all_pooled


def plot_figure3_style(
    targets: List[str],
    all_ref: Dict[str, Dict[str, List[float]]],
    all_pooled: Dict[str, Dict[str, float]],
    modes: List[str],
    distance_type: str,
    out_path: str,
    tractable_targets: List[str],
    title: str,
):
    mode_colors = {
        "3D": "#A6EC9A",
        "2D": "#97CCE8",
        "concat": "#E18683",
        "molformer": "#E18683",
    }

    n_targets = len(targets)
    n_modes = len(modes)

    # fig_width = max(7.5, 0.95 * n_targets + 2.2)
    fig_width = max(7.5, 0.65 * n_targets)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    if n_targets <= 5:
        bar_width = 0.07
        within_gap = 0.017
        target_gap = 0.14
    elif n_targets <= 10:
        bar_width = 0.05
        within_gap = 0.01
        target_gap = 0.12
    else:
        bar_width = 0.03
        within_gap = 0.005
        target_gap = 0.1

    target_centers = []
    mode_positions = {m: [] for m in modes}
    target_lefts = []
    target_rights = []

    current_x = 0.0
    group_width = n_modes * bar_width + (n_modes - 1) * within_gap

    for _target in targets:
        left = current_x

        if n_modes == 1:
            # put it at the center
            pos = left + group_width / 2.0
            mode_positions[modes[0]].append(pos)
        else:
            for i, mode in enumerate(modes):
                pos = left + i * (bar_width + within_gap)
                mode_positions[mode].append(pos)

        right = left + group_width
        center = 0.5 * (left + right)

        target_lefts.append(left)
        target_rights.append(right)
        target_centers.append(center)

        current_x += group_width + target_gap

    tractable_set = set(tractable_targets)
    tractable_mask = [t in tractable_set for t in targets]

    spans = []
    start_idx = None
    for i, is_tractable in enumerate(tractable_mask):
        if is_tractable and start_idx is None:
            start_idx = i
        if (not is_tractable) and (start_idx is not None):
            spans.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        spans.append((start_idx, len(targets) - 1))

    for s, e in spans:
        left = target_lefts[s] - target_gap * 0.40
        right = target_rights[e] + target_gap * 0.40
        ax.axvspan(left, right, color="#d8ecf3", alpha=0.85, zorder=0)

    legend_handles = []

    for mode in modes:
        positions = mode_positions[mode]
        data = [all_ref[t][mode] if len(all_ref[t][mode]) > 0 else [np.nan] for t in targets]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=bar_width,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            boxprops=dict(edgecolor="black", linewidth=1.0),
            flierprops=dict(
                marker="x",
                markersize=6,
                markeredgewidth=2,
                markerfacecolor=mode_colors[mode],
                markeredgecolor=mode_colors[mode],
            ),
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(mode_colors[mode])

        pooled_y = [all_pooled[t][mode] for t in targets]
        ax.scatter(
            positions,
            pooled_y,
            s=90,
            color=mode_colors[mode],
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

        legend_handles.append(
            Patch(facecolor=mode_colors[mode], edgecolor="black", label=mode)
        )

    ax.axhline(1.0, linestyle="--", color="red", linewidth=2, alpha=0.8)
    ax.axhline(2.0, linestyle="--", color="blue", linewidth=2, alpha=0.8)

    ax.set_xticks(target_centers)
    ax.set_xticklabels(targets, rotation=90, fontsize=12)
    ax.set_ylabel("Enrichment factor 1%", fontsize=15)
    ax.set_xlabel("Target", fontsize=15)
    ax.set_title(title, fontsize=15)

    marker_handle = plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Maximum-pooled",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=9,
    )
    ax.legend(
        handles=legend_handles + [marker_handle],
        fontsize=12,
        frameon=True,
    )

    ax.tick_params(axis="y", labelsize=12)

    xpad = max(0.18, target_gap * 0.7)
    ax.set_xlim(target_lefts[0] - xpad, target_rights[-1] + xpad)

    ymax = 0.0
    for t in targets:
        for mode in modes:
            vals = list(all_ref[t][mode])
            pooled = all_pooled[t][mode]
            if np.isfinite(pooled):
                vals.append(pooled)
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) > 0:
                ymax = max(ymax, max(vals))

    ax.set_ylim(0, max(4.0, ymax * 1.08))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_tables(
    targets: List[str],
    all_ref: Dict[str, Dict[str, List[float]]],
    all_pooled: Dict[str, Dict[str, float]],
    modes: List[str],
    out_prefix: str,
):
    rows_ref = []
    rows_pooled = []
    rows_summary = []

    for target in targets:
        for mode in modes:
            for i, ef1 in enumerate(all_ref[target][mode]):
                rows_ref.append({
                    "target": target,
                    "mode": mode,
                    "reference_idx": i,
                    "ef1": ef1,
                })
            rows_pooled.append({
                "target": target,
                "mode": mode,
                "pooled_ef1": all_pooled[target][mode],
            })

    ref_df = pd.DataFrame(rows_ref)
    pooled_df = pd.DataFrame(rows_pooled)

    for mode in modes:
        mode_vals = pooled_df.loc[pooled_df["mode"] == mode, "pooled_ef1"].dropna().to_numpy()
        rows_summary.append({
            "mode": mode,
            "mean_pooled_ef1": float(np.mean(mode_vals)) if len(mode_vals) else np.nan,
            "median_pooled_ef1": float(np.median(mode_vals)) if len(mode_vals) else np.nan,
            "n_targets": int(len(mode_vals)),
        })

    summary_df = pd.DataFrame(rows_summary)

    ref_df.to_csv(out_prefix + "_per_reference.csv", index=False)
    pooled_df.to_csv(out_prefix + "_pooled.csv", index=False)
    summary_df.to_csv(out_prefix + "_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--distance_type", nargs="+", default=["cosine", "euclidean"], choices=["cosine", "euclidean"])
    parser.add_argument("--modes", nargs="+", default=["2D", "3D", "concat"], choices=["2D", "3D", "concat"])
    parser.add_argument("--model", type=str, default="geodiff", choices=["geodiff", "molformer"], help="Model that generates embeddings")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--top_frac", type=float, default=0.01)
    parser.add_argument("--tractable_targets", nargs="*", default=[])
    parser.add_argument("--filename_pattern", type=str, default=r".*geodiff_distances\.csv$")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    csv_map = find_distance_csvs(args.out_root, filename_pattern=args.filename_pattern)
    if len(csv_map) == 0:
        raise ValueError(f"No distance CSVs found under {args.out_root}")

    if args.model == "molformer":
        args.modes = ["molformer"]

    for distance_type in args.distance_type:
        targets, all_ref, all_pooled = build_plot_inputs(
            csv_map=csv_map,
            modes=args.modes,
            distance_type=distance_type,
            label_col=args.label_col,
            top_frac=args.top_frac,
        )

        fig_dir = os.path.join(args.out_root, f"figures_{args.model}")
        os.makedirs(fig_dir, exist_ok=True)

        title = args.title
        if title is None:
            title = f"LIT-PCBA Figure 3-style EF1% by target ({distance_type})"

        out_prefix = os.path.join(fig_dir, f"lit_pcba_{args.model}_{distance_type}")
        plot_figure3_style(
            targets=targets,
            all_ref=all_ref,
            all_pooled=all_pooled,
            modes=args.modes,
            distance_type=distance_type,
            out_path=out_prefix + ".png",
            tractable_targets=args.tractable_targets,
            title=title,
        )

        save_summary_tables(
            targets=targets,
            all_ref=all_ref,
            all_pooled=all_pooled,
            modes=args.modes,
            out_prefix=out_prefix,
        )

        print(f"Saved figure: {out_prefix}.png")
        print(f"Saved tables: {out_prefix}_per_reference.csv, {out_prefix}_pooled.csv, {out_prefix}_summary.csv")


if __name__ == "__main__":
    main()
