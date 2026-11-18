"""Per-target differences, leave-one-target-out and the 2x2 stability figure."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ef1 import TARGETS
import paired_stats as PS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASELINES_CSV = os.path.join(REPO_ROOT, "data", "baseline_ef1.csv")

INK, INK_2, RULE = "#0b0b0b", "#52514e", "#c9c8c3"
POS, NEG, BAND, SURFACE = "#2a78d6", "#e34948", "#e5e4df", "#ffffff"

METRICS = [("mean_ef1", "average EF1%"), ("pooled_ef1", "best-pooled EF1%")]
PED_MODES = {"2D": "GeoDiff 2D", "3D": "GeoDiff 3D", "concat": "GeoDiff concat",
             "molformer": "MoLFormer"}


def load_ped(results_dir: str, distance_type: str) -> pd.DataFrame:
    """Per-target average and best-pooled EF1% for every PED variant found."""
    rows: List[Dict] = []
    for backend in ("geodiff", "molformer"):
        stem = os.path.join(results_dir, f"lit_pcba_{backend}_{distance_type}")
        if not os.path.exists(stem + "_pooled.csv"):
            continue
        per_ref = pd.read_csv(stem + "_per_reference.csv")
        pooled = pd.read_csv(stem + "_pooled.csv")
        for mode, label in PED_MODES.items():
            sub = per_ref[per_ref["mode"] == mode]
            pool = pooled[pooled["mode"] == mode]
            if len(sub) == 0:
                continue
            agg = sub.groupby("target", observed=True)["ef1"].mean()
            for target in TARGETS:
                if target not in agg.index:
                    continue
                rows.append({
                    "method": f"{label} {distance_type}", "target": target,
                    "mean_ef1": float(agg[target]),
                    "pooled_ef1": float(pool.loc[pool.target == target, "pooled_ef1"].iloc[0]),
                })
    return pd.DataFrame(rows)


def best_method(df: pd.DataFrame, metric: str) -> str:
    """The PED variant with the highest average over targets for this metric."""
    scores = df.groupby("method", observed=True)[metric].mean()
    return str(scores.idxmax())


def series(df: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    return df[df.method == method].set_index("target")[metric].reindex(TARGETS).to_numpy(float)


def make_figure(ped: pd.DataFrame, base: pd.DataFrame, methods: Dict[str, str],
                baseline: str, out_base: str) -> None:
    diffs, loo = {}, {}
    for metric, _ in METRICS:
        a = series(ped, methods[metric], metric)
        b = series(base, baseline, metric)
        diffs[metric] = a - b
        loo[metric] = PS.leave_one_out(a, b, TARGETS)

    all_d = np.concatenate([diffs[m] for m, _ in METRICS])
    span = all_d.max() - all_d.min()
    xlim0 = (all_d.min() - 0.20 * span, all_d.max() + 0.16 * span)
    all_l = np.concatenate([loo[m].mean_diff_loo.to_numpy() for m, _ in METRICS])
    span_l = all_l.max() - all_l.min()
    xlim1 = (min(all_l.min(), 0.0) - 0.16 * span_l, all_l.max() + 0.16 * span_l)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), dpi=300)
    panel = iter("ABCD")
    ypos = np.arange(len(TARGETS))

    for r, (metric, label) in enumerate(METRICS):
        d = diffs[metric]
        order = np.argsort(d)
        names = [TARGETS[i] for i in order]
        vals = d[order]
        mean_d = float(np.nanmean(d))
        lo, hi = PS.bootstrap_mean_ci(d)

        ax = axes[r, 0]
        ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        ax.barh(ypos, vals, height=0.62,
                color=[POS if v > 0 else NEG for v in vals], zorder=2)
        ax.axvline(0, color=RULE, lw=0.9, ls="--", zorder=3)
        ax.axvline(mean_d, color=INK_2, lw=1.2, zorder=3)
        ax.set_xlim(*xlim0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=8, color=INK)
        ax.annotate(f"mean {mean_d:+.2f}\n95% CI [{lo:+.2f}, {hi:+.2f}]",
                    xy=(0.97, 0.015), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=7.5, color=INK_2)
        off = 0.02 * (xlim0[1] - xlim0[0])
        for i in {int(np.argmin(vals)), int(np.argmax(vals))}:
            ax.text(vals[i] + (off if vals[i] > 0 else -off), ypos[i], f"{vals[i]:+.1f}",
                    va="center", ha="left" if vals[i] > 0 else "right",
                    fontsize=7.5, color=INK_2)
        ax.set_xlabel(f"$\\Delta$ EF1% ({methods[metric]} $-$ {baseline})",
                      fontsize=8.5, color=INK)
        ax.set_title(f"{next(panel)}   Per-target difference — {label}",
                     loc="left", fontsize=9.5, color=INK, pad=6)

        l = loo[metric].set_index("dropped_target").reindex(names)
        ax = axes[r, 1]
        ax.axvline(0, color=RULE, lw=0.9, ls="--", zorder=3)
        ax.axvline(mean_d, color=INK_2, lw=1.2, zorder=3)
        ax.hlines(ypos, np.minimum(mean_d, l.mean_diff_loo),
                  np.maximum(mean_d, l.mean_diff_loo), color=RULE, lw=1.2, zorder=1)
        ax.plot(l.mean_diff_loo, ypos, "o", ms=4.5, color=POS,
                markeredgecolor=SURFACE, markeredgewidth=0.8, ls="none", zorder=4)
        ax.set_xlim(*xlim1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([])
        ax.annotate(f"all {len(TARGETS)} targets: {mean_d:+.2f}", xy=(0.03, 0.015),
                    xycoords="axes fraction", ha="left", va="bottom",
                    fontsize=7.5, color=INK_2,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        ax.set_xlabel("mean $\\Delta$ EF1% with target removed", fontsize=8.5, color=INK)
        ax.set_title(f"{next(panel)}   Leave-one-target-out — {label}",
                     loc="left", fontsize=9.5, color=INK, pad=6)

    for ax in axes.ravel():
        ax.set_ylim(-2.4, len(TARGETS) - 0.3)
        ax.tick_params(axis="x", labelsize=8, colors=INK_2, length=2.5, width=0.6)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.spines["bottom"].set_linewidth(0.7)

    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(out_base + ".png", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True, help="Folder holding the EF1 tables")
    ap.add_argument("--distance_type", default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--baselines_csv", default=DEFAULT_BASELINES_CSV)
    ap.add_argument("--baseline", default="2D ECFP4 similarity")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    ped = load_ped(args.results_dir, args.distance_type)
    if ped.empty:
        raise SystemExit(f"no EF1 tables for {args.distance_type} in {args.results_dir}")
    base = pd.read_csv(args.baselines_csv)
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    methods = {m: best_method(ped, m) for m, _ in METRICS}
    others = sorted(set(base.method) - {args.baseline})

    rows = []
    for metric, label in METRICS:
        a = series(ped, methods[metric], metric)
        results = [PS.compare_paired(a, series(base, b, metric), methods[metric], b)
                   for b in [args.baseline] + others]
        PS.holm(results)
        loo = PS.leave_one_out(a, series(base, args.baseline, metric), TARGETS)
        flips = loo.loc[loo.sign_flips, "dropped_target"].tolist()
        for res in results:
            row = res.as_dict()
            row.update({"metric": label, "avg_ef1": float(np.nanmean(a)),
                        "sign_reversed_by": ", ".join(flips) if res.method_b == args.baseline
                        and flips else ("none" if res.method_b == args.baseline else "")})
            rows.append(row)

    table = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"stability_{args.distance_type}.csv")
    table.to_csv(path, index=False)

    show = ["metric", "method_a", "method_b", "avg_ef1", "mean_diff", "ci_lo", "ci_hi",
            "n_higher", "n_lower", "n_tied", "wilcoxon_p", "p_holm", "sign_reversed_by"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig_base = os.path.join(out_dir, f"loo_stability_{args.distance_type}")
    make_figure(ped, base, methods, args.baseline, fig_base)
    print(f"\nwrote {path}\nwrote {fig_base}.pdf / .png")


if __name__ == "__main__":
    main()
