"""The three geometry diagnostics, shown for every reported embedding space.

One row per space, one column per diagnostic: the per-dimension mean and spread,
the distribution of embedding norms, and the distribution of distances from the
library to the reference ligand.  The three failures are not detected by the same
column, which is why all three checks are needed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ampc200k as A

SPACES = [
    ("GeoDiff 2D", "GeoDiff 2D"),
    ("GeoDiff 3D", "GeoDiff 3D"),
    ("MoLFormer", "MoLFormer"),
    ("MoleculeSDE 2D", "MoleculeSDE 2D"),
    ("MoleculeSDE 3D", "MoleculeSDE 3D"),
    ("Uni-Mol2", "Uni-Mol2 164m atom"),
]

C_BAND = "#5B8DB8"
C_LINE = "#1F4E79"
C_MARK = "#C0392B"
N_BINS = 100


def panel_dims(ax, x):
    mu = x.mean(0)
    sd = x.std(0, ddof=1)
    dims = np.arange(len(mu))
    ax.fill_between(dims, mu - sd, mu + sd, color=C_BAND, alpha=0.45, linewidth=0,
                    label=r"$\pm$ 1 s.d.")
    ax.plot(dims, mu, color=C_LINE, linewidth=0.6, label="mean")
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")

    lo = np.percentile(mu - sd, 0.1)
    hi = np.percentile(mu + sd, 99.9)
    pad = 0.18 * max(hi - lo, 1e-9)
    ax.set_ylim(lo - pad, hi + pad)

    v = sd ** 2
    share = v.max() / v.sum()
    dom = int(np.argmax(v))
    off = (mu < lo - pad) | (mu > hi + pad)
    if off.any():
        ax.plot(dims[off], np.clip(mu[off], lo - pad, hi + pad), "v",
                color=C_MARK, markersize=6, clip_on=False, linestyle="none")
        ax.text(0.97, 0.06, f"dim {dom} off scale\n(mean {mu[dom]:+.0f})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color=C_MARK)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85, handlelength=1.8,
              borderpad=0.35, labelspacing=0.3)
    ax.text(0.03, 0.95, f"max share = {share:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="gray", linewidth=0.5))
    return share


def panel_norm(ax, x):
    nrm = np.linalg.norm(x, axis=1)
    med = np.median(nrm)
    p999 = np.percentile(nrm, 99.9)
    tail = p999 / med
    bins = np.linspace(nrm.min(), nrm.max(), 120)
    ax.hist(nrm, bins=bins, color=C_BAND, edgecolor="none")
    ax.set_yscale("log")
    # Per-row limits: the printed ratio carries the cross-row comparison, so the
    # axis is free to show each distribution's actual shape.
    pad = 0.03 * (nrm.max() - nrm.min())
    ax.set_xlim(nrm.min() - pad, nrm.max() + pad)
    ax.axvline(med, color=C_LINE, lw=1.0, ls="--", label="median")
    ax.axvline(p999, color=C_MARK, lw=1.0, ls="-.", label="$p_{99.9}$")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85, handlelength=1.8,
              borderpad=0.35, labelspacing=0.3)
    ax.text(0.03, 0.95, f"$p_{{99.9}}$/median = {tail:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="gray", linewidth=0.5))
    return tail


def panel_dist(ax, x, ref):
    d = np.linalg.norm(x - ref, axis=1)
    lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
    h, edges = np.histogram(d, bins=N_BINS, range=(lo, hi))
    ax.bar(edges[:-1], h, width=np.diff(edges), align="edge",
           color=C_BAND, edgecolor="none")
    empty = h == 0
    for i in np.flatnonzero(empty):
        ax.axvspan(edges[i], edges[i + 1], color=C_MARK, alpha=0.35, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    gapless = "yes" if not empty.any() else "no"
    ax.text(0.03, 0.95,
            f"gapless: {gapless}" + ("" if gapless == "yes" else f"  ({empty.sum()}/{N_BINS} bins)"),
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            color=C_LINE if gapless == "yes" else C_MARK,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="gray", linewidth=0.5))
    return gapless


def main(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(len(SPACES), 3, figsize=(12.5, 15.5))
    rows = []
    for i, (label, key) in enumerate(SPACES):
        x = A.load_library(key).astype(np.float64)
        ref = A.load_reference(key).astype(np.float64)
        s = panel_dims(axes[i, 0], x)
        t = panel_norm(axes[i, 1], x)
        g = panel_dist(axes[i, 2], x, ref)
        rows.append((label, x.shape[1], s, t, g))

        axes[i, 0].set_ylabel(f"{label}\nDimension value", fontsize=13)
        axes[i, 1].set_ylabel("#molecules", fontsize=13)
        axes[i, 2].set_ylabel("#molecules", fontsize=13)
        if i == len(SPACES) - 1:
            axes[i, 0].set_xlabel("Embedding dimension", fontsize=13)
            axes[i, 1].set_xlabel(r"Embedding norm $\|x\|$", fontsize=13)
            axes[i, 2].set_xlabel("Euclidean PED to reference", fontsize=13)
        for j in range(3):
            axes[i, j].tick_params(labelsize=11)

    axes[0, 0].set_title("Per-dimension mean $\\pm$ s.d.", fontsize=14)
    axes[0, 1].set_title("Embedding norms", fontsize=14)
    axes[0, 2].set_title("Distances to the reference", fontsize=14)

    fig.tight_layout()
    out = os.path.join(outdir, "emb_diagnostics_grid.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=110, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"\n{'space':18s} {'d':>5s} {'maxshare':>9s} {'p99.9/med':>10s} {'gapless':>8s}")
    for label, d, s, t, g in rows:
        print(f"{label:18s} {d:5d} {s:9.3f} {t:10.2f} {g:>8s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=str, default="./figs")
    main(p.parse_args().outdir)
