"""Uni-Mol2 dimension 428 and distances to the AmpC reference ligand."""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

import ampc200k as A

SPACE = "Uni-Mol2 164m atom"
ROCS = "3d_similarity_ROCS"

# The two modes of the dominant coordinate are separated by an empty band; any
# cut inside it gives the same split.
MODE_CUT = -470.0

C_MAJOR = "#2B7BBA"
C_MINOR = "#D1495B"
C_MARK = "#C0392B"


def dim_stats(x: np.ndarray):
    return x.mean(0), x.std(0, ddof=1)


def panel_dims(ax, mu, sd, dom, zoom: bool):
    dims = np.arange(len(mu))
    ax.fill_between(dims, mu - sd, mu + sd, color=C_MAJOR, alpha=0.45, linewidth=0,
                    label=r"mean $\pm$ s.d.")
    ax.plot(dims, mu, color="#1F4E79", linewidth=0.7, label="mean")
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Embedding dimension", fontsize=15)
    ax.set_ylabel("Dimension value", fontsize=15)
    if zoom:
        lo = np.percentile(mu - sd, 1.0)
        hi = np.percentile(mu + sd, 99.0)
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.plot([dom], [lo - pad], "v", color=C_MARK, markersize=7, clip_on=False)
        ax.set_title("B. Other dimensions in detail", fontsize=16)
        ax.text(0.98, 0.03, f"Dimension {dom} is below scale",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=13, color=C_MARK)
    else:
        ax.annotate(f"dimension {dom}\nmean {mu[dom]:.0f}, s.d. {sd[dom]:.0f}",
                    xy=(dom, mu[dom]), xytext=(0.52, 0.42), textcoords="axes fraction",
                    fontsize=13, color=C_MARK,
                    arrowprops=dict(arrowstyle="->", color=C_MARK, lw=1.0))
        ax.set_title("A. Variance across dimensions", fontsize=16)
        ax.text(0.98, 0.05,
                f"Dimension {dom}: {sd[dom] ** 2 / (sd ** 2).sum():.1%} of variance",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                          edgecolor="gray"))
    ax.legend(loc="upper right", fontsize=12, framealpha=0.85)


def main(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    df = A.aligned()
    x = A.load_library(SPACE)
    ref = A.load_reference(SPACE)
    y = df[ROCS].to_numpy()

    mu, sd = dim_stats(x)
    dom = int(np.argmax(sd))                      # the dimension with the largest spread
    c = x[:, dom]
    minor = c < MODE_CUT
    major = ~minor

    sq = (x - ref) ** 2
    d = np.sqrt(sq.sum(1))
    d_dom = np.abs(c - ref[dom])                  # the dominant dimension's own term
    share = sq[:, dom] / sq.sum(1)
    # Panels D and E share this distance axis so they can be read together.
    DIST_LIM = (0.0, float(max(d.max(), d_dom.max())) * 1.05)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    panel_dims(axes[0, 0], mu, sd, dom, zoom=False)
    panel_dims(axes[0, 1], mu, sd, dom, zoom=True)

    # ---- C: the dominant dimension is bimodal, and the reference is in the
    # smaller mode, so every major molecule is pushed to the same side of it.
    ax = axes[0, 2]
    bins = np.linspace(c.min(), c.max(), 160)
    ax.hist(c[major], bins=bins, color=C_MAJOR, label=f"Larger group ({major.mean():.1%})")
    ax.hist(c[minor], bins=bins, color=C_MINOR, label=f"Smaller group ({minor.mean():.1%})")
    ax.axvline(ref[dom], color="black", lw=1.4, ls="--")
    ax.axvline(MODE_CUT, color="gray", lw=1.0, ls="-.")
    ax.text(MODE_CUT, 1.2, " cut", color="gray", fontsize=12, va="bottom")
    ax.annotate("reference ligand", xy=(ref[dom], ax.get_ylim()[1] * 0.55),
                xytext=(0.07, 0.72), textcoords="axes fraction", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlabel(f"Value of dimension {dom}", fontsize=15)
    ax.set_ylabel("#molecules", fontsize=15)
    ax.set_title(f"C. Two groups of dimension {dom}", fontsize=16)
    ax.legend(fontsize=12, loc="upper left")

    sc = dict(s=1.2, linewidths=0, rasterized=True, alpha=0.25)

    # ---- D: the scatter, with the two groups colored.
    ax = axes[1, 0]
    ax.scatter(y[major], d[major], color=C_MAJOR, **sc)
    ax.scatter(y[minor], d[minor], color=C_MINOR, **sc)
    ax.set_ylim(DIST_LIM)
    ax.set_xlabel("ROCS combination similarity", fontsize=15)
    ax.set_ylabel("Uni-Mol2 Euclidean PED", fontsize=15)
    ax.set_title("D. PED and ROCS similarity", fontsize=16)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_MAJOR, label="Larger group"),
                       Line2D([], [], marker="o", ls="", color=C_MINOR, label="Smaller group")],
              fontsize=12, loc="lower left")
    ax.text(0.98, 0.97,
            f"All molecules $r$ = {stats.pearsonr(d, y)[0]:+.3f}\n"
            f"Larger group $r$ = {stats.pearsonr(d[major], y[major])[0]:+.3f}\n"
            f"Smaller group $r$ = {stats.pearsonr(d[minor], y[minor])[0]:+.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    # ---- E: the distance is almost exactly the dominant dimension's own term.
    ax = axes[1, 1]
    ax.scatter(d_dom[major], d[major], color=C_MAJOR, **sc)
    ax.scatter(d_dom[minor], d[minor], color=C_MINOR, **sc)
    ax.plot(DIST_LIM, DIST_LIM, color="gray", lw=0.9, ls=":")
    ax.set_xlim(DIST_LIM)
    ax.set_ylim(DIST_LIM)   # shared with panel D so the two read on one scale
    ax.set_xlabel(f"Distance to reference (dimension {dom})", fontsize=15)
    ax.set_ylabel("Full Euclidean PED (768 dimensions)", fontsize=15)
    ax.set_title("E. One-dimensional and full distances", fontsize=16)
    ax.text(0.03, 0.97,
            f"Median share of squared distance\n"
            f"Larger group {np.median(share[major]):.1%}, smaller group {np.median(share[minor]):.1%}\n"
            f"$r(d,\\,|x_{{{dom}}}-\\mathrm{{ref}}|)$ = {stats.pearsonr(d, d_dom)[0]:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=13,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    # ---- F: subtract each mode's own mean distance and nothing else.  The
    # within-group structure is untouched; only the offset between them goes.
    ax = axes[1, 2]
    dz = d.copy()
    dz[minor] -= d[minor].mean()
    dz[major] -= d[major].mean()
    ax.scatter(y[major], dz[major], color=C_MAJOR, **sc)
    ax.scatter(y[minor], dz[minor], color=C_MINOR, **sc)
    ax.set_xlabel("ROCS combination similarity", fontsize=15)
    ax.set_ylabel("Group-centered Euclidean PED", fontsize=15)
    ax.set_title("F. PED after subtracting group means", fontsize=16)
    gap = d[major].mean() - d[minor].mean()
    ax.text(0.98, 0.97,
            f"$r$ = {stats.pearsonr(dz, y)[0]:+.3f}  "
            f"(was {stats.pearsonr(d, y)[0]:+.3f})\n"
            f"$\\rho$ = {stats.spearmanr(dz, y)[0]:+.3f}  "
            f"(was {stats.spearmanr(d, y)[0]:+.3f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    for ax in axes.flat:
        ax.tick_params(labelsize=12)
    fig.tight_layout()
    out = os.path.join(outdir, "unimol_dim428_story.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out}")

    print(f"\ndominant dimension: {dom}  "
          f"(mean {mu[dom]:+.1f}, s.d. {sd[dom]:.1f}, "
          f"{sd[dom] ** 2 / (sd ** 2).sum():.1%} of the variance across dimensions)")
    print(f"reference value: {ref[dom]:+.1f}")
    print(f"Smaller group n = {minor.sum()} ({minor.mean():.2%}), larger group n = {major.sum()}")
    print(f"distance gap between groups: {gap:.1f} "
          f"= {gap / d[major].std():.1f} x the within-major-cluster s.d.")
    gy = y[major].mean() - y[minor].mean()
    print(f"ROCS gap between groups:  {gy:+.3f} "
          f"= {gy / y[major].std():+.2f} x the within-major-cluster s.d.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=str, default="./figs")
    main(p.parse_args().outdir)
