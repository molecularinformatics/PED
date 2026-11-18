"""Per-coordinate mean and standard deviation of each pretrained embedding space.

Measured on the AmpC 200k library, the same molecule set as the correlation study,
so the spaces are directly comparable.  Per-coordinate statistics do not depend on
row order, so the stored matrices are read as they are rather than through the
``ampc200k`` row alignment.

Two things the figure is meant to settle: whether a space has an individual runaway
coordinate, and how large the common offset shared by every molecule is relative to
the spread around it.  A space can look well behaved coordinate by coordinate and
still be nearly one-dimensional, because the dominant direction need not be axis
aligned; the printed table reports the centered PCA spectrum alongside.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "results" / "ampc200k" / "library" / "embeddings"

# (panel label, folder) for the spaces the manuscript reports.
SPACES = [
    ("GeoDiff 2D", "geodiff_2D"),
    ("GeoDiff 3D", "geodiff_3D"),
    ("MoLFormer", "molformer"),
    ("MoleculeSDE 2D", "sde_2D"),
    ("MoleculeSDE 3D", "sde_3D"),
    ("Uni-Mol2 164M", "unimol_164m_atom"),
]


def dim_stats(folder: str, chunk: int = 20000):
    """Per-coordinate mean and sample standard deviation, in float64."""
    x = np.load(ROOT / folder / "emb.npy", mmap_mode="r")
    n, d = x.shape
    s1 = np.zeros(d)
    s2 = np.zeros(d)
    for i in range(0, n, chunk):
        b = np.asarray(x[i:i + chunk], dtype=np.float64)
        s1 += b.sum(0)
        s2 += (b ** 2).sum(0)
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 0.0) * n / (n - 1)
    return mu, np.sqrt(var), n


def pca_spectrum(folder: str, n_sample: int = 20000, seed: int = 0):
    """PC1 share and participation ratio of the mean-centered covariance."""
    x = np.load(ROOT / folder / "emb.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), min(n_sample, len(x)), replace=False))
    a = np.asarray(x[idx], dtype=np.float64)
    a -= a.mean(0, keepdims=True)
    lam = np.linalg.svd(a, compute_uv=False) ** 2
    return lam[0] / lam.sum(), lam.sum() ** 2 / np.sum(lam ** 2)


def robust_limits(mu: np.ndarray, sd: np.ndarray, pad: float = 0.15):
    """Y limits that frame the bulk of the coordinates, ignoring extreme ones."""
    lo = np.percentile(mu - sd, 1.0)
    hi = np.percentile(mu + sd, 99.0)
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    m = (hi - lo) * pad
    return lo - m, hi + m


def main(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    stats_cache = {}

    print(f"{'space':16s} {'d':>5s} {'||mu||':>9s} {'sqrt(sum var)':>14s} "
          f"{'offset share':>13s} {'PC1 (centered)':>15s} {'PR':>6s}")
    for ax, (label, folder) in zip(axes.ravel(), SPACES):
        mu, sd, n = dim_stats(folder)
        stats_cache[folder] = (mu, sd, n)
        d = len(mu)
        dims = np.arange(d)

        ax.fill_between(dims, mu - sd, mu + sd, color="#5B8DB8", alpha=0.45,
                        linewidth=0, label=r"mean $\pm$ s.d.")
        ax.plot(dims, mu, color="#1F4E79", linewidth=0.7, label="mean")
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":")

        lo, hi = robust_limits(mu, sd)
        ax.set_ylim(lo, hi)

        # Mark coordinates whose mean leaves the frame; name them in the stats box
        # rather than with arrows, which collide with the curve at this density.
        off = (mu < lo) | (mu > hi)
        off_note = ""
        if off.any():
            ax.plot(dims[off], np.clip(mu[off], lo, hi), "v", color="#C0392B",
                    markersize=5, clip_on=False, linestyle="none",
                    label=f"off scale ({off.sum()})")
            worst = dims[off][np.argsort(-np.abs(mu[off]))][:2]
            off_note = "\n" + "\n".join(
                f"off scale: dim {w} ({mu[w]:+.3g}, s.d. {sd[w]:.3g})" for w in worst)

        nrm_mu = np.linalg.norm(mu)
        tot_var = (sd ** 2).sum()
        share = nrm_mu ** 2 / (nrm_mu ** 2 + tot_var)
        pc1, pr = pca_spectrum(folder)

        ax.set_title(f"{label}  ($d$ = {d})", fontsize=12)
        ax.set_xlabel("embedding dimension", fontsize=10)
        ax.set_ylabel("coordinate value", fontsize=10)
        ax.text(0.98, 0.97,
                f"$\\|\\mu\\|$ = {nrm_mu:.1f},  $\\sqrt{{\\Sigma\\,\\mathrm{{var}}}}$ = {np.sqrt(tot_var):.1f}\n"
                f"offset share = {share:.2f}\n"
                f"centered PC1 = {pc1:.2f}" + off_note,
                transform=ax.transAxes, fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                          edgecolor="gray"))
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

        print(f"{label:16s} {d:5d} {nrm_mu:9.2f} {np.sqrt(tot_var):14.2f} "
              f"{share:13.3f} {pc1:15.3f} {pr:6.1f}")

    fig.suptitle("Per-coordinate mean and standard deviation of the pretrained "
                 "embedding spaces (AmpC 200k library)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, "emb_dim_mean_std.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {out}")

    # Companion figure: the coordinates sorted by spread, each space divided by its
    # own median so the six can share one axis.  A space whose curve stays flat has
    # no individual runaway coordinate.
    fig2, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for (label, folder), c in zip(SPACES, colors):
        mu, sd, _ = stats_cache[folder]
        d = len(mu)
        frac = np.arange(1, d + 1) / d
        a0.plot(frac, np.sort(sd)[::-1] / np.median(sd), color=c, label=label, lw=1.5)
        a1.plot(frac, np.sort(np.abs(mu))[::-1] / np.median(sd), color=c, label=label, lw=1.5)
    for a, t in ((a0, "s.d. / median s.d."), (a1, "|mean| / median s.d.")):
        a.set_yscale("log")
        a.set_xlabel("coordinates, ranked (fraction of $d$)", fontsize=10)
        a.set_ylabel(t, fontsize=10)
        a.axhline(1.0, color="gray", lw=0.6, ls=":")
        a.grid(alpha=0.25, which="both", lw=0.4)
    a0.set_title("Spread per coordinate", fontsize=12)
    a1.set_title("Offset per coordinate", fontsize=12)
    a0.legend(fontsize=8, loc="upper right")
    fig2.tight_layout()
    out2 = os.path.join(outdir, "emb_dim_spread_ranked.pdf")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out2}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=str, default="./figs")
    main(p.parse_args().outdir)
