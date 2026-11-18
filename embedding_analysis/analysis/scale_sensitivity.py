"""Effect of per-dimension rescaling on the distance-to-similarity correlations.

Compares raw embeddings against mean-centering, z-scoring and PCA whitening, with
all statistics taken from the library alone. Rescaling makes a distance depend on
the surrounding database, which is why the pipeline does not apply it; this
measures what that choice costs.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

import ampc200k as A

EPS = 1e-8


def fit_transform(x: np.ndarray, ref: np.ndarray, mode: str):
    """Return (library, reference) under ``mode``; statistics from the library only."""
    if mode == "raw":
        return x, ref

    mu = x.mean(axis=0, keepdims=True)
    if mode == "center":
        return x - mu, (ref[None, :] - mu).ravel()

    if mode == "zscore":
        sd = x.std(axis=0, keepdims=True)
        sd = np.clip(sd, EPS, None)
        return (x - mu) / sd, ((ref[None, :] - mu) / sd).ravel()

    if mode == "whiten":
        xc = x - mu
        # SVD of the centred library gives the principal directions and their scales.
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        scale = np.clip(s / np.sqrt(max(len(x) - 1, 1)), EPS, None)
        w = vt.T / scale[None, :]
        return xc @ w, ((ref[None, :] - mu) @ w).ravel()

    raise ValueError(f"unknown mode {mode!r}")


def correlations(x: np.ndarray, ref: np.ndarray, df: pd.DataFrame) -> Dict[str, float]:
    xn = A.l2_normalize(x)
    rn = ref / max(np.linalg.norm(ref), EPS)
    d = {
        "cosine": 1.0 - xn @ rn,
        "euclidean": np.linalg.norm(x - ref[None, :], axis=1),
    }
    out: Dict[str, float] = {}
    for sim_label, col in A.SIMILARITY_COLS.items():
        y = df[col].to_numpy(float)
        ok = np.isfinite(y)
        for dist_label, dv in d.items():
            out[f"{dist_label}|{sim_label}"] = float(stats.pearsonr(dv[ok], y[ok])[0])
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_sample", type=int, default=0,
                    help="0 uses all 199,770 molecules; a smaller number subsamples")
    ap.add_argument("--modes", nargs="+", default=["raw", "zscore", "whiten"])
    ap.add_argument("--spaces", nargs="*", default=None,
                    help="default: every space in ampc200k.SPACES")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()

    rows = (np.arange(len(A.aligned())) if args.n_sample <= 0
            else A.subsample(args.n_sample, seed=args.seed))
    df = A.aligned().iloc[rows]
    spaces = args.spaces or list(A.SPACES)
    print(f"{len(rows)} molecules, {len(spaces)} spaces, modes={args.modes}\n")

    records: List[Dict] = []
    for space in spaces:
        x0 = A.load_library(space, rows)
        r0 = A.load_reference(space)
        for mode in args.modes:
            x, r = fit_transform(x0, r0, mode)
            rec = {"space": space, "mode": mode, "dim": x.shape[1]}
            rec.update(correlations(x, r, df))
            records.append(rec)
            key = "euclidean|ROCS combination"
            print(f"  {space:20s} {mode:7s} r(euclid, ROCS combo) = {rec[key]:+.3f}",
                  flush=True)
        del x0

    out = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "scale_sensitivity.csv")
    out.to_csv(path, index=False)

    # Side-by-side view of the quantity Figure 2 reports.
    print("\n=== Pearson r, Euclidean PED vs each traditional similarity ===")
    for sim_label in A.SIMILARITY_COLS:
        col = f"euclidean|{sim_label}"
        piv = out.pivot(index="space", columns="mode", values=col)
        piv = piv.reindex(columns=[m for m in args.modes if m in piv.columns])
        piv["d(zscore-raw)"] = piv.get("zscore", np.nan) - piv["raw"]
        print(f"\n-- {sim_label}")
        print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\n=== Pearson r, cosine PED (what the main text reports) ===")
    for sim_label in A.SIMILARITY_COLS:
        col = f"cosine|{sim_label}"
        piv = out.pivot(index="space", columns="mode", values=col)
        piv = piv.reindex(columns=[m for m in args.modes if m in piv.columns])
        piv["d(zscore-raw)"] = piv.get("zscore", np.nan) - piv["raw"]
        print(f"\n-- {sim_label}")
        print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
