"""Correlation structure among the reference similarity measures on the AmpC library.

Shows that ROCS combination is the sum of the shape and color arms, how strongly the
arms correlate with each other, and how far each is predictable from a 2D fingerprint.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import ampc200k as A


def combination_identity(shape: np.ndarray, color: np.ndarray,
                         combo: np.ndarray) -> dict:
    """Least-squares fit of combination against the two arms."""
    design = np.c_[shape, color, np.ones(len(combo))]
    beta, residual, *_ = np.linalg.lstsq(design, combo, rcond=None)
    ss_tot = float(np.sum((combo - combo.mean()) ** 2))
    r2 = 1.0 - float(residual[0]) / ss_tot if len(residual) else np.nan
    return {"coef_shape": float(beta[0]), "coef_color": float(beta[1]),
            "intercept": float(beta[2]), "r2": r2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()

    df = A.aligned()
    cols = {"shape": "ShapeTanimoto", "color": "ColorTanimoto",
            "combination": "3d_similarity_ROCS", "ECFP4": "ecfp_tanimoto"}
    M = pd.DataFrame({k: df[v].to_numpy(float) for k, v in cols.items()})

    fit = combination_identity(M["shape"].to_numpy(), M["color"].to_numpy(),
                               M["combination"].to_numpy())
    print(f"n = {len(M)}")
    print(f"combination = {fit['coef_shape']:.3f}*shape + {fit['coef_color']:.3f}*color "
          f"+ {fit['intercept']:.3f},  R^2 = {fit['r2']:.4f}\n")
    corr = M.corr()
    print(corr.to_string(float_format=lambda v: f"{v:+.3f}"))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "reference_measure_correlations.csv")
    corr.to_csv(path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
