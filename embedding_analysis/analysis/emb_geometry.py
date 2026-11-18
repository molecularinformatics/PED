"""Geometry diagnostics of pretrained embedding spaces, and how they order against
the downstream correlation.

Variance spectrum, anisotropy before and after mean-centering, TwoNN intrinsic
dimension, distance concentration, and neighbourhood agreement with ECFP4, all
measured on one shared molecule set so the spaces are comparable. The spaces are
few and not independent, so the rank correlations printed at the end describe an
ordering rather than establish a criterion.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

import ampc200k as A


# --------------------------------------------------------------------------- #
# Individual diagnostics                                                      #
# --------------------------------------------------------------------------- #

def variance_spectrum(x: np.ndarray) -> Dict[str, float]:
    xc = x - x.mean(axis=0, keepdims=True)
    n, d = xc.shape
    # Eigenvalues of the covariance via the SVD of the centred matrix.
    s = np.linalg.svd(xc, compute_uv=False)
    lam = (s ** 2) / max(n - 1, 1)
    tot = lam.sum()
    if tot <= 0:
        return dict(pc1_var_frac=np.nan, top10_var_frac=np.nan,
                    participation_ratio=np.nan, pr_over_d=np.nan)
    pr = float(tot ** 2 / np.sum(lam ** 2))
    return dict(
        pc1_var_frac=float(lam[0] / tot),
        top10_var_frac=float(lam[:10].sum() / tot),
        participation_ratio=pr,
        pr_over_d=pr / d,
    )


def anisotropy(x: np.ndarray, n_pairs: int = 200_000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    def mean_cos(v: np.ndarray) -> float:
        vn = A.l2_normalize(v)
        return float(np.mean(np.sum(vn[i] * vn[j], axis=1)))

    mu = x.mean(axis=0, keepdims=True)
    return dict(
        mean_cos_raw=mean_cos(x),
        mean_cos_centered=mean_cos(x - mu),
        mu_norm_ratio=float(np.linalg.norm(mu) / np.mean(np.linalg.norm(x, axis=1))),
    )


def intrinsic_dimension_twonn(x: np.ndarray, n_sample: int = 20_000,
                              discard_frac: float = 0.1, seed: int = 0) -> float:
    """Intrinsic dimension by TwoNN (Facco et al. 2017), from the ratio of each
    point's second to first nearest-neighbour distance."""
    rng = np.random.default_rng(seed)
    if len(x) > n_sample:
        x = x[rng.choice(len(x), n_sample, replace=False)]
    nn = NearestNeighbors(n_neighbors=3).fit(x)
    dist, _ = nn.kneighbors(x)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    mu = np.sort(mu)
    n = len(mu)
    keep = int(np.floor(n * (1 - discard_frac)))
    mu = mu[:keep]
    f = np.arange(1, keep + 1) / n
    xs = np.log(mu)
    ys = -np.log1p(-f)
    denom = float(np.sum(xs ** 2))
    return float(np.sum(xs * ys) / denom) if denom > 0 else np.nan


def distance_concentration(x: np.ndarray, ref: np.ndarray,
                           n_queries: int = 50, seed: int = 0) -> Dict[str, float]:
    def stats_for(q: np.ndarray) -> Dict[str, float]:
        d = np.linalg.norm(x - q[None, :], axis=1)
        m = float(d.mean())
        if m <= 0:
            return dict(cv=np.nan, rel_contrast=np.nan)
        return dict(cv=float(d.std() / m),
                    rel_contrast=float((d.max() - d.min()) / m))

    ref_s = stats_for(ref)
    rng = np.random.default_rng(seed)
    qs = x[rng.choice(len(x), n_queries, replace=False)]
    rand = [stats_for(q) for q in qs]
    return dict(
        cv_dist_ref=ref_s["cv"],
        rel_contrast_ref=ref_s["rel_contrast"],
        cv_dist_rand=float(np.mean([r["cv"] for r in rand])),
        rel_contrast_rand=float(np.mean([r["rel_contrast"] for r in rand])),
    )


def knn_overlap_with_ecfp(x: np.ndarray, ecfp: np.ndarray, k: int = 100,
                          n_query: int = 2_000, seed: int = 0) -> float:
    """Mean |N_k(embedding) ∩ N_k(ECFP4)| / k over random query molecules.

    ECFP4 neighbours use the Tanimoto metric on the binary fingerprint; embedding
    neighbours use cosine, which is how PED ranks.
    """
    rng = np.random.default_rng(seed)
    q = rng.choice(len(x), n_query, replace=False)

    xn = A.l2_normalize(x)
    sim_emb = xn[q] @ xn.T
    np.put_along_axis(sim_emb, q[:, None], -np.inf, axis=1)
    top_emb = np.argpartition(-sim_emb, k, axis=1)[:, :k]

    fp = (ecfp > 0).astype(np.float32)
    inter = fp[q] @ fp.T
    pop = fp.sum(axis=1)
    union = pop[q][:, None] + pop[None, :] - inter
    tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.put_along_axis(tan, q[:, None], -np.inf, axis=1)
    top_fp = np.argpartition(-tan, k, axis=1)[:, :k]

    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(top_emb, top_fp)]))


# --------------------------------------------------------------------------- #
# Downstream performance, for comparison                                      #
# --------------------------------------------------------------------------- #

def downstream_correlations(space: str, rows: np.ndarray) -> Dict[str, float]:
    """|Pearson r| between PED and each traditional similarity, cosine and Euclidean."""
    df = A.aligned().iloc[rows]
    x = A.load_library(space, rows)
    ref = A.load_reference(space)

    xn, rn = A.l2_normalize(x), ref / max(np.linalg.norm(ref), 1e-12)
    d_cos = 1.0 - xn @ rn
    d_euc = np.linalg.norm(x - ref[None, :], axis=1)

    out: Dict[str, float] = {}
    for label, col in A.SIMILARITY_COLS.items():
        y = df[col].to_numpy(float)
        ok = np.isfinite(y)
        for tag, d in (("cos", d_cos), ("euc", d_euc)):
            out[f"r_{tag}__{label}"] = float(stats.pearsonr(d[ok], y[ok])[0])
    out["best_abs_r_cos"] = max(abs(v) for k, v in out.items() if k.startswith("r_cos__"))
    out["best_abs_r_euc"] = max(abs(v) for k, v in out.items() if k.startswith("r_euc__"))
    return out


# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_sample", type=int, default=30_000,
                    help="molecules used for the spectrum/anisotropy/concentration")
    ap.add_argument("--n_id", type=int, default=20_000, help="molecules for TwoNN")
    ap.add_argument("--knn_k", type=int, default=100)
    ap.add_argument("--knn_query", type=int, default=2_000)
    ap.add_argument("--skip_knn", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()

    rows = A.subsample(args.n_sample, seed=args.seed)
    ecfp = None if args.skip_knn else A.load_ecfp(rows)
    print(f"{len(rows)} molecules, {len(A.SPACES)} embedding spaces\n")

    records: List[Dict] = []
    for space in A.SPACES:
        x = A.load_library(space, rows)
        ref = A.load_reference(space)
        rec: Dict = {"space": space, "reported_in_paper": space in A.REPORTED,
                     "dim": x.shape[1]}
        rec.update(variance_spectrum(x))
        rec.update(anisotropy(x, seed=args.seed))
        rec["id_twonn"] = intrinsic_dimension_twonn(x, args.n_id, seed=args.seed)
        rec["id_over_d"] = rec["id_twonn"] / x.shape[1]
        rec.update(distance_concentration(x, ref, seed=args.seed))
        if ecfp is not None:
            rec["knn_overlap_ecfp"] = knn_overlap_with_ecfp(
                x, ecfp, k=args.knn_k, n_query=args.knn_query, seed=args.seed)
        rec.update(downstream_correlations(space, rows))
        records.append(rec)
        print(f"  {space:20s} d={x.shape[1]:5d} PR/d={rec['pr_over_d']:.3f} "
              f"mean_cos={rec['mean_cos_raw']:+.3f} ID={rec['id_twonn']:.1f} "
              f"cv_ref={rec['cv_dist_ref']:.3f} best|r|cos={rec['best_abs_r_cos']:.3f}",
              flush=True)

    df = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "emb_geometry.csv")
    df.to_csv(path, index=False)

    diags = ["pc1_var_frac", "top10_var_frac", "pr_over_d", "participation_ratio",
             "mean_cos_raw", "mean_cos_centered", "mu_norm_ratio",
             "id_twonn", "id_over_d", "cv_dist_ref", "cv_dist_rand",
             "rel_contrast_ref"]
    if "knn_overlap_ecfp" in df.columns:
        diags.append("knn_overlap_ecfp")

    print("\n=== does the diagnostic order the spaces like the downstream result? ===")
    print("Spearman across the 12 spaces; descriptive only (5 are Uni-Mol2 variants)")
    rank_rows = []
    for dgn in diags:
        row = {"diagnostic": dgn}
        for tgt in ("best_abs_r_cos", "best_abs_r_euc"):
            rho, p = stats.spearmanr(df[dgn], df[tgt])
            row[f"rho__{tgt}"] = rho
            row[f"p__{tgt}"] = p
        rank_rows.append(row)
    rank = pd.DataFrame(rank_rows).sort_values("rho__best_abs_r_cos",
                                               key=lambda s: -s.abs())
    print(rank.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rank.to_csv(os.path.join(args.out_dir, "emb_geometry_rank_corr.csv"), index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
