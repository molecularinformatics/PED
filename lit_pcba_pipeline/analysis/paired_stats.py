"""Distribution-free paired statistics for comparing two methods across targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_mean_ci(d: np.ndarray, n_boot: int = 20_000, seed: int = 0,
                      alpha: float = 0.05):
    """Percentile bootstrap interval for the mean difference, resampling targets."""
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def interval_variants(d: np.ndarray, n_boot: int = 20_000, seed: int = 0,
                      alpha: float = 0.05) -> Dict[str, tuple]:
    """Percentile, basic, BCa and Student intervals, to show the choice does not matter."""
    rng = np.random.default_rng(seed)
    theta = float(d.mean())
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    pct = tuple(np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

    z0 = stats.norm.ppf((boot < theta).mean())
    jack = np.array([np.delete(d, i).mean() for i in range(len(d))])
    centered = jack.mean() - jack
    denom = 6 * (np.sum(centered ** 2) ** 1.5)
    a = float(np.sum(centered ** 3) / denom) if denom else 0.0
    bca = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = stats.norm.ppf(q)
        bca.append(float(np.percentile(boot, 100 * stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))))

    return {
        "percentile": (float(pct[0]), float(pct[1])),
        "basic": (2 * theta - float(pct[1]), 2 * theta - float(pct[0])),
        "bca": tuple(bca),
        "student": tuple(float(v) for v in stats.t.interval(1 - alpha, len(d) - 1,
                                                            loc=theta, scale=stats.sem(d))),
    }


@dataclass
class PairedResult:
    method_a: str
    method_b: str
    n_targets: int
    mean_diff: float
    ci_lo: float
    ci_hi: float
    wilcoxon_stat: float
    wilcoxon_p: float
    wilcoxon_method: str
    n_higher: int
    n_lower: int
    n_tied: int
    p_holm: Optional[float] = None

    def as_dict(self) -> Dict:
        return asdict(self)


def _wilcoxon(d: np.ndarray):
    zero_method = "pratt" if np.any(d == 0) else "wilcox"
    for method in ("exact", "approx"):
        try:
            res = stats.wilcoxon(d, zero_method=zero_method, method=method,
                                 alternative="two-sided")
            return float(res.statistic), float(res.pvalue), f"{method}/{zero_method}"
        except (ValueError, TypeError):
            continue
    return np.nan, np.nan, "failed"


def compare_paired(a: Sequence[float], b: Sequence[float], name_a: str, name_b: str,
                   n_boot: int = 20_000, seed: int = 0) -> PairedResult:
    """Compare two methods over paired per-target values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    stat, p, how = _wilcoxon(d)
    lo, hi = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed)
    return PairedResult(name_a, name_b, len(d), float(d.mean()), lo, hi,
                        stat, p, how,
                        int(np.sum(d > 0)), int(np.sum(d < 0)), int(np.sum(d == 0)))


def holm(results: List[PairedResult]) -> List[PairedResult]:
    """Holm step-down correction over one family of comparisons, in place."""
    order = np.argsort([r.wilcoxon_p for r in results])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(results) - rank) * results[i].wilcoxon_p))
        results[i].p_holm = float(running)
    return results


def leave_one_out(a: Sequence[float], b: Sequence[float],
                  targets: Sequence[str]) -> pd.DataFrame:
    """Recompute the mean difference with each target removed in turn."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    full = float(np.nanmean(d))
    rows = []
    for i, t in enumerate(targets):
        kept = np.delete(d, i)
        loo = float(np.nanmean(kept))
        rows.append({"dropped_target": t, "dropped_diff": float(d[i]),
                     "n_remaining": len(kept), "mean_diff_loo": loo,
                     "delta_vs_full": loo - full,
                     "sign_flips": bool(np.sign(loo) != np.sign(full))})
    return pd.DataFrame(rows)
