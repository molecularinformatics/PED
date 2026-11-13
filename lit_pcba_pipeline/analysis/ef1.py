"""EF1% aggregation: turn per-target distance CSVs into per-reference and pooled tables."""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TARGETS = ["ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A",
           "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]

MODES = {"geodiff": ["2D", "3D", "concat"], "molformer": ["molformer"]}


def ef1_from_distances(distances: np.ndarray, labels: np.ndarray,
                       top_frac: float = 0.01) -> float:
    """Enrichment factor at ``top_frac``: hit rate in the top slice over hit rate overall."""
    valid = np.isfinite(distances)
    distances, labels = distances[valid], labels[valid]
    n = len(labels)
    if n == 0:
        return np.nan
    n_top = max(1, int(np.ceil(n * top_frac)))
    n_actives = int(labels.sum())
    if n_actives == 0:
        return np.nan
    top = labels[np.argsort(distances)[:n_top]]
    return float(top.mean() / labels.mean())


def ef1_granularity(n_lib: int, n_actives: int, top_frac: float = 0.01) -> Dict[str, float]:
    """How coarse the EF1% grid is for a target: one more active moves it by ``ef1_step``."""
    n_top = max(1, int(math.ceil(n_lib * top_frac)))
    hit_rate_all = n_actives / n_lib
    return {
        "n_lib": n_lib, "n_actives": n_actives, "n_top": n_top,
        "ef1_step": (1.0 / n_top) / hit_rate_all,
        "ef1_ceiling": (min(n_top, n_actives) / n_top) / hit_rate_all,
    }


def reference_columns(columns: Sequence[str], mode: str, distance_type: str):
    prefix = f"{mode}_{distance_type}_dist_"
    pooled = f"{mode}_{distance_type}_dist_min"
    return sorted(c for c in columns if c.startswith(prefix) and c != pooled), pooled


def _pdb_id(column: str, mode: str, distance_type: str) -> str:
    return column[len(f"{mode}_{distance_type}_dist_"):].replace("_ligand", "")


def collect(distance_csvs: Dict[str, str], backend: str, distance_type: str,
            top_frac: float = 0.01):
    """Per-reference and pooled EF1% for every target, plus the granularity table."""
    per_ref: List[Dict] = []
    pooled_rows: List[Dict] = []
    gran_rows: List[Dict] = []

    for target in TARGETS:
        path = distance_csvs.get(target)
        if path is None:
            continue
        head = pd.read_csv(path, nrows=0)
        wanted: List[str] = ["label"]
        for mode in MODES[backend]:
            cols, pooled = reference_columns(head.columns, mode, distance_type)
            wanted += cols + [pooled]
        df = pd.read_csv(path, usecols=sorted(set(wanted)))
        labels = df["label"].to_numpy(dtype=float)
        gran_rows.append({"target": target,
                          **ef1_granularity(len(labels), int(labels.sum()), top_frac)})

        for mode in MODES[backend]:
            cols, pooled = reference_columns(head.columns, mode, distance_type)
            for i, col in enumerate(cols):
                per_ref.append({
                    "target": target, "mode": mode, "reference_idx": i,
                    "pdb_id": _pdb_id(col, mode, distance_type),
                    "ef1": ef1_from_distances(df[col].to_numpy(float), labels, top_frac),
                })
            pooled_rows.append({
                "target": target, "mode": mode,
                "pooled_ef1": ef1_from_distances(df[pooled].to_numpy(float), labels, top_frac),
            })

    return pd.DataFrame(per_ref), pd.DataFrame(pooled_rows), pd.DataFrame(gran_rows)


def find_distance_csvs(out_root: str, backend: str) -> Dict[str, str]:
    """Map target -> distance CSV under ``out_root/<TARGET>/distance/``."""
    found: Dict[str, str] = {}
    pattern = re.compile(rf".*{backend}_distances\.csv$")
    for root, _, files in os.walk(out_root):
        if os.path.basename(root) != "distance":
            continue
        target = os.path.basename(os.path.dirname(root))
        for name in files:
            if pattern.match(name):
                found[target] = os.path.join(root, name)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root", required=True, help="Root holding <TARGET>/distance/")
    ap.add_argument("--backend", default="molformer", choices=["molformer", "geodiff"])
    ap.add_argument("--distance_type", nargs="+", default=["cosine", "euclidean"],
                    choices=["cosine", "euclidean"])
    ap.add_argument("--top_frac", type=float, default=0.01)
    ap.add_argument("--results_dir", default=None,
                    help="Where to write the EF1 tables (default: <out_root>/ef1)")
    args = ap.parse_args()

    csvs = find_distance_csvs(args.out_root, args.backend)
    if not csvs:
        raise SystemExit(f"no {args.backend} distance CSVs under {args.out_root}")
    results_dir = args.results_dir or os.path.join(args.out_root, "ef1")
    os.makedirs(results_dir, exist_ok=True)

    for distance_type in args.distance_type:
        per_ref, pooled, gran = collect(csvs, args.backend, distance_type, args.top_frac)
        stem = os.path.join(results_dir, f"lit_pcba_{args.backend}_{distance_type}")
        per_ref.to_csv(stem + "_per_reference.csv", index=False)
        pooled.to_csv(stem + "_pooled.csv", index=False)
        gran.to_csv(os.path.join(results_dir, "ef1_granularity.csv"), index=False)
        print(f"{args.backend} {distance_type}: {len(csvs)} targets, "
              f"{len(per_ref)} reference rows -> {stem}_*.csv")


if __name__ == "__main__":
    main()
