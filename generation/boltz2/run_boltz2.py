"""Boltz-2 binding-affinity prediction for the BTK generation benchmark.

Three steps, each skipped when its output already exists: fetch one MSA for the
target, write one input per molecule, run `boltz predict`, collect the affinities.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "btk_6O8I.yaml"

# Transform used throughout the paper; see README.
PIC50_INTERCEPT, PIC50_SLOPE = 8.184, -1.364


def target_sequence() -> str:
    line = next(l for l in TEMPLATE.read_text().splitlines() if "sequence:" in l)
    return line.split("sequence:")[1].strip()


BUNDLED_MSA = HERE / "btk_6O8I.a3m"


def fetch_msa(out: Path, refetch: bool) -> Path:
    """One MMseqs2 query for the target. Boltz queries per input file otherwise."""
    if not refetch and BUNDLED_MSA.exists():
        return BUNDLED_MSA
    if out.exists():
        return out
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    out.parent.mkdir(parents=True, exist_ok=True)
    a3m = run_mmseqs2(target_sequence(), prefix=str(out.parent / "mmseqs2"), use_env=True)
    out.write_text(a3m[0])
    return out


def write_inputs(df: pd.DataFrame, smiles_col: str, name_col: str, msa: Path,
                 work: Path) -> int:
    template = TEMPLATE.read_text().replace("MSA_PATH", str(msa.resolve()))
    work.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, smiles in zip(df[name_col], df[smiles_col]):
        path = work / f"{name}.yaml"
        if path.exists():
            continue
        path.write_text(template.replace("LIGAND_SMILES", str(smiles)))
        written += 1
    return written


def predict(work: Path, out: Path, boltz: str, seed: int, devices: int,
            samples: int) -> None:
    cmd = [boltz, "predict", str(work), "--out_dir", str(out), "--model", "boltz2",
           "--devices", str(devices), "--seed", str(seed),
           "--diffusion_samples_affinity", str(samples), "--output_format", "pdb"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def collect(out: Path, df: pd.DataFrame, name_col: str, smiles_col: str) -> pd.DataFrame:
    scores = {}
    for path in out.rglob("affinity_*.json"):
        scores[path.stem[len("affinity_"):]] = json.loads(path.read_text())
    rows = []
    for name, smiles in zip(df[name_col], df[smiles_col]):
        s = scores.get(str(name), {})
        value = s.get("affinity_pred_value")
        rows.append({
            name_col: name,
            smiles_col: smiles,
            "affinity_pred_value": value,
            "affinity_probability_binary": s.get("affinity_probability_binary"),
            "predicted_pIC50": None if value is None else PIC50_INTERCEPT + PIC50_SLOPE * value,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV of molecules to score")
    ap.add_argument("--out_dir", required=True, help="Working and output directory")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--name_col", default="Name", help="Unique, filename-safe ids")
    ap.add_argument("--boltz", default="boltz", help="Path to the boltz executable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--diffusion_samples_affinity", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Score the first N rows only")
    ap.add_argument("--refetch_msa", action="store_true",
                    help="Query the MSA server instead of using the bundled alignment")
    ap.add_argument("--collect_only", action="store_true")
    args = ap.parse_args()

    if shutil.which(args.boltz) is None and not Path(args.boltz).exists():
        sys.exit(f"boltz executable not found: {args.boltz}")

    df = pd.read_csv(args.input)
    for column in (args.smiles_col, args.name_col):
        if column not in df.columns:
            sys.exit(f"{args.input} has no column {column!r}; it has {list(df.columns)}")
    df = df.drop_duplicates(subset=args.name_col)
    if args.limit:
        df = df.head(args.limit)
    if df[args.name_col].astype(str).str.contains(r"[/\\\s]").any():
        sys.exit(f"{args.name_col} must be filename-safe")

    out = Path(args.out_dir)
    work, preds = out / "inputs", out / "predictions"

    if not args.collect_only:
        msa = fetch_msa(out / "msa" / "btk_6O8I.a3m", args.refetch_msa)
        print(f"MSA: {msa}  ({msa.read_text().count('>')} sequences)")
        print(f"wrote {write_inputs(df, args.smiles_col, args.name_col, msa, work)} inputs")
        predict(work, preds, args.boltz, args.seed, args.devices,
                args.diffusion_samples_affinity)

    scored = collect(preds, df, args.name_col, args.smiles_col)
    path = out / "affinity.csv"
    scored.to_csv(path, index=False)
    missing = int(scored.affinity_pred_value.isna().sum())
    print(f"wrote {path}  ({len(scored)} molecules, {missing} without a prediction)")


if __name__ == "__main__":
    main()
