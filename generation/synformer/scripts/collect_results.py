"""
Turn the raw output of a run into the two CSVs reported in the paper.

A run writes ``molecules.csv`` under its log directory: one row per generated
molecule, in generation order, duplicates kept. This script derives from it:

  <name>_generated.csv   byte-identical copy of molecules.csv -- every
                         generation, in order (step, smiles, score, synthesis)
  <name>_explored.csv    one row per unique molecule, sorted by score;
                         `step` is the step the molecule was FIRST generated

Use the generated file for anything per-step (the policy's output distribution)
and the explored file for "what the oracle budget was spent on". Note that the
explored file credits a molecule to its first occurrence, so the best molecules
carry early step numbers even when the policy keeps re-emitting them later.

Usage:
    python scripts/collect_results.py logs/ecfp4 --name synformer_rl_btk_ecfp4
    python scripts/collect_results.py logs/ecfp4/<task_dir>/molecules.csv -n run1
"""

import argparse
import pathlib
import shutil

import pandas as pd


def find_molecules_csv(path: pathlib.Path) -> pathlib.Path:
    """Accept either molecules.csv itself or any directory above it."""
    if path.is_file():
        return path
    hits = sorted(path.rglob("molecules.csv"))
    if not hits:
        raise SystemExit(f"No molecules.csv found under {path}")
    if len(hits) > 1:
        raise SystemExit(
            "Several runs found; pass one explicitly:\n  "
            + "\n  ".join(str(h) for h in hits)
        )
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=pathlib.Path,
                    help="a run's log directory, or its molecules.csv")
    ap.add_argument("-n", "--name", required=True,
                    help="output basename, e.g. synformer_rl_btk_ecfp4")
    ap.add_argument("-o", "--out-dir", type=pathlib.Path, default=pathlib.Path("results"))
    args = ap.parse_args()

    src = find_molecules_csv(args.run)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    generated = args.out_dir / f"{args.name}_generated.csv"
    shutil.copyfile(src, generated)

    df = pd.read_csv(src)
    df["smiles"] = df["smiles"].fillna("")
    n_raw, n_empty = len(df), int((df["smiles"].str.len() == 0).sum())

    uniq = (df[df["smiles"].str.len() > 0]
            .drop_duplicates(subset="smiles", keep="first")
            .sort_values("score", ascending=False)
            .reset_index(drop=True)[["smiles", "score", "step", "synthesis"]])
    explored = args.out_dir / f"{args.name}_explored.csv"
    uniq.to_csv(explored, index=False)

    top = uniq["score"].to_numpy()
    print(f"source              {src}")
    print(f"steps               {int(df['step'].max())}")
    print(f"generated rows      {n_raw}  ({n_empty} with an empty SMILES: stacks that "
          f"failed to build, scored 0.0)")
    print(f"unique molecules    {len(uniq)}  (duplicate rate "
          f"{(n_raw - len(uniq)) / n_raw * 100:.1f}%)")
    print(f"top1 / top10 / top100   {top[0]:.4f} / {top[:10].mean():.4f} / {top[:100].mean():.4f}")
    print(f"\nwrote  {generated}\nwrote  {explored}")


if __name__ == "__main__":
    main()
