"""Why MoleculeSDE 2D has a large Pearson/Spearman gap on the AmpC 200k library.

The 2D GIN branch of MoleculeSDE produces embeddings whose norm is tightly
distributed around 5.2 for almost every molecule, and one to two orders of
magnitude larger for a small subset.  Those molecules dominate the Pearson
correlation with ROCS similarity while leaving the ranking untouched, which is
the whole of the gap between r = -0.08 and rho = -0.64 for Euclidean PED.

This script identifies the subset.  It is a single substructure, and the trigger
is narrower than "contains fluorine": it is a trifluoromethyl group bonded to an
sp3 carbon or to a heteroatom.  The same group bonded to an aromatic carbon
behaves normally, so the cause is local to the attachment rather than to the
fluorine atoms themselves.

Run after the AmpC 200k embeddings are in place; see ``ampc200k.py`` for layout.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import ampc200k as A

RDLogger.DisableLog("rdApp.*")

CF3 = Chem.MolFromSmarts("[CX4](F)(F)F")
CF3_AROMATIC = Chem.MolFromSmarts("c[CX4](F)(F)F")
CF3_SP3 = Chem.MolFromSmarts("[CX4;!c][CX4](F)(F)F")
CF3_HETERO = Chem.MolFromSmarts("[!#6][CX4](F)(F)F")

# Molecules above this embedding norm are the separated cluster; the bulk sits
# near 5.2 and the gap between the two populations spans a factor of two.
NORM_CUT = 10.0


def main(space: str, outdir: str) -> None:
    df = A.aligned()
    x = A.load_library(space)
    norm = np.linalg.norm(x, axis=1)

    # The GIN is fed the canonical SMILES, so the substructure search uses those.
    mols = [Chem.MolFromSmiles(s) for s in df["canonical_smiles"]]
    has_cf3 = np.array([m.HasSubstructMatch(CF3) if m else False for m in mols])
    aromatic = np.array([m.HasSubstructMatch(CF3_AROMATIC) if m else False for m in mols])
    sp3 = np.array([m.HasSubstructMatch(CF3_SP3) if m else False for m in mols])
    hetero = np.array([m.HasSubstructMatch(CF3_HETERO) if m else False for m in mols])
    n_f = np.array([sum(1 for a in m.GetAtoms() if a.GetSymbol() == "F") if m else 0
                    for m in mols])

    big = norm > NORM_CUT
    print(f"{space}: n = {len(df)}, norm > {NORM_CUT} for {big.sum()} molecules "
          f"({100 * big.mean():.2f}%)")
    print(f"  bulk norm   : median {np.median(norm[~big]):.2f}  "
          f"p1 {np.percentile(norm[~big], 1):.2f}  p99 {np.percentile(norm[~big], 99):.2f}")
    print(f"  cluster norm: median {np.median(norm[big]):.2f}  "
          f"p1 {np.percentile(norm[big], 1):.2f}  max {norm[big].max():.2f}")
    print(f"  fraction of the cluster that contains CF3: {100 * has_cf3[big].mean():.1f}%")
    print(f"  fraction of the bulk that contains CF3   : {100 * has_cf3[~big].mean():.1f}%")

    print("\n  P(norm > cut) by fluorine count")
    for k in range(0, int(n_f.max()) + 1):
        m = n_f == k
        if m.sum() < 50:
            continue
        print(f"    {k} F : n = {m.sum():6d}   {100 * big[m].mean():6.2f}%")

    print("\n  P(norm > cut) by what the CF3 carbon is bonded to")
    # Categories overlap only for the 11 molecules carrying both kinds of CF3.
    groups = [
        ("sp3 carbon", sp3),
        ("aromatic carbon", aromatic),
        ("aromatic only", aromatic & ~sp3 & ~hetero),
        ("heteroatom", hetero),
        ("no CF3 group", ~has_cf3),
    ]
    for label, m in groups:
        if m.sum() < 5:
            continue
        print(f"    {label:16s} n = {m.sum():6d}   median norm {np.median(norm[m]):8.2f}"
              f"   {100 * big[m].mean():6.2f}%")

    # The effect on the correlation: drop the cluster and recompute.  ``aligned()``
    # keeps only the traditional similarity columns, so the PED columns are pulled
    # from the same table and put in its row order.
    from scipy import stats
    raw = pd.read_csv(os.path.join(A.ROOT, "similarity_distance_200k.csv"))
    raw = raw.drop_duplicates(subset="smiles").set_index("smiles")
    for col, name in [("sde_2D_euclidean", "Euclidean"), ("sde_2D_cosine", "cosine")]:
        if col not in raw.columns:
            continue
        y = df["3d_similarity_ROCS"].to_numpy()
        d = raw[col].reindex(df["raw_smiles"]).to_numpy()
        r_all = stats.pearsonr(d, y)[0]
        rho_all = stats.spearmanr(d, y)[0]
        r_cut = stats.pearsonr(d[~big], y[~big])[0]
        rho_cut = stats.spearmanr(d[~big], y[~big])[0]
        print(f"\n  {name} PED vs ROCS combination")
        print(f"    all molecules      Pearson {r_all:+.3f}  Spearman {rho_all:+.3f}")
        print(f"    cluster removed    Pearson {r_cut:+.3f}  Spearman {rho_cut:+.3f}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, "sde2d_cf3_outliers.csv")
        pd.DataFrame({
            "canonical_smiles": df["canonical_smiles"],
            "norm": norm,
            "has_cf3": has_cf3,
            "cf3_on_aromatic": aromatic,
            "cf3_on_sp3": sp3,
            "cf3_on_heteroatom": hetero,
            "n_fluorine": n_f,
        }).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--space", type=str, default="MoleculeSDE 2D")
    p.add_argument("--outdir", type=str, default="")
    a = p.parse_args()
    main(a.space, a.outdir)
