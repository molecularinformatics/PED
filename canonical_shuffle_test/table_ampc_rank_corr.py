import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr


# =========================
# paths
# =========================
all_distance_csv = "/mnt/depts/dept01/misd/molml/embeddings/200k_GeoDiff_emb/similarity_distance_200k_with_all_distances.csv"
geodiff_canonical_csv = "/mnt/depts/dept01/misd/molml/embeddings/200k_emb/distance/smiles_reference_geodiff_distances.csv"
molformer_canonical_csv = "/mnt/depts/dept01/misd/molml/embeddings/200k_emb/distance/smiles_reference_molformer_distances.csv"

out_merged_csv = "/mnt/depts/dept01/misd/molml/embeddings/200k_emb/raw_canonical_distances.csv"


# =========================
# canonicalization
# =========================
def canonicalize_smiles(smiles: str) -> Optional[str]:
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


# =========================
# helpers
# =========================
def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    corr, _ = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return corr


def format_corr(val: float) -> str:
    if pd.isna(val):
        return "NA"
    return f"{val:.4f}"


def build_latex_table(corr_df: pd.DataFrame) -> str:
    lines = []
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Method & Cosine distance $\\rho$ & Euclidean distance $\\rho$ \\\\")
    lines.append("\\midrule")
    for _, row in corr_df.iterrows():
        lines.append(
            f"{row['Method']} & {row['Cosine distance rank correlation']} & {row['Euclidean distance rank correlation']} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# =========================
# main
# =========================
def main():
    # -------- read --------
    df_all = pd.read_csv(all_distance_csv)
    df_geo_can = pd.read_csv(geodiff_canonical_csv)
    df_mol_can = pd.read_csv(molformer_canonical_csv)

    # -------- check core columns --------
    if "smiles" not in df_all.columns:
        raise ValueError("all_distance_csv must contain column 'smiles'")
    if "smiles" not in df_geo_can.columns:
        raise ValueError("geodiff_canonical_csv must contain column 'smiles'")
    if "smiles" not in df_mol_can.columns:
        raise ValueError("molformer_canonical_csv must contain column 'smiles'")

    # -------- canonicalize all-distance raw smiles --------
    df_all = df_all.copy()
    df_all["raw_smiles"] = df_all["smiles"].astype(str)
    df_all["canonical_smiles"] = df_all["raw_smiles"].apply(canonicalize_smiles)
    df_all = df_all[df_all["canonical_smiles"].notna()].reset_index(drop=True)

    # -------- define distance columns --------
    # raw-based columns from all-distance csv
    raw_col_map = {
        "MoLFormer": {
            "cosine": "cosine",
            "euclidean": "euclidean",
        },
        "GeoDiff 2D": {
            "cosine": "rocs_2d_cosine",
            "euclidean": "rocs_2d_euclidean",
        },
        "GeoDiff 3D": {
            "cosine": "rocs_3d_cosine",
            "euclidean": "rocs_3d_euclidean",
        },
        "GeoDiff Concat": {
            "cosine": "rocs_concat_norm_cosine",
            "euclidean": "rocs_concat_norm_euclidean",
        },
    }

    # canonical-based columns
    canonical_col_map = {
        "MoLFormer": {
            "cosine": "molformer_cosine",
            "euclidean": "molformer_euclidean",
        },
        "GeoDiff 2D": {
            "cosine": "geodiff_2D_cosine",
            "euclidean": "geodiff_2D_euclidean",
        },
        "GeoDiff 3D": {
            "cosine": "geodiff_3D_cosine",
            "euclidean": "geodiff_3D_euclidean",
        },
        "GeoDiff Concat": {
            "cosine": "geodiff_concat_norm_cosine",
            "euclidean": "geodiff_concat_norm_euclidean",
        },
    }

    # -------- verify columns exist --------
    needed_all_cols = ["raw_smiles", "canonical_smiles"]
    for method in raw_col_map:
        needed_all_cols.extend(list(raw_col_map[method].values()))
    missing_all = [c for c in needed_all_cols if c not in df_all.columns]
    if missing_all:
        raise ValueError(f"Missing columns in all_distance_csv: {missing_all}")

    needed_geo_cols = ["smiles"]
    for method in ["GeoDiff 2D", "GeoDiff 3D", "GeoDiff Concat"]:
        needed_geo_cols.extend(list(canonical_col_map[method].values()))
    missing_geo = [c for c in needed_geo_cols if c not in df_geo_can.columns]
    if missing_geo:
        raise ValueError(f"Missing columns in geodiff_canonical_csv: {missing_geo}")

    needed_mol_cols = ["smiles"]
    needed_mol_cols.extend(list(canonical_col_map["MoLFormer"].values()))
    missing_mol = [c for c in needed_mol_cols if c not in df_mol_can.columns]
    if missing_mol:
        raise ValueError(f"Missing columns in molformer_canonical_csv: {missing_mol}")

    # -------- deduplicate all-distance csv by canonical smiles --------
    agg_dict = {"raw_smiles": "first"}
    for method in raw_col_map:
        for dist_type, col in raw_col_map[method].items():
            agg_dict[col] = "first"

    df_all_dedup = (
        df_all.groupby("canonical_smiles", as_index=False)
        .agg(agg_dict)
        .reset_index(drop=True)
    )

    # rename raw-based columns
    rename_raw = {"raw_smiles": "raw_smiles"}
    for method in raw_col_map:
        for dist_type, col in raw_col_map[method].items():
            rename_raw[col] = f"{method.lower().replace(' ', '_')}_raw_{dist_type}"
    df_all_dedup = df_all_dedup.rename(columns=rename_raw)

    # -------- prepare canonical distance tables --------
    df_geo_can = df_geo_can.copy().rename(columns={"smiles": "canonical_smiles"})
    df_mol_can = df_mol_can.copy().rename(columns={"smiles": "canonical_smiles"})

    rename_geo = {}
    for method in ["GeoDiff 2D", "GeoDiff 3D", "GeoDiff Concat"]:
        for dist_type, col in canonical_col_map[method].items():
            rename_geo[col] = f"{method.lower().replace(' ', '_')}_canonical_{dist_type}"
    df_geo_can = df_geo_can.rename(columns=rename_geo)

    rename_mol = {}
    for dist_type, col in canonical_col_map["MoLFormer"].items():
        rename_mol[col] = f"molformer_canonical_{dist_type}"
    df_mol_can = df_mol_can.rename(columns=rename_mol)

    # -------- merge all --------
    merged = df_all_dedup.merge(df_geo_can, on="canonical_smiles", how="inner")
    merged = merged.merge(df_mol_can, on="canonical_smiles", how="inner")

    # reorder columns
    ordered_cols = [
        "raw_smiles",
        "canonical_smiles",
        "molformer_raw_cosine",
        "molformer_raw_euclidean",
        "molformer_canonical_cosine",
        "molformer_canonical_euclidean",
        "geodiff_2d_raw_cosine",
        "geodiff_2d_raw_euclidean",
        "geodiff_2d_canonical_cosine",
        "geodiff_2d_canonical_euclidean",
        "geodiff_3d_raw_cosine",
        "geodiff_3d_raw_euclidean",
        "geodiff_3d_canonical_cosine",
        "geodiff_3d_canonical_euclidean",
        "geodiff_concat_raw_cosine",
        "geodiff_concat_raw_euclidean",
        "geodiff_concat_canonical_cosine",
        "geodiff_concat_canonical_euclidean",
    ]

    col_fix = {}
    for c in merged.columns:
        c_new = c.replace("geodiff_concat_raw", "geodiff_concat_raw")
        c_new = c_new.replace("geodiff_concat_canonical", "geodiff_concat_canonical")
        col_fix[c] = c_new
    merged = merged.rename(columns=col_fix)

    # 如果 concat 名字来自 lower+replace，手动映射一下
    # 实际上 lower().replace(' ', '_') for "GeoDiff Concat" -> "geodiff_concat"
    # 所以上面的 ordered_cols 已经匹配

    ordered_cols_existing = [c for c in ordered_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in ordered_cols_existing]
    merged = merged[ordered_cols_existing + other_cols]

    merged.to_csv(out_merged_csv, index=False)
    print(f"Saved merged csv to: {out_merged_csv}")
    print(f"Merged rows: {len(merged)}")

    # -------- compute rank correlations --------
    method_pairs = {
        "MoLFormer": {
            "cosine": ("molformer_raw_cosine", "molformer_canonical_cosine"),
            "euclidean": ("molformer_raw_euclidean", "molformer_canonical_euclidean"),
        },
        "GeoDiff 2D": {
            "cosine": ("geodiff_2d_raw_cosine", "geodiff_2d_canonical_cosine"),
            "euclidean": ("geodiff_2d_raw_euclidean", "geodiff_2d_canonical_euclidean"),
        },
        "GeoDiff 3D": {
            "cosine": ("geodiff_3d_raw_cosine", "geodiff_3d_canonical_cosine"),
            "euclidean": ("geodiff_3d_raw_euclidean", "geodiff_3d_canonical_euclidean"),
        },
        "GeoDiff Concat": {
            "cosine": ("geodiff_concat_raw_cosine", "geodiff_concat_canonical_cosine"),
            "euclidean": ("geodiff_concat_raw_euclidean", "geodiff_concat_canonical_euclidean"),
        },
    }

    rows = []
    for method, pair_dict in method_pairs.items():
        cos_raw, cos_can = pair_dict["cosine"]
        euc_raw, euc_can = pair_dict["euclidean"]

        cos_corr = safe_spearman(merged[cos_raw], merged[cos_can])
        euc_corr = safe_spearman(merged[euc_raw], merged[euc_can])

        rows.append({
            "Method": method,
            "Cosine distance rank correlation": format_corr(cos_corr),
            "Euclidean distance rank correlation": format_corr(euc_corr),
        })

    corr_df = pd.DataFrame(rows)
    print("\nRank correlation summary:")
    print(corr_df)

    latex_table = build_latex_table(corr_df)

    print("\nLaTeX table:\n")
    print(latex_table)


if __name__ == "__main__":
    main()
