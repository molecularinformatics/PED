import os
import argparse

import numpy as np
import pandas as pd

from lit_pcba_utils import (
    concat_normalized_embeddings,
    cosine_distance_matrix,
    euclidean_distance_matrix,
    load_emb_npy,
    load_smiles_npy,
    sanitize_name,
)


def load_embedding_table(emb_dir: str, prefix: str) -> pd.DataFrame:
    smiles = load_smiles_npy(os.path.join(emb_dir, "smiles.npy"))
    emb = load_emb_npy(os.path.join(emb_dir, "emb.npy"))
    if len(smiles) != len(emb):
        raise ValueError(f"Length mismatch in {emb_dir}: {len(smiles)} smiles vs {len(emb)} embeddings")
    df = pd.DataFrame({"smiles": smiles})
    df[f"{prefix}_row_idx"] = np.arange(len(df), dtype=np.int64)
    return df, emb


def deduplicate_embeddings(df: pd.DataFrame, emb: np.ndarray, keep: str):
    if keep != "first":
        raise ValueError("Only keep='first' is currently supported for embedding dedup")
    keep_idx = df.drop_duplicates(subset=["smiles"], keep="first").index.to_numpy()
    out_df = df.loc[keep_idx].reset_index(drop=True)
    out_emb = emb[keep_idx]
    # The row index is a position into emb, so it has to follow the dedup.
    idx_col = next(c for c in out_df.columns if c.endswith("_row_idx"))
    out_df[idx_col] = np.arange(len(out_df), dtype=np.int64)
    return out_df, out_emb


def deduplicate_references(refs_meta: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """Keep one row per distinct reference SMILES. The kept ``ref_name`` is arbitrary
    among identical ligands; the SMILES, and so the distance, is not."""
    before = len(refs_meta)
    out = refs_meta.drop_duplicates(subset=["smiles"], keep=keep).reset_index(drop=True)
    if before != len(out):
        print(f"[refs] {before} reference rows -> {len(out)} distinct SMILES "
              f"({before - len(out)} duplicate(s) dropped)")
    return out


def check_reference_merge(ref_df: pd.DataFrame, refs_meta: pd.DataFrame, what: str) -> None:
    """Fail if the merge expanded the reference set or repeated a column name."""
    if len(ref_df) != len(refs_meta):
        raise ValueError(
            f"{what}: reference merge changed the row count, {len(refs_meta)} -> "
            f"{len(ref_df)}; a duplicate SMILES corrupts the per-reference columns "
            f"and the pooled minimum."
        )
    names = [sanitize_name(n) for n in ref_df["ref_name"]]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"{what}: duplicate reference column names after merge: {dupes}")


def build_mode_embeddings(lib_2d, lib_3d, ref_2d, ref_3d, mode: str):
    if mode == "2D":
        return lib_2d, ref_2d
    if mode == "3D":
        return lib_3d, ref_3d
    if mode == "concat":
        return concat_normalized_embeddings(lib_2d, lib_3d), concat_normalized_embeddings(ref_2d, ref_3d)
    raise ValueError(f"Unknown mode: {mode}")


def compute_distance_matrix(lib_emb: np.ndarray, ref_emb: np.ndarray, distance_type: str) -> np.ndarray:
    if distance_type == "cosine":
        return cosine_distance_matrix(lib_emb, ref_emb).astype(np.float32)
    if distance_type == "euclidean":
        return euclidean_distance_matrix(lib_emb, ref_emb).astype(np.float32)
    raise ValueError(f"Unsupported distance_type: {distance_type}")


def compute_geodiff(args):
    refs_meta = pd.read_csv(args.refs_csv)
    lib_meta = pd.read_csv(args.library_csv)
    refs_meta["smiles"] = refs_meta["smiles"].astype(str).str.strip()
    lib_meta["smiles"] = lib_meta["smiles"].astype(str).str.strip()

    ref_df_2d, ref_emb_2d = load_embedding_table(os.path.join(args.refs_emb_root, "2D"), prefix="ref")
    ref_df_3d, ref_emb_3d = load_embedding_table(os.path.join(args.refs_emb_root, "3D"), prefix="ref")
    lib_df_2d, lib_emb_2d = load_embedding_table(os.path.join(args.library_emb_root, "2D"), prefix="lib")
    lib_df_3d, lib_emb_3d = load_embedding_table(os.path.join(args.library_emb_root, "3D"), prefix="lib")

    # Collapse duplicate reference SMILES on both sides, or the merge expands.
    refs_meta = deduplicate_references(refs_meta, keep=args.dedup_embeddings)
    ref_df_2d, ref_emb_2d = deduplicate_embeddings(ref_df_2d, ref_emb_2d, keep=args.dedup_embeddings)
    ref_df_3d, ref_emb_3d = deduplicate_embeddings(ref_df_3d, ref_emb_3d, keep=args.dedup_embeddings)
    lib_df_2d, lib_emb_2d = deduplicate_embeddings(lib_df_2d, lib_emb_2d, keep=args.dedup_embeddings)
    lib_df_3d, lib_emb_3d = deduplicate_embeddings(lib_df_3d, lib_emb_3d, keep=args.dedup_embeddings)

    ref_df = refs_meta.merge(ref_df_2d[["smiles", "ref_row_idx"]], on="smiles", how="inner")
    ref_df = ref_df.merge(ref_df_3d[["smiles", "ref_row_idx"]], on=["smiles", "ref_row_idx"], how="inner")
    check_reference_merge(ref_df, refs_meta, "geodiff")

    lib_df = lib_meta.merge(lib_df_2d[["smiles", "lib_row_idx"]], on="smiles", how="inner")
    lib_df = lib_df.merge(lib_df_3d[["smiles", "lib_row_idx"]], on=["smiles", "lib_row_idx"], how="inner")

    if len(ref_df) == 0:
        raise ValueError("No reference smiles survived the merge between refs csv and embeddings")
    if len(lib_df) == 0:
        raise ValueError("No library smiles survived the merge between library csv and embeddings")

    ref_emb_2d = ref_emb_2d[ref_df["ref_row_idx"].to_numpy()]
    ref_emb_3d = ref_emb_3d[ref_df["ref_row_idx"].to_numpy()]
    lib_emb_2d = lib_emb_2d[lib_df["lib_row_idx"].to_numpy()]
    lib_emb_3d = lib_emb_3d[lib_df["lib_row_idx"].to_numpy()]

    out_df = lib_df.copy()
    for mode in args.modes:
        for distance in args.distance_type:
            lib_mode, ref_mode = build_mode_embeddings(lib_emb_2d, lib_emb_3d, ref_emb_2d, ref_emb_3d, mode=mode)
            dist_mat = compute_distance_matrix(lib_mode, ref_mode, distance_type=distance)

            for ref_col_idx, row in ref_df.reset_index(drop=True).iterrows():
                ref_name = sanitize_name(row["ref_name"])
                col_name = f"{mode}_{distance}_dist_{ref_name}"
                out_df[col_name] = dist_mat[:, ref_col_idx]

            pooled_name = f"{mode}_{distance}_dist_min"
            out_df[pooled_name] = dist_mat.min(axis=1)

    return out_df, len(ref_df)


def compute_molformer(args):
    refs_meta = pd.read_csv(args.refs_csv)
    lib_meta = pd.read_csv(args.library_csv)
    refs_meta["smiles"] = refs_meta["smiles"].astype(str).str.strip()
    lib_meta["smiles"] = lib_meta["smiles"].astype(str).str.strip()

    ref_df_emb, ref_emb = load_embedding_table(args.refs_emb_root, prefix="ref")
    lib_df_emb, lib_emb = load_embedding_table(args.library_emb_root, prefix="lib")

    refs_meta = deduplicate_references(refs_meta, keep=args.dedup_embeddings)
    ref_df_emb, ref_emb = deduplicate_embeddings(ref_df_emb, ref_emb, keep=args.dedup_embeddings)
    lib_df_emb, lib_emb = deduplicate_embeddings(lib_df_emb, lib_emb, keep=args.dedup_embeddings)

    ref_df = refs_meta.merge(ref_df_emb[["smiles", "ref_row_idx"]], on="smiles", how="inner")
    lib_df = lib_meta.merge(lib_df_emb[["smiles", "lib_row_idx"]], on="smiles", how="inner")
    check_reference_merge(ref_df, refs_meta, "molformer")

    if len(ref_df) == 0:
        raise ValueError("No reference smiles survived the merge between refs csv and molformer embeddings")
    if len(lib_df) == 0:
        raise ValueError("No library smiles survived the merge between library csv and molformer embeddings")

    ref_emb = ref_emb[ref_df["ref_row_idx"].to_numpy()]
    lib_emb = lib_emb[lib_df["lib_row_idx"].to_numpy()]

    out_df = lib_df.copy()
    for distance in args.distance_type:
        dist_mat = compute_distance_matrix(lib_emb, ref_emb, distance_type=distance)
        for ref_col_idx, row in ref_df.reset_index(drop=True).iterrows():
            ref_name = sanitize_name(row["ref_name"])
            col_name = f"molformer_{distance}_dist_{ref_name}"
            out_df[col_name] = dist_mat[:, ref_col_idx]
        out_df[f"molformer_{distance}_dist_min"] = dist_mat.min(axis=1)

    return out_df, len(ref_df)


def main():
    parser = argparse.ArgumentParser(description="Compute reference-to-library distances for one LIT-PCBA target")
    parser.add_argument("--refs_csv", required=True)
    parser.add_argument("--library_csv", required=True)
    parser.add_argument("--refs_emb_root", required=True)
    parser.add_argument("--library_emb_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--embedding_backend", default="geodiff", choices=["geodiff", "molformer"])
    parser.add_argument("--modes", nargs="+", default=["2D", "3D", "concat"], choices=["2D", "3D", "concat"])
    parser.add_argument("--distance_type", nargs="+", default=["cosine", "euclidean"], choices=["cosine", "euclidean"])
    parser.add_argument("--dedup_embeddings", default="first", choices=["first"])
    args = parser.parse_args()

    if args.embedding_backend == "geodiff":
        out_df, n_ref = compute_geodiff(args)
    else:
        out_df, n_ref = compute_molformer(args)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved distance csv: {args.out_csv}")
    print(f"Embedding backend: {args.embedding_backend}")
    print(f"Reference compounds matched: {n_ref}")
    print(f"Library compounds matched: {len(out_df)}")


if __name__ == "__main__":
    main()
