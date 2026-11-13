import argparse
import os

import numpy as np
import pandas as pd

from pipeline_utils import (
    concat_embeddings,
    concat_normalized_embeddings,
    cosine_distance_matrix,
    euclidean_distance_matrix,
    load_emb_npy,
    load_smiles_npy,
)


DUAL_MODE_CHOICES = ["2D", "3D", "concat", "concat_norm"]
SINGLE_MODE_CHOICES = ["single"]
DISTANCE_CHOICES = ["cosine", "euclidean"]
MODEL_CHOICES = ["geodiff", "sde", "molformer", "unimol", "ecfp"]


def load_embedding_table(emb_dir: str, prefix: str) -> tuple[pd.DataFrame, np.ndarray]:
    smiles = load_smiles_npy(os.path.join(emb_dir, "smiles.npy"))
    emb = load_emb_npy(os.path.join(emb_dir, "emb.npy"))
    if len(smiles) != len(emb):
        raise ValueError(f"Length mismatch in {emb_dir}: {len(smiles)} smiles vs {len(emb)} embeddings")

    df = pd.DataFrame({"smiles": pd.Series(smiles).astype(str).str.strip()})
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


def compute_distance_matrix(lib_emb: np.ndarray, ref_emb: np.ndarray, distance_type: str) -> np.ndarray:
    if distance_type == "cosine":
        return cosine_distance_matrix(lib_emb, ref_emb).astype(np.float32)
    if distance_type == "euclidean":
        return euclidean_distance_matrix(lib_emb, ref_emb).astype(np.float32)
    raise ValueError(f"Unsupported distance_type: {distance_type}")


def compute_tanimoto_similarity_matrix(lib_emb: np.ndarray, ref_emb: np.ndarray) -> np.ndarray:
    lib_bin = (np.asarray(lib_emb) > 0).astype(np.float32)
    ref_bin = (np.asarray(ref_emb) > 0).astype(np.float32)
    intersect = lib_bin @ ref_bin.T
    lib_sum = lib_bin.sum(axis=1, keepdims=True)
    ref_sum = ref_bin.sum(axis=1, keepdims=True).T
    union = lib_sum + ref_sum - intersect
    sim = np.divide(intersect, union, out=np.zeros_like(intersect, dtype=np.float32), where=union > 0)
    return sim.astype(np.float32)


def build_dual_mode_embeddings(lib_2d, lib_3d, ref_2d, ref_3d, mode: str):
    if mode == "2D":
        return lib_2d, ref_2d
    if mode == "3D":
        return lib_3d, ref_3d
    if mode == "concat":
        return concat_embeddings(lib_2d, lib_3d), concat_embeddings(ref_2d, ref_3d)
    if mode == "concat_norm":
        return concat_normalized_embeddings(lib_2d, lib_3d), concat_normalized_embeddings(ref_2d, ref_3d)
    raise ValueError(f"Unknown dual mode: {mode}")


def validate_modes(model: str, modes: list[str]) -> list[str]:
    if model in {"geodiff", "sde"}:
        invalid = [m for m in modes if m not in DUAL_MODE_CHOICES]
        if invalid:
            raise ValueError(f"For model={model}, modes must be chosen from {DUAL_MODE_CHOICES}, got invalid modes: {invalid}")
        return modes

    invalid = [m for m in modes if m not in SINGLE_MODE_CHOICES]
    if invalid:
        raise ValueError(f"For model={model}, modes must be chosen from {SINGLE_MODE_CHOICES}, got invalid modes: {invalid}")
    return modes


def load_single_model_embeddings(refs_csv, library_csv, refs_emb_root, library_emb_root, dedup_embeddings):
    refs_meta = pd.read_csv(refs_csv)
    lib_meta = pd.read_csv(library_csv)
    refs_meta["smiles"] = refs_meta["smiles"].astype(str).str.strip()
    lib_meta["smiles"] = lib_meta["smiles"].astype(str).str.strip()

    ref_df, ref_emb = load_embedding_table(refs_emb_root, prefix="ref")
    lib_df, lib_emb = load_embedding_table(library_emb_root, prefix="lib")

    ref_df, ref_emb = deduplicate_embeddings(ref_df, ref_emb, keep=dedup_embeddings)
    lib_df, lib_emb = deduplicate_embeddings(lib_df, lib_emb, keep=dedup_embeddings)

    ref_df = refs_meta.merge(ref_df[["smiles", "ref_row_idx"]], on="smiles", how="inner")
    lib_df = lib_meta.merge(lib_df[["smiles", "lib_row_idx"]], on="smiles", how="inner")

    if len(ref_df) == 0:
        raise ValueError("No reference smiles survived the merge between refs csv and embeddings")
    if len(ref_df) != 1:
        raise ValueError(f"Expected exactly one reference smiles, but got {len(ref_df)} after merge")
    if len(lib_df) == 0:
        raise ValueError("No library smiles survived the merge between library csv and embeddings")

    ref_emb = ref_emb[ref_df["ref_row_idx"].to_numpy()]
    lib_emb = lib_emb[lib_df["lib_row_idx"].to_numpy()]
    return lib_df, lib_emb, ref_emb


def load_dual_embeddings(model_name, refs_csv, library_csv, refs_emb_root, library_emb_root, dedup_embeddings):
    refs_meta = pd.read_csv(refs_csv)
    lib_meta = pd.read_csv(library_csv)
    refs_meta["smiles"] = refs_meta["smiles"].astype(str).str.strip()
    lib_meta["smiles"] = lib_meta["smiles"].astype(str).str.strip()

    ref_df_2d, ref_emb_2d = load_embedding_table(os.path.join(refs_emb_root, f"{model_name}_2D"), prefix="ref")
    ref_df_3d, ref_emb_3d = load_embedding_table(os.path.join(refs_emb_root, f"{model_name}_3D"), prefix="ref")
    lib_df_2d, lib_emb_2d = load_embedding_table(os.path.join(library_emb_root, f"{model_name}_2D"), prefix="lib")
    lib_df_3d, lib_emb_3d = load_embedding_table(os.path.join(library_emb_root, f"{model_name}_3D"), prefix="lib")

    ref_df_2d, ref_emb_2d = deduplicate_embeddings(ref_df_2d, ref_emb_2d, keep=dedup_embeddings)
    ref_df_3d, ref_emb_3d = deduplicate_embeddings(ref_df_3d, ref_emb_3d, keep=dedup_embeddings)
    lib_df_2d, lib_emb_2d = deduplicate_embeddings(lib_df_2d, lib_emb_2d, keep=dedup_embeddings)
    lib_df_3d, lib_emb_3d = deduplicate_embeddings(lib_df_3d, lib_emb_3d, keep=dedup_embeddings)

    ref_df = refs_meta.merge(ref_df_2d[["smiles", "ref_row_idx"]], on="smiles", how="inner")
    ref_df = ref_df.merge(ref_df_3d[["smiles", "ref_row_idx"]], on=["smiles", "ref_row_idx"], how="inner")

    lib_df = lib_meta.merge(lib_df_2d[["smiles", "lib_row_idx"]], on="smiles", how="inner")
    lib_df = lib_df.merge(lib_df_3d[["smiles", "lib_row_idx"]], on=["smiles", "lib_row_idx"], how="inner")

    if len(ref_df) == 0:
        raise ValueError("No reference smiles survived the merge between refs csv and embeddings")
    if len(ref_df) != 1:
        raise ValueError(f"Expected exactly one reference smiles, but got {len(ref_df)} after merge")
    if len(lib_df) == 0:
        raise ValueError("No library smiles survived the merge between library csv and embeddings")

    ref_emb_2d = ref_emb_2d[ref_df["ref_row_idx"].to_numpy()]
    ref_emb_3d = ref_emb_3d[ref_df["ref_row_idx"].to_numpy()]
    lib_emb_2d = lib_emb_2d[lib_df["lib_row_idx"].to_numpy()]
    lib_emb_3d = lib_emb_3d[lib_df["lib_row_idx"].to_numpy()]

    return lib_df, lib_emb_2d, lib_emb_3d, ref_emb_2d, ref_emb_3d


def main():
    parser = argparse.ArgumentParser(description="Compute distances from one reference SMILES to one library CSV")
    parser.add_argument("--refs_csv", required=True)
    parser.add_argument("--library_csv", required=True)
    parser.add_argument("--refs_emb_root", required=True)
    parser.add_argument("--library_emb_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--model", type=str, default="geodiff", choices=MODEL_CHOICES)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--distance_type", nargs="+", default=DISTANCE_CHOICES, choices=DISTANCE_CHOICES)
    parser.add_argument("--dedup_embeddings", default="first", choices=["first"])
    args = parser.parse_args()

    if args.modes is None:
        args.modes = DUAL_MODE_CHOICES if args.model in ["geodiff", "sde"] else SINGLE_MODE_CHOICES
    args.modes = validate_modes(args.model, args.modes)

    if args.model in ["geodiff", "sde"]:
        lib_df, lib_emb_2d, lib_emb_3d, ref_emb_2d, ref_emb_3d = load_dual_embeddings(
            model_name=args.model,
            refs_csv=args.refs_csv,
            library_csv=args.library_csv,
            refs_emb_root=args.refs_emb_root,
            library_emb_root=args.library_emb_root,
            dedup_embeddings=args.dedup_embeddings,
        )

        out_df = pd.DataFrame({"smiles": lib_df["smiles"].astype(str).to_numpy()})
        for mode in args.modes:
            lib_mode, ref_mode = build_dual_mode_embeddings(lib_emb_2d, lib_emb_3d, ref_emb_2d, ref_emb_3d, mode=mode)
            for distance in args.distance_type:
                dist_mat = compute_distance_matrix(lib_mode, ref_mode, distance_type=distance)
                out_df[f"{args.model}_{mode}_{distance}"] = dist_mat[:, 0]
    else:
        lib_df, lib_emb, ref_emb = load_single_model_embeddings(
            refs_csv=args.refs_csv,
            library_csv=args.library_csv,
            refs_emb_root=args.refs_emb_root,
            library_emb_root=args.library_emb_root,
            dedup_embeddings=args.dedup_embeddings,
        )
        out_df = pd.DataFrame({"smiles": lib_df["smiles"].astype(str).to_numpy()})
        if args.model == "ecfp":
            sim_mat = compute_tanimoto_similarity_matrix(lib_emb, ref_emb)
            out_df["ecfp_tanimoto"] = sim_mat[:, 0]
        else:
            for distance in args.distance_type:
                dist_mat = compute_distance_matrix(lib_emb, ref_emb, distance_type=distance)
                out_df[f"{args.model}_{distance}"] = dist_mat[:, 0]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved csv: {args.out_csv}")
    print("Reference compounds matched: 1")
    print(f"Library compounds matched: {len(out_df)}")


if __name__ == "__main__":
    main()
