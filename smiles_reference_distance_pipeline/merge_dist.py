import os
import glob
import pandas as pd


# ============================================================
# paths
# ============================================================
base_csv = "/mnt/depts/dept01/misd/molml/embeddings/200k_GeoDiff_emb/similarity_distance_200k_with_all_distances.csv"
prepared_library_csv = "/mnt/depts/dept01/misd/molml/embeddings/200k_emb/prepared/library.csv"

distance_dir = "/mnt/depts/dept01/misd/molml/embeddings/200k_emb"

out_csv = os.path.join(distance_dir, "similarity_distance_200k.csv")


# ============================================================
# 1. read base csv and keep selected columns
# ============================================================
print(f"[INFO] Reading base csv: {base_csv}")
base_df = pd.read_csv(base_csv)

keep_cols = ["smiles", "3d_similarity_ROCS", "ShapeTanimoto", "ColorTanimoto"]
missing_base_cols = [c for c in keep_cols if c not in base_df.columns]
if missing_base_cols:
    raise ValueError(f"Missing required columns in base csv: {missing_base_cols}")

base_df = base_df[keep_cols].copy()
print(f"[INFO] Base df shape after keeping selected columns: {base_df.shape}")


# ============================================================
# 2. read prepared_library.csv
# ============================================================
print(f"[INFO] Reading prepared library csv: {prepared_library_csv}")
prep_df = pd.read_csv(prepared_library_csv)

required_prep_cols = ["smiles", "raw_smiles"]
missing_prep_cols = [c for c in required_prep_cols if c not in prep_df.columns]
if missing_prep_cols:
    raise ValueError(f"Missing required columns in prepared_library.csv: {missing_prep_cols}")

prep_df = prep_df[["smiles", "raw_smiles"]].drop_duplicates().copy()
print(f"[INFO] Prepared library df shape: {prep_df.shape}")


# ============================================================
# 3. find all distance csv files
# ============================================================
pattern = os.path.join(distance_dir, "**", "smiles_reference_*_distances.csv")
distance_csvs = sorted(glob.glob(pattern, recursive=True))

if len(distance_csvs) == 0:
    raise FileNotFoundError(f"No files found matching pattern: {pattern}")

print("[INFO] Found distance files:")
for p in distance_csvs:
    print("   ", p)


# ============================================================
# 4. merge each distance csv
# ============================================================
merged_df = base_df.copy()

for dist_csv in distance_csvs:
    print(f"\n[INFO] Processing: {dist_csv}")
    dist_df = pd.read_csv(dist_csv)

    if "smiles" not in dist_df.columns:
        raise ValueError(f"'smiles' column not found in {dist_csv}")

    # all the cols except smiles
    value_cols = [c for c in dist_df.columns if c != "smiles"]
    if len(value_cols) == 0:
        print(f"[WARNING] No value columns found in {dist_csv}, skipped.")
        continue

    basename = os.path.splitext(os.path.basename(dist_csv))[0]
    rename_map = {}
    for c in value_cols:
        if c in merged_df.columns:
            rename_map[c] = f"{basename}__{c}"
    if rename_map:
        dist_df = dist_df.rename(columns=rename_map)
        value_cols = [rename_map.get(c, c) for c in value_cols]

    # --------------------------------------------------------
    # distance smiles -> prepared_library smiles
    # prepared_library raw_smiles -> base smiles
    #
    # merge steps:
    #   merge dist_df.smiles to prep_df.smiles, by using smiles
    #   get the corresponding raw_smiles
    #   merge to merged_df.smiles using raw_smiles as the key
    # --------------------------------------------------------
    tmp_df = dist_df.merge(
        prep_df,
        on="smiles",
        how="left",
        validate="many_to_one",
    )

    n_missing_raw = tmp_df["raw_smiles"].isna().sum()
    if n_missing_raw > 0:
        print(f"[WARNING] {n_missing_raw} rows in {dist_csv} could not be matched to prepared_library via smiles")

    # keep raw_smiles + value cols
    tmp_df = tmp_df[["raw_smiles"] + value_cols].copy()

    # rename raw_smiles to smiles, so as to merge to base_df
    tmp_df = tmp_df.rename(columns={"raw_smiles": "smiles"})

    before_dedup = len(tmp_df)
    tmp_df = tmp_df.drop_duplicates(subset=["smiles"])
    after_dedup = len(tmp_df)
    if before_dedup != after_dedup:
        print(f"[INFO] Dropped {before_dedup - after_dedup} duplicated rows after raw_smiles mapping")

    # merge into final df
    old_shape = merged_df.shape
    merged_df = merged_df.merge(
        tmp_df,
        on="smiles",
        how="left",
        validate="many_to_one",
    )
    print(f"[INFO] Merged shape: {old_shape} -> {merged_df.shape}")


# ============================================================
# 5. save
# ============================================================
merged_df.to_csv(out_csv, index=False)
print(f"\n[INFO] Saved merged csv to: {out_csv}")
print(f"[INFO] Final shape: {merged_df.shape}")
print("[INFO] Final columns:")
for c in merged_df.columns:
    print("   ", c)
