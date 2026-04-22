import os
import pandas as pd


root_dir = "/mnt/depts/dept01/misd/molml/embeddings/shuffle"

setting_order = [
    # "original_canonical",
    "original_raw",
    "shuffle_1",
    "original_canonical_conf",
    # "shuffle_5",
    # "shuffle_10",
]

setting_display = {
    "original_raw": "Original raw",
    "shuffle_1": "Shuffle 1x",
    "shuffle_5": "Shuffle 5x",
    "shuffle_10": "Shuffle 10x",
    "original_canonical": "Original canonical",
    "original_canonical_conf": "Different conformers",
}

def mean_std_str(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "NA"
    return f"{x.mean():.4f} $\\pm$ {x.std():.4f}"

rows_by_setting = {}

for setting in setting_order:
    setting_dir = os.path.join(root_dir, setting)
    rows = []

    # ---------- MolFormer ----------
    mol_csv = os.path.join(setting_dir, "molformer_results.csv")
    if os.path.exists(mol_csv):
        df = pd.read_csv(mol_csv)
        rows.append({
            "Method": "MoLFormer",
            "Cosine distance": mean_std_str(df["cosine_distance_to_original_canonical"]),
            "Euclidean distance": mean_std_str(df["euclidean_distance_to_original_canonical"]),
        })

    # ---------- GeoDiff ----------
    geo_csv = os.path.join(setting_dir, "geodiff_results.csv")
    if os.path.exists(geo_csv):
        df = pd.read_csv(geo_csv)

        rows.append({
            "Method": "GeoDiff 2D",
            "Cosine distance": mean_std_str(df["geodiff_2D_cosine"]),
            "Euclidean distance": mean_std_str(df["geodiff_2D_euclidean"]),
        })

        rows.append({
            "Method": "GeoDiff 3D",
            "Cosine distance": mean_std_str(df["geodiff_3D_cosine"]),
            "Euclidean distance": mean_std_str(df["geodiff_3D_euclidean"]),
        })

        rows.append({
            "Method": "GeoDiff Concat",
            "Cosine distance": mean_std_str(df["geodiff_concat_norm_cosine"]),
            "Euclidean distance": mean_std_str(df["geodiff_concat_norm_euclidean"]),
        })

    if len(rows) > 0:
        rows_by_setting[setting] = rows

# -------- LaTeX table with multirow --------
latex_lines = []
latex_lines.append("\\begin{table}[ht]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{\\textbf{Distance between molecular embeddings under different SMILES settings.} Mean $\\pm$ standard deviation of cosine and Euclidean distances between embeddings of the same molecule under each SMILES setting and the original canonical setting. Smaller values indicate stronger representation invariance to SMILES variation.}")
latex_lines.append("\\begin{tabular}{llcc}")
latex_lines.append("\\toprule")
latex_lines.append("SMILES type & Method & Cosine distance & Euclidean distance \\\\")
latex_lines.append("\\midrule")

for setting in setting_order:
    if setting not in rows_by_setting:
        continue

    rows = rows_by_setting[setting]
    n_rows = len(rows)
    setting_name = setting_display.get(setting, setting)

    for i, row in enumerate(rows):
        if i == 0:
            latex_lines.append(
                f"\\multirow{{{n_rows}}}{{*}}{{{setting_name}}} & "
                f"{row['Method']} & {row['Cosine distance']} & {row['Euclidean distance']} \\\\"
            )
        else:
            latex_lines.append(
                f"& {row['Method']} & {row['Cosine distance']} & {row['Euclidean distance']} \\\\"
            )


latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

latex_table = "\n".join(latex_lines)

print(latex_table)