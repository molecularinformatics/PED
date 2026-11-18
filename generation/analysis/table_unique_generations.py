from pathlib import Path

import pandas as pd
from paths import RESULTS_ROOT

# =========================
# CONFIG
# =========================

BASE_ROOT = RESULTS_ROOT

MODEL_TITLES = ["REINVENT", "LibINVENT", "SynFormer"]
MODEL_KEYS = {
    "REINVENT": "reinvent",
    "LibINVENT": "libinvent",
    "SynFormer": "synformer",
}

FILE_SUFFIXES = {
    "ROSHAMBO2": "roshambo2",
    "ECFP4": "ecfp4",
    "GeoDiff 2D": "geodiff_2d",
    "GeoDiff 3D": "geodiff_3d",
    "GeoDiff Concat": "geodiff_concat",
    "MoLFormer": "molformer",
}


def get_file_path(model_key, suffix):
    return BASE_ROOT / model_key / "scaffold_memory" / f"{model_key}_{suffix}.csv"


# =========================
# CORE FUNCTION
# =========================

def compute_stats(file_path):
    df = pd.read_csv(file_path)

    smiles_col = "SMILES"
    scaffold_col = "ScaffoldRingsLinkersSMILES"
    required_cols = [smiles_col, scaffold_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path} is missing required columns: {missing_cols}")

    # drop NA
    df = df[[smiles_col, scaffold_col]].dropna()

    n_unique_mol = df[smiles_col].nunique()
    n_unique_scaffold = df[scaffold_col].nunique()

    scaffold_ratio = n_unique_scaffold / n_unique_mol if n_unique_mol > 0 else 0

    return n_unique_mol, n_unique_scaffold, scaffold_ratio


# =========================
# RUN
# =========================

all_results = {}

for model_title in MODEL_TITLES:
    model_key = MODEL_KEYS[model_title]
    model_results = {}

    print(f"=== {model_title} ===")

    for method, suffix in FILE_SUFFIXES.items():
        path = get_file_path(model_key, suffix)
        if not path.exists():
            print(f"{method:15s} | SKIPPED (file not found: {path})")
            continue
        mol, scaffold, ratio = compute_stats(path)
        model_results[method] = (mol, scaffold, ratio)

        print(
            f"{method:15s} | mol={mol:6d} | scaffold={scaffold:6d} | "
            f"ratio={ratio:.4f}"
        )

    all_results[model_title] = model_results
    print()


# =========================
# GENERATE LATEX TABLE ROWS
# =========================

def format_row(method, stats):
    mol, scaffold, ratio = stats
    # return f"& {method:<15s} & {mol:,} & {ratio*100:.2f}\\% \\\\"
    return f"& {method:<15s} & {mol:,} & {scaffold:,} & {ratio*100:.2f}\\%  \\\\"


for model_title in MODEL_TITLES:
    print(f"=== LATEX ({model_title}) ===")
    for method in FILE_SUFFIXES.keys():
        if method not in all_results[model_title]:
            continue
        print(format_row(method, all_results[model_title][method]))
    print()
