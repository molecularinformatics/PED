from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from paths import RESULTS_ROOT, OUT_DIR

BASE_ROOT = RESULTS_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent

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

METHOD_ORDER = list(FILE_SUFFIXES.keys())

method_colors = {
    "ROSHAMBO2": "#15751a",        # dark green
    "ECFP4": "#949494",            # neutral gray (2D fingerprint baseline)
    "GeoDiff 3D": "#8ac0f2",       # light blue
    "GeoDiff 2D": "#438cd0",       # medium blue
    "GeoDiff Concat": "#235787",   # dark blue
    "MoLFormer": "#d15353",        # dark red
}


def get_file_path(model_key, suffix):
    return BASE_ROOT / model_key / "scaffold_memory" / f"{model_key}_{suffix}.csv"


def compute_quantile_scaffold_ratio(df):
    required_cols = ["SMILES", "total_score", "ScaffoldRingsLinkersSMILES"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["total_score"]).copy()

    if len(df) == 0:
        raise ValueError("No rows remaining after dropping NaN total_score.")

    # qcut may fail when too many duplicated values exist
    df["score_bin"] = pd.qcut(
        df["total_score"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop"
    )

    ratios = []
    counts = []

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[df["score_bin"] == q]

        mol_count = len(sub)
        unique_scaffold_count = (
            sub["ScaffoldRingsLinkersSMILES"]
            .dropna()
            .astype(str)
            .nunique()
        )

        ratio = unique_scaffold_count / mol_count if mol_count > 0 else 0.0

        ratios.append(ratio)
        counts.append(mol_count)

    return ratios, counts


def load_df(path):
    df = pd.read_csv(path)

    df.rename(columns={"score": "total_score"}, inplace=True)
    if "smiles" in df.columns:
        df.rename(columns={"smiles": "SMILES"}, inplace=True)

    if "SMILES" not in df.columns:
        raise ValueError(f"'SMILES' column not found in file: {path}")

    df.drop_duplicates(subset=["SMILES"], inplace=True)
    return df


def collect_results(model_title):
    results = {}
    model_key = MODEL_KEYS[model_title]

    for method in METHOD_ORDER:
        path = get_file_path(model_key, FILE_SUFFIXES[method])

        if not path.exists():
            print(f"\n[{model_title}] {method}: SKIPPED (file not found: {path})")
            continue

        df = load_df(path)
        ratios, counts = compute_quantile_scaffold_ratio(df)
        results[method] = ratios

        print(f"\n[{model_title}] {method}")
        for q, r, c in zip(["Q1", "Q2", "Q3", "Q4"], ratios, counts):
            print(f"  {q}: ratio={r:.4f}, n={c}")

    return results


def plot_panel(ax, results, subtitle, show_ylabel=False):
    x = [1, 2, 3, 4]

    for method in METHOD_ORDER:
        if method not in results:
            continue
        ax.plot(
            x,
            results[method],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=method_colors[method],
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=12)
    ax.set_xlabel("Total Score Quantile", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("Unique Scaffold Ratio", fontsize=12)

    ax.set_title(subtitle, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)


def main():
    all_results = {
        model_title: collect_results(model_title)
        for model_title in MODEL_TITLES
    }

    fig, axes = plt.subplots(1, len(MODEL_TITLES), figsize=(10, 4), sharey=True)

    for i, model_title in enumerate(MODEL_TITLES):
        plot_panel(
            ax=axes[i],
            results=all_results[model_title],
            subtitle=model_title,
            show_ylabel=(i == 0),
        )

    # use only one legend for the full figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = OUT_DIR / "scaffold_unique_ratio_by_score_all.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    main()
