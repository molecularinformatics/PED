import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from paths import RESULTS_ROOT

MODEL_TITLES = ["REINVENT", "LibINVENT", "SynFormer"]

MODEL_KEYS = {
    "REINVENT": "reinvent",
    "LibINVENT": "libinvent",
    "SynFormer": "synformer",
}

BASE_ROOT = str(RESULTS_ROOT)

METHOD_ORDER = [
    "ROSHAMBO2",
    "ECFP4",
    "GeoDiff 2D",
    "GeoDiff 3D",
    "GeoDiff Concat",
    "MoLFormer",
]

method_colors = {
    "ROSHAMBO2": "#15751a",
    "ECFP4": "#949494",           # dark gray (2D fingerprint baseline)
    "GeoDiff 3D": "#8ac0f2",
    "GeoDiff 2D": "#438cd0",
    "GeoDiff Concat": "#235787",
    "MoLFormer": "#d15353",
}

y_col = "predicted_pIC50"
scaffold_col = "ScaffoldRingsLinkersSMILES"
sample_n = 500
random_state = 42

out_dir = "./generated_analysis_outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "top1k_pIC50_boxplot_scaffold_balanced_sample500_all.pdf")


def get_csv_path(model_title):
    model_key = MODEL_KEYS[model_title]
    return os.path.join(BASE_ROOT, model_key, "scaffold_memory", "top1k_boltz.csv")


def balanced_sample_across_scaffolds(sub_df, scaffold_col, sample_n=500, random_state=42):
    """
    Sample up to sample_n molecules from sub_df as evenly as possible across scaffolds.

    Strategy:
    1. Randomize rows within each scaffold once.
    2. Repeatedly take one molecule from each scaffold that still has unused molecules.
    3. Continue round-robin until reaching sample_n or exhausting all molecules.
    """
    groups = {}
    for scaffold, g in sub_df.groupby(scaffold_col):
        g_shuffled = g.sample(frac=1, random_state=random_state).reset_index(drop=True)
        groups[scaffold] = g_shuffled

    scaffold_names = sorted(groups.keys())
    pointers = {scaffold: 0 for scaffold in scaffold_names}

    sampled_rows = []
    total_available = len(sub_df)
    target_n = min(sample_n, total_available)

    while len(sampled_rows) < target_n:
        added_this_round = 0

        for scaffold in scaffold_names:
            ptr = pointers[scaffold]
            g = groups[scaffold]

            if ptr < len(g):
                sampled_rows.append(g.iloc[[ptr]])
                pointers[scaffold] += 1
                added_this_round += 1

                if len(sampled_rows) >= target_n:
                    break

        if added_this_round == 0:
            break

    sampled_df = pd.concat(sampled_rows, ignore_index=True)

    return sampled_df


def prepare_plot_df(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["Method", y_col, scaffold_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["Method", y_col, scaffold_col]).copy()
    df = df[df["Method"].isin(METHOD_ORDER)].copy()
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)

    sampled_parts = []

    # An arm whose Boltz-2 predictions have not been computed yet is absent from
    # the CSV; drop it from the figure instead of drawing an empty slot.
    for method in [m for m in METHOD_ORDER if (df["Method"] == m).any()]:
        sub = df[df["Method"] == method].copy()

        sampled = balanced_sample_across_scaffolds(
            sub_df=sub,
            scaffold_col=scaffold_col,
            sample_n=sample_n,
            random_state=random_state,
        )

        sampled_parts.append(sampled)

    plot_df = pd.concat(sampled_parts, ignore_index=True)
    return plot_df


plot_dfs = {
    model_title: prepare_plot_df(get_csv_path(model_title))
    for model_title in MODEL_TITLES
}


# =========================
# plot 1x3
# =========================
# keep an arm on the shared x axis only if at least one model has it
_present = set()
for _df in plot_dfs.values():
    _present |= set(_df["Method"].unique())
METHOD_ORDER = [m for m in METHOD_ORDER if m in _present]

fig, axes = plt.subplots(1, 3, figsize=(9.75, 4.5), sharey=True)

for i, model_title in enumerate(MODEL_TITLES):
    sns.boxplot(
        data=plot_dfs[model_title],
        x="Method",
        y=y_col,
        order=METHOD_ORDER,
        palette=method_colors,
        width=0.2,
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "gray",
            "markeredgecolor": "gray",
            "alpha": 0.6,
            "markeredgewidth": 0.0,
        },
        linewidth=1.2,
        ax=axes[i],
    )

    axes[i].set_title(model_title, fontsize=12)
    axes[i].set_xlabel("Method", fontsize=12)
    plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if i == 0:
        axes[i].set_ylabel("Predicted pIC50", fontsize=12)
        axes[i].yaxis.set_major_locator(MaxNLocator(nbins=7))
    else:
        axes[i].set_ylabel("")
        axes[i].tick_params(axis='y', left=False, labelleft=False)

# -------- shared styling --------
for ax in axes:
    ax.grid(axis="y", which="major", alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")

print(f"\nSaved figure to: {out_path}")
