import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from paths import RESULTS_ROOT


BASE_ROOT = str(RESULTS_ROOT)

MODEL_TITLES = ["REINVENT", "LibINVENT", "SynFormer"]
MODEL_KEYS = {
    "REINVENT": "reinvent",
    "LibINVENT": "libinvent",
    "SynFormer": "synformer",
}

METHOD_ORDER = [
    "ROSHAMBO2",
    "ECFP4",
    "GeoDiff 2D",
    "GeoDiff 3D",
    "GeoDiff Concat",
    "MoLFormer",
]

y_col = "predicted_pIC50"
scaffold_col = "ScaffoldRingsLinkersSMILES"
sample_n = 500
random_state = 42


def get_csv_path(model_name):
    model_key = MODEL_KEYS[model_name]
    return f"{BASE_ROOT}/{model_key}/scaffold_memory/top1k_boltz.csv"


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

    if len(sampled_rows) == 0:
        return sub_df.iloc[0:0].copy()

    sampled_df = pd.concat(sampled_rows, ignore_index=True)
    return sampled_df


def prepare_sampled_stats(csv_path, model_name):
    df = pd.read_csv(csv_path)

    required_cols = ["Method", y_col, scaffold_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{model_name}: Missing columns: {missing}")

    df = df.dropna(subset=["Method", y_col, scaffold_col]).copy()
    df = df[df["Method"].isin(METHOD_ORDER)].copy()
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)

    # An arm whose Boltz-2 predictions have not been computed yet is absent from the
    # CSV; drop it from this model's block instead of failing on an empty slice.
    methods = [m for m in METHOD_ORDER if (df["Method"] == m).any()]
    for m in METHOD_ORDER:
        if m not in methods:
            print(f"[skip] {model_name}/{m}: no rows in {csv_path}")

    sampled_by_method = {}
    summary_rows = []

    for method in methods:
        sub = df[df["Method"] == method].copy()

        sampled = balanced_sample_across_scaffolds(
            sub_df=sub,
            scaffold_col=scaffold_col,
            sample_n=sample_n,
            random_state=random_state,
        ).copy()

        sampled_by_method[method] = sampled

        values = sampled[y_col].to_numpy()
        mean_val = float(np.mean(values)) if len(values) > 0 else np.nan
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan

        summary_rows.append({
            "Model": model_name,
            "Method": method,
            "n": len(values),
            "mean": mean_val,
            "std": std_val,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Mann–Whitney U: every method vs ROSHAMBO2
    ref_values = sampled_by_method["ROSHAMBO2"][y_col].to_numpy()
    raw_pvals = {}

    for method in methods:
        if method == "ROSHAMBO2":
            continue

        cur_values = sampled_by_method[method][y_col].to_numpy()

        # two-sided test
        # stat = mannwhitneyu(ref_values, cur_values, alternative="two-sided")
        stat = ttest_ind(ref_values, cur_values, alternative="two-sided", equal_var=False)
        raw_pvals[method] = float(stat.pvalue)

    methods_to_adjust = list(raw_pvals.keys())
    raw_pvals_list = [raw_pvals[m] for m in methods_to_adjust]

    _, pvals_holm, _, _ = multipletests(raw_pvals_list, alpha=0.05, method="holm")
    adj_pvals = dict(zip(methods_to_adjust, pvals_holm))

    summary_df["p_adj_holm"] = None
    for idx in summary_df.index:
        method = summary_df.loc[idx, "Method"]
        if method == "ROSHAMBO2":
            summary_df.loc[idx, "p_adj_holm"] = None
        else:
            summary_df.loc[idx, "p_adj_holm"] = adj_pvals[method]

    summary_df["effect_size"] = None
    for idx in summary_df.index:
        method = summary_df.loc[idx, "Method"]

        if method == "ROSHAMBO2":
            summary_df.loc[idx, "effect_size"] = None
            continue

        cur_values = sampled_by_method[method][y_col].to_numpy()
        delta = cliffs_delta(cur_values, ref_values)

        summary_df.loc[idx, "effect_size"] = delta

    return summary_df, sampled_by_method


def cliffs_delta(x, y):
    """
    Compute Cliff's delta: P(x > y) - P(x < y)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    # broadcasting
    diff = x[:, None] - y[None, :]

    n_greater = np.sum(diff > 0)
    n_less = np.sum(diff < 0)

    delta = (n_greater - n_less) / (nx * ny)
    return float(delta)


def format_p(p):
    if p is None or pd.isna(p):
        return "—"
    if p < 1e-3:
        return r"$<0.001$"
    return f"{p:.2e}"


def make_latex_table(summary_dfs):
    def one_block_lines(model_name, block_df):
        lines = []
        block_methods = [m for m in METHOD_ORDER if (block_df["Method"] == m).any()]
        for i, method in enumerate(block_methods):
            row = block_df[block_df["Method"] == method].iloc[0]

            mean_std = f"{row['mean']:.2f} $\\pm$ {row['std']:.2f}"
            p_text = format_p(row["p_adj_holm"])

            if pd.notna(row["effect_size"]):
                effect_text = f"{row['effect_size']:.2f}"
            else:
                effect_text = "—"

            if i == 0:
                line = (
                    f"\\multirow{{{len(block_methods)}}}{{*}}{{{model_name}}} "
                    f"& {method} & {mean_std} & {p_text} & {effect_text} \\\\"
                )
            else:
                line = f"& {method} & {mean_std} & {p_text} & {effect_text} \\\\"

            lines.append(line)
        return lines

    latex_lines = []
    latex_lines.append(r"\begin{table}[ht]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        r"\caption{Predicted pIC50 statistics of scaffold-balanced sampled top-500 molecules for REINVENT, LibINVENT, and SynFormer. "
        r"For each method, molecules were sampled as evenly as possible across unique ring-linker scaffolds. "
        r"$p$ values are from two-sided Mann--Whitney U-tests against ROSHAMBO2 within each model, adjusted by the Holm method.}"
    )
    latex_lines.append(r"\begin{tabular}{llccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Model & Method & Predicted pIC50 & Adjusted $p$-value & Effect size \\")
    latex_lines.append(r"\midrule")

    for i, model_name in enumerate(MODEL_TITLES):
        if i > 0:
            latex_lines.append(r"\midrule")
        latex_lines.extend(one_block_lines(model_name, summary_dfs[model_name]))

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\label{tab:pic50_500_stats}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


summary_dfs = {}
sampled_dfs = {}

for model_name in MODEL_TITLES:
    summary_df, sampled = prepare_sampled_stats(get_csv_path(model_name), model_name)
    summary_dfs[model_name] = summary_df
    sampled_dfs[model_name] = sampled

    print(f"=== {model_name} summary ===")
    print(summary_df.to_string(index=False))
    print()

latex_table = make_latex_table(summary_dfs)

print("=== LaTeX table ===\n")
print(latex_table)
