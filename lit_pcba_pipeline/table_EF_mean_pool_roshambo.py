import argparse
import ast
import numpy as np
import pandas as pd


TARGETS = [
    "ADRB2",
    "ALDH1",
    "ESR1_ago",
    "ESR1_ant",
    "FEN1",
    "GBA",
    "IDH1",
    "KAT2A",
    "MAPK1",
    "MTORC1",
    "OPRK1",
    "PKM2",
    "PPARG",
    "TP53",
    "VDR",
]


# fixed metrics from your pasted tables, restricted to the 15 targets above
# format:
# target: {"mean": ..., "sd": ..., "pooled": ...}
FIXED_METRICS = {
    "2D ECFP4 similarity": {
        "ADRB2":   {"mean": 18.38, "sd": 6.62, "pooled": 17.65},
        "ALDH1":   {"mean": 1.40, "sd": 0.50, "pooled": 2.61},
        "ESR1_ago": {"mean": 7.52, "sd": 6.09, "pooled": 7.69},
        "ESR1_ant": {"mean": 1.39, "sd": 1.06, "pooled": 0.98},
        "FEN1":    {"mean": 1.08, "sd": None, "pooled": 1.08},
        "GBA":     {"mean": 2.01, "sd": 1.41, "pooled": 3.61},
        "IDH1":    {"mean": 1.83, "sd": 3.09, "pooled": 5.13},
        "KAT2A":   {"mean": 0.17, "sd": 0.30, "pooled": 0.52},
        "MAPK1":   {"mean": 1.20, "sd": 0.74, "pooled": 1.95},
        "MTORC1":  {"mean": 0.28, "sd": 0.48, "pooled": 0.00},
        "OPRK1":   {"mean": 12.50, "sd": None, "pooled": 12.50},
        "PKM2":    {"mean": 3.30, "sd": 2.73, "pooled": 7.95},
        "PPARG":   {"mean": 3.65, "sd": 5.14, "pooled": 7.41},
        "TP53":    {"mean": 1.06, "sd": 0.95, "pooled": 0.00},
        "VDR":     {"mean": 3.37, "sd": None, "pooled": 3.37},
    },
    "ROCS 3D similarity": {
        "ADRB2":   {"mean": 9.76, "sd": 5.35, "pooled": 20.00},
        "ALDH1":   {"mean": 1.20, "sd": 0.39, "pooled": 1.85},
        "ESR1_ago": {"mean": 4.61, "sd": 4.86, "pooled": 7.69},
        "ESR1_ant": {"mean": 1.66, "sd": 1.55, "pooled": 3.92},
        "FEN1":    {"mean": 0.27, "sd": None, "pooled": 0.27},
        "GBA":     {"mean": 0.72, "sd": 0.35, "pooled": 0.86},
        "IDH1":    {"mean": 0.86, "sd": 2.32, "pooled": 0.00},
        "KAT2A":   {"mean": 1.48, "sd": 0.07, "pooled": 1.55},
        "MAPK1":   {"mean": 1.96, "sd": 1.01, "pooled": 5.19},
        "MTORC1":  {"mean": 1.38, "sd": 1.29, "pooled": 3.23},
        "OPRK1":   {"mean": 0.00, "sd": None, "pooled": 0.00},
        "PKM2":    {"mean": 4.86, "sd": 4.30, "pooled": 6.22},
        "PPARG":   {"mean": 8.79, "sd": 5.24, "pooled": 18.52},
        "TP53":    {"mean": 0.88, "sd": 1.36, "pooled": 2.53},
        "VDR":     {"mean": 0.35, "sd": None, "pooled": 0.35},
    },
    "Molecular docking": {
        "ADRB2":   {"mean": 2.94, "sd": 3.14, "pooled": 5.88},
        "ALDH1":   {"mean": 1.19, "sd": 0.10, "pooled": 0.96},
        "ESR1_ago": {"mean": 0.00, "sd": None, "pooled": 0.00},
        "ESR1_ant": {"mean": 0.91, "sd": 0.78, "pooled": 0.98},
        "FEN1":    {"mean": 4.34, "sd": None, "pooled": 4.34},
        "GBA":     {"mean": 5.72, "sd": 1.97, "pooled": 8.43},
        "IDH1":    {"mean": 0.73, "sd": 1.20, "pooled": 0.00},
        "KAT2A":   {"mean": 1.89, "sd": 0.59, "pooled": 2.58},
        "MAPK1":   {"mean": 1.32, "sd": 0.53, "pooled": 1.62},
        "MTORC1":  {"mean": 1.03, "sd": 0.65, "pooled": 1.03},
        "OPRK1":   {"mean": 4.17, "sd": None, "pooled": 4.17},
        "PKM2":    {"mean": 0.57, "sd": 0.37, "pooled": 0.18},
        "PPARG":   {"mean": 5.92, "sd": 5.20, "pooled": 7.41},
        "TP53":    {"mean": 0.00, "sd": None, "pooled": 0.00},
        "VDR":     {"mean": 0.34, "sd": 0.48, "pooled": 0.00},
    },
}


def fmt_mean_sd(mean_val, sd_val):
    if sd_val is None or pd.isna(sd_val):
        return f"{mean_val:.2f}"
    return f"{mean_val:.2f} $\\pm$ {sd_val:.2f}"


def parse_list_cell(x):
    if x is None or pd.isna(x):
        return []

    if isinstance(x, list):
        return [float(v) for v in x]

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    parsed = ast.literal_eval(s)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list-like cell, got: {x}")

    return [float(v) for v in parsed]


def compute_average(metric_dict):
    means = []
    sds = []
    pooleds = []

    for t in TARGETS:
        means.append(metric_dict[t]["mean"])
        pooleds.append(metric_dict[t]["pooled"])
        if metric_dict[t]["sd"] is not None:
            sds.append(metric_dict[t]["sd"])

    avg_mean = float(np.mean(means))
    avg_sd = float(np.mean(sds)) if len(sds) > 0 else None
    avg_pooled = float(np.mean(pooleds))

    return avg_mean, avg_sd, avg_pooled


def load_roshambo_combination_metric(
    xlsx_path,
    sheet_name="litpcba_combination_EF1_results",
    target_col="Target",
    per_ref_col="individual_EF1s",
    pooled_col="max_pooled_EF1",
):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    required_cols = {target_col, per_ref_col, pooled_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in sheet '{sheet_name}': {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    result = {}
    for target in TARGETS:
        sub = df[df[target_col].astype(str).str.strip() == target]
        if len(sub) == 0:
            raise ValueError(f"No row found for target={target} in {xlsx_path}, sheet={sheet_name}")
        if len(sub) > 1:
            raise ValueError(f"Multiple rows found for target={target} in {xlsx_path}, sheet={sheet_name}")

        row = sub.iloc[0]
        vals = parse_list_cell(row[per_ref_col])
        if len(vals) == 0:
            raise ValueError(f"No per-reference EF1 values for target={target} in {xlsx_path}, sheet={sheet_name}")

        mean_v = float(np.mean(vals))
        sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        pooled_v = float(row[pooled_col])

        result[target] = {
            "mean": mean_v,
            "sd": sd_v,
            "pooled": pooled_v,
        }

    return result


def load_dynamic_metric(
    per_ref_csv_path,
    pooled_csv_path,
    target_col="target",
    ef1_col="ef1",
    pooled_col="pooled_ef1",
    mode_col=None,
    mode_value=None,
):
    df_ref = pd.read_csv(per_ref_csv_path)
    df_pool = pd.read_csv(pooled_csv_path)

    if mode_col is not None and mode_value is not None:
        df_ref = df_ref[df_ref[mode_col].astype(str) == str(mode_value)].copy()
        df_pool = df_pool[df_pool[mode_col].astype(str) == str(mode_value)].copy()

    result = {}

    for target in TARGETS:
        sub_ref = df_ref[df_ref[target_col].astype(str) == target]
        vals = sub_ref[ef1_col].astype(float).to_numpy()

        if len(vals) == 0:
            raise ValueError(f"No per-reference rows found for target={target} in {per_ref_csv_path}")

        sub_pool = df_pool[df_pool[target_col].astype(str) == target]
        if len(sub_pool) == 0:
            raise ValueError(f"No pooled rows found for target={target} in {pooled_csv_path}")

        pooled_val = float(sub_pool.iloc[0][pooled_col])
        mean_v = float(np.mean(vals))
        sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else None

        result[target] = {
            "mean": mean_v,
            "sd": sd_v,
            "pooled": pooled_val,
        }

    return result


def metric_to_columns(metric_dict):
    cols = {}
    for t in TARGETS:
        cols[t] = {
            "mean_sd": fmt_mean_sd(metric_dict[t]["mean"], metric_dict[t]["sd"]),
            "pooled": f"{metric_dict[t]['pooled']:.2f}",
        }
    return cols


def build_latex_table(metric_order, metric_data, raw_metric_data, caption, label):
    header_1 = "& " + " & ".join(
        [f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in metric_order]
    ) + " \\\\"

    cmidrules = []
    start = 2
    for _ in metric_order:
        end = start + 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1

    header_2 = (
        "Target"
        + "".join([" & Mean $\\pm$ SD & Best-pooled" for _ in metric_order])
        + " \\\\"
    )

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{l" + " cc |" * len(metric_order) + "}")
    lines.append("\\toprule")
    lines.append(header_1)
    lines.append(" ".join(cmidrules))
    lines.append(header_2)
    lines.append("\\midrule")

    for t in TARGETS:
        row = [t.replace("_", "\\_")]
        for m in metric_order:
            row.append(metric_data[m][t]["mean_sd"])
            row.append(metric_data[m][t]["pooled"])
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\midrule")

    avg_row = ["Average"]
    for m in metric_order:
        avg_mean, avg_sd, avg_pooled = compute_average(raw_metric_data[m])
        avg_row.append(fmt_mean_sd(avg_mean, avg_sd))
        avg_row.append(f"{avg_pooled:.2f}")
    lines.append(" & ".join(avg_row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist_type",
        type=str,
        required=True,
        choices=["cosine", "euclidean"],
        help="Distance type for GeoDiff and MoLFormer."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="concat",
        choices=["3D", "2D", "concat"],
        help="Embedding mode for GeoDiff."
    )
    parser.add_argument(
        "--xlsx",
        type=str,
        default="./roshambo2.xlsx",
        help="Path to Roshambo2 Excel file."
    )
    parser.add_argument(
        "--roshambo_sheet",
        type=str,
        default="litpcba_combination_EF1_results",
        help="Sheet name for Roshambo2 combination results."
    )

    args = parser.parse_args()

    geodiff_per_ref_csv = (
        f"/mnt/depts/dept01/misd/molml/embeddings/figures_geodiff/"
        f"lit_pcba_geodiff_{args.dist_type}_per_reference.csv"
    )
    geodiff_pooled_csv = (
        f"/mnt/depts/dept01/misd/molml/embeddings/figures_geodiff/"
        f"lit_pcba_geodiff_{args.dist_type}_pooled.csv"
    )

    molformer_per_ref_csv = (
        f"/mnt/depts/dept01/misd/molml/embeddings/figures_molformer/"
        f"lit_pcba_molformer_{args.dist_type}_per_reference.csv"
    )
    molformer_pooled_csv = (
        f"/mnt/depts/dept01/misd/molml/embeddings/figures_molformer/"
        f"lit_pcba_molformer_{args.dist_type}_pooled.csv"
    )

    roshambo_comb_name = "ROSHAMBO2 score"
    geodiff_name = f"GeoDiff {args.dist_type} distance"
    molformer_name = f"MoLFormer {args.dist_type} distance"

    roshambo_comb_metric = load_roshambo_combination_metric(
        xlsx_path=args.xlsx,
        sheet_name=args.roshambo_sheet,
        target_col="Target",
        per_ref_col="individual_EF1s",
        pooled_col="max_pooled_EF1",
    )

    geodiff_metric = load_dynamic_metric(
        per_ref_csv_path=geodiff_per_ref_csv,
        pooled_csv_path=geodiff_pooled_csv,
        target_col="target",
        ef1_col="ef1",
        pooled_col="pooled_ef1",
        mode_col="mode",
        mode_value=args.mode,
    )

    molformer_metric = load_dynamic_metric(
        per_ref_csv_path=molformer_per_ref_csv,
        pooled_csv_path=molformer_pooled_csv,
        target_col="target",
        ef1_col="ef1",
        pooled_col="pooled_ef1",
        mode_col="mode",
        mode_value="molformer",
    )

    metric_data = {
        "2D ECFP4 similarity": metric_to_columns(FIXED_METRICS["2D ECFP4 similarity"]),
        "ROCS 3D similarity": metric_to_columns(FIXED_METRICS["ROCS 3D similarity"]),
        "Molecular docking": metric_to_columns(FIXED_METRICS["Molecular docking"]),
        roshambo_comb_name: metric_to_columns(roshambo_comb_metric),
        geodiff_name: metric_to_columns(geodiff_metric),
        molformer_name: metric_to_columns(molformer_metric),
    }

    raw_metric_data = {
        "2D ECFP4 similarity": FIXED_METRICS["2D ECFP4 similarity"],
        "ROCS 3D similarity": FIXED_METRICS["ROCS 3D similarity"],
        "Molecular docking": FIXED_METRICS["Molecular docking"],
        roshambo_comb_name: roshambo_comb_metric,
        geodiff_name: geodiff_metric,
        molformer_name: molformer_metric,
    }

    metric_order = [
        "2D ECFP4 similarity",
        "ROCS 3D similarity",
        "Molecular docking",
        roshambo_comb_name,
        geodiff_name,
        molformer_name,
    ]

    latex = build_latex_table(
        metric_order=metric_order,
        metric_data=metric_data,
        raw_metric_data=raw_metric_data,
        caption=(
            f"Virtual screening performance (EF1\\%) obtained by different methods. "
            f"Each method reports per-reference Mean $\\pm$ SD and Best-pooled EF1\\%. "
            f"ROSHAMBO2 score is based on the combination mode. "
            f"GeoDiff uses {args.mode} mode with {args.dist_type} distance, and "
            f"MoLFormer uses {args.dist_type} distance."
        ),
        label=f"tab:vs_comparison_mean_pooled_roshambo2_{args.mode}_{args.dist_type}",
    )

    print("\n===== Averages over selected 15 targets =====")
    for name in metric_order:
        avg_mean, avg_sd, avg_pooled = compute_average(raw_metric_data[name])
        if avg_sd is None:
            print(f"{name}: Mean={avg_mean:.2f}, SD=NA, Best-pooled={avg_pooled:.2f}")
        else:
            print(f"{name}: Mean±SD={avg_mean:.2f}±{avg_sd:.2f}, Best-pooled={avg_pooled:.2f}")

    print("\n===== LaTeX table =====\n")
    print(latex)


if __name__ == "__main__":
    main()