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


def load_roshambo_metric(
    xlsx_path,
    sheet_name,
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


def load_geodiff_metric(
    per_ref_csv_path,
    pooled_csv_path,
    mode_value,
    target_col="target",
    ef1_col="ef1",
    pooled_col="pooled_ef1",
    mode_col="mode",
):
    df_ref = pd.read_csv(per_ref_csv_path)
    df_pool = pd.read_csv(pooled_csv_path)

    df_ref = df_ref[df_ref[mode_col].astype(str) == str(mode_value)].copy()
    df_pool = df_pool[df_pool[mode_col].astype(str) == str(mode_value)].copy()

    result = {}

    for target in TARGETS:
        sub_ref = df_ref[df_ref[target_col].astype(str) == target]
        vals = sub_ref[ef1_col].astype(float).to_numpy()

        if len(vals) == 0:
            raise ValueError(f"No per-reference rows found for target={target}, mode={mode_value} in {per_ref_csv_path}")

        sub_pool = df_pool[df_pool[target_col].astype(str) == target]
        if len(sub_pool) == 0:
            raise ValueError(f"No pooled rows found for target={target}, mode={mode_value} in {pooled_csv_path}")

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
        help="Distance type for GeoDiff."
    )
    parser.add_argument(
        "--xlsx",
        type=str,
        default="./roshambo2.xlsx",
        help="Path to Roshambo2 Excel file."
    )
    parser.add_argument(
        "--sheet_color",
        type=str,
        default="litpcba_color_EF1_results",
        help="Sheet name for ROSHAMBO2 color results."
    )
    parser.add_argument(
        "--sheet_shape",
        type=str,
        default="litpcba_shape_EF1_results",
        help="Sheet name for ROSHAMBO2 shape results."
    )
    parser.add_argument(
        "--sheet_combination",
        type=str,
        default="litpcba_combination_EF1_results",
        help="Sheet name for ROSHAMBO2 combination results."
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

    roshambo_color_name = "ROSHAMBO2 color"
    roshambo_shape_name = "ROSHAMBO2 shape"
    roshambo_comb_name = "ROSHAMBO2 combination"
    geodiff_2d_name = "GeoDiff 2D"
    geodiff_3d_name = "GeoDiff 3D"
    geodiff_concat_name = "GeoDiff concat"

    roshambo_color_metric = load_roshambo_metric(
        xlsx_path=args.xlsx,
        sheet_name=args.sheet_color,
        target_col="Target",
        per_ref_col="individual_EF1s",
        pooled_col="max_pooled_EF1",
    )

    roshambo_shape_metric = load_roshambo_metric(
        xlsx_path=args.xlsx,
        sheet_name=args.sheet_shape,
        target_col="Target",
        per_ref_col="individual_EF1s",
        pooled_col="max_pooled_EF1",
    )

    roshambo_comb_metric = load_roshambo_metric(
        xlsx_path=args.xlsx,
        sheet_name=args.sheet_combination,
        target_col="Target",
        per_ref_col="individual_EF1s",
        pooled_col="max_pooled_EF1",
    )

    geodiff_2d_metric = load_geodiff_metric(
        per_ref_csv_path=geodiff_per_ref_csv,
        pooled_csv_path=geodiff_pooled_csv,
        mode_value="2D",
        target_col="target",
        ef1_col="ef1",
        pooled_col="pooled_ef1",
        mode_col="mode",
    )

    geodiff_3d_metric = load_geodiff_metric(
        per_ref_csv_path=geodiff_per_ref_csv,
        pooled_csv_path=geodiff_pooled_csv,
        mode_value="3D",
        target_col="target",
        ef1_col="ef1",
        pooled_col="pooled_ef1",
        mode_col="mode",
    )

    geodiff_concat_metric = load_geodiff_metric(
        per_ref_csv_path=geodiff_per_ref_csv,
        pooled_csv_path=geodiff_pooled_csv,
        mode_value="concat",
        target_col="target",
        ef1_col="ef1",
        pooled_col="pooled_ef1",
        mode_col="mode",
    )

    metric_data = {
        roshambo_color_name: metric_to_columns(roshambo_color_metric),
        roshambo_shape_name: metric_to_columns(roshambo_shape_metric),
        roshambo_comb_name: metric_to_columns(roshambo_comb_metric),
        geodiff_2d_name: metric_to_columns(geodiff_2d_metric),
        geodiff_3d_name: metric_to_columns(geodiff_3d_metric),
        geodiff_concat_name: metric_to_columns(geodiff_concat_metric),
    }

    raw_metric_data = {
        roshambo_color_name: roshambo_color_metric,
        roshambo_shape_name: roshambo_shape_metric,
        roshambo_comb_name: roshambo_comb_metric,
        geodiff_2d_name: geodiff_2d_metric,
        geodiff_3d_name: geodiff_3d_metric,
        geodiff_concat_name: geodiff_concat_metric,
    }

    metric_order = [
        roshambo_color_name,
        roshambo_shape_name,
        roshambo_comb_name,
        geodiff_2d_name,
        geodiff_3d_name,
        geodiff_concat_name,
    ]

    latex = build_latex_table(
        metric_order=metric_order,
        metric_data=metric_data,
        raw_metric_data=raw_metric_data,
        caption=(
            f"Complete virtual screening performance (EF1\\%) obtained by different "
            f"modes of ROSHAMBO2 and GeoDiff. GeoDiff uses {args.dist_type} distance."
        ),
        label=f"tab:roshambo2_geodiff_complete_{args.dist_type}",
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