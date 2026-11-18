import os
import sys
import math
import gzip
import pickle
import urllib.request
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from paths import RESULTS_ROOT, REWARDS
from pathlib import Path

# =========================
# user settings
# =========================
# Default model; override with `python table_top_5000.py <reinvent|libinvent|synformer>`
MODEL_KEY = "libinvent"


def file_map(model_key):
    return {label: f"{model_key}_{suffix}.csv" for label, suffix in REWARDS.items()}


BASE_DIR = str(RESULTS_ROOT / MODEL_KEY / "scaffold_memory")
FILE_MAP = file_map(MODEL_KEY)

TOP_N = 5000

# =========================
# SA score helper
# adapted from RDKit sascorer logic
# =========================
SA_FSCORE_PATH = str(Path(__file__).resolve().parent / "fpscores.pkl.gz")


def download_sa_model_if_needed():
    if os.path.exists(SA_FSCORE_PATH):
        return

    # official RDKit Contrib source mirror commonly used
    url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"
    urllib.request.urlretrieve(url, SA_FSCORE_PATH)


def read_fragment_scores():
    download_sa_model_if_needed()
    with gzip.open(SA_FSCORE_PATH, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    out = {}
    for entry in data:
        score = float(entry[0])
        for bit_id in entry[1:]:
            out[bit_id] = score
    return out


SA_FSCORES = None


def num_bridgeheads_and_spiro(mol):
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    return n_bridge, n_spiro


def calculate_sa_score(mol):
    """
    Returns synthetic accessibility score in the usual ~[1,10] range
    where lower is easier to synthesize.
    """
    global SA_FSCORES
    if SA_FSCORES is None:
        SA_FSCORES = read_fragment_scores()

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()

    # fragment score
    fragment_score = 0.0
    total_count = 0
    for bit_id, count in fps.items():
        total_count += count
        fragment_score += SA_FSCORES.get(bit_id, -4.0) * count
    fragment_score /= max(total_count, 1)

    # feature penalties
    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_spiro, n_bridgeheads = num_bridgeheads_and_spiro(mol)
    n_macrocycles = sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) > 8)

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = 0.0 - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty

    # fingerprint density correction
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5
    else:
        score3 = 0.0

    sascore = fragment_score + score2 + score3

    # transform to 1..10
    min_val = -4.0
    max_val = 2.5
    sascore = 11.0 - (sascore - min_val + 1.0) / (max_val - min_val) * 9.0

    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


# =========================
# descriptor calculation
# =========================
def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = rdMolDescriptors.CalcExactMolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    qed = QED.qed(mol)
    sa = calculate_sa_score(mol)

    return {
        "MW": mw,
        "TPSA": tpsa,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "QED": qed,
        "SA": sa,
    }


def safe_ratio(count, denom):
    return 100.0 * count / denom if denom > 0 else float("nan")


def summarize_top5000(df, method_name):
    required_cols = ["SMILES", "total_score", "ScaffoldRingsLinkersSMILES"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{method_name} missing columns: {missing}")

    top_df = df.sort_values("total_score", ascending=False).head(TOP_N).copy()

    unique_scaffold_count = top_df["ScaffoldRingsLinkersSMILES"].dropna().astype(str).nunique()
    unique_scaffold_ratio = unique_scaffold_count / TOP_N

    desc_rows = []
    invalid_smiles = 0
    for smi in top_df["SMILES"].astype(str):
        d = calc_descriptors(smi)
        if d is None:
            invalid_smiles += 1
            continue
        desc_rows.append(d)

    desc_df = pd.DataFrame(desc_rows)
    n_valid = len(desc_df)

    if n_valid == 0:
        raise ValueError(f"{method_name} has no valid SMILES in top {TOP_N}.")

    summary = {
        "Method": method_name,
        "N_valid": n_valid,
        "Invalid_SMILES": invalid_smiles,
        "UniqueScaffold": unique_scaffold_count,
        "UniqueScaffoldRatio": unique_scaffold_ratio,
        "MW_min": desc_df["MW"].min(),
        "MW_max": desc_df["MW"].max(),
        "TPSA_min": desc_df["TPSA"].min(),
        "TPSA_max": desc_df["TPSA"].max(),
        "LogP_min": desc_df["LogP"].min(),
        "LogP_max": desc_df["LogP"].max(),
        "HBD_min": desc_df["HBD"].min(),
        "HBD_max": desc_df["HBD"].max(),
        "HBA_min": desc_df["HBA"].min(),
        "HBA_max": desc_df["HBA"].max(),
        "MW_pct": safe_ratio(((desc_df["MW"] >= 200) & (desc_df["MW"] <= 500)).sum(), n_valid),
        "TPSA_pct": safe_ratio(((desc_df["TPSA"] >= 20) & (desc_df["TPSA"] <= 130)).sum(), n_valid),
        "LogP_pct": safe_ratio(((desc_df["LogP"] >= -1) & (desc_df["LogP"] <= 6)).sum(), n_valid),
        "HBD_pct": safe_ratio(((desc_df["HBD"] >= 0) & (desc_df["HBD"] <= 5)).sum(), n_valid),
        "HBA_pct": safe_ratio(((desc_df["HBA"] >= 0) & (desc_df["HBA"] <= 10)).sum(), n_valid),
        "QED_pct": safe_ratio(((desc_df["QED"] >= 0.4) & (desc_df["QED"] <= 1.0)).sum(), n_valid),
        "SA_pct": safe_ratio(((desc_df["SA"] >= 1) & (desc_df["SA"] <= 5)).sum(), n_valid),
    }

    return summary


def fmt_num(x):
    return f"{x:.2f}"


def fmt_pct(x):
    return f"{x:.2f}\\%"


def build_table3_latex(summary_df):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Physical-chemical properties of top-5000 generated compounds.}")
    lines.append(r"\label{tab:physchem_top5000}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lcccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & MW$_{min}$ & MW$_{max}$ & TPSA$_{min}$ & TPSA$_{max}$ & LogP$_{min}$ & LogP$_{max}$ & HBD$_{min}$ & HBD$_{max}$ & HBA$_{min}$ & HBA$_{max}$ \\"
    )
    lines.append(r"\midrule")

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['Method']} & "
            f"{fmt_num(row['MW_min'])} & {fmt_num(row['MW_max'])} & "
            f"{fmt_num(row['TPSA_min'])} & {fmt_num(row['TPSA_max'])} & "
            f"{fmt_num(row['LogP_min'])} & {fmt_num(row['LogP_max'])} & "
            f"{fmt_num(row['HBD_min'])} & {fmt_num(row['HBD_max'])} & "
            f"{fmt_num(row['HBA_min'])} & {fmt_num(row['HBA_max'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_table4_latex(summary_df):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Percentages of top-5000 generated compounds in good drug-likeness range.}")
    lines.append(r"\label{tab:druglikeness_percent}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & MW ([200, 500]) & TPSA ([20, 130]) & LogP ([$-1$, 6]) & HBD ([0, 5]) & HBA ([0, 10]) & QED ([0.4, 1]) & SA ([1, 5]) & Unique scaffold ratio \\"
    )
    lines.append(r"\midrule")

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['Method']} & "
            f"{fmt_pct(row['MW_pct'])} & "
            f"{fmt_pct(row['TPSA_pct'])} & "
            f"{fmt_pct(row['LogP_pct'])} & "
            f"{fmt_pct(row['HBD_pct'])} & "
            f"{fmt_pct(row['HBA_pct'])} & "
            f"{fmt_pct(row['QED_pct'])} & "
            f"{fmt_pct(row['SA_pct'])} & "
            f"{fmt_num(row['UniqueScaffoldRatio'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    global BASE_DIR, FILE_MAP

    # optional: `python table_top_5000.py <reinvent|libinvent|synformer>`
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        if model_key not in ("reinvent", "libinvent", "synformer"):
            raise SystemExit(f"unknown model: {model_key}")
        BASE_DIR = str(RESULTS_ROOT / model_key / "scaffold_memory")
        FILE_MAP = file_map(model_key)
        print(f"[config] model={model_key}  base_dir={BASE_DIR}")

    all_summaries = []

    for method, filename in FILE_MAP.items():
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            print(f"[skip] {method}: file not found: {path}")
            continue

        df = pd.read_csv(path)
        if "smiles" in df.columns:
            df.rename(columns={"smiles": "SMILES"}, inplace=True)

        df.drop_duplicates(subset=["SMILES"], inplace=True)

        summary = summarize_top5000(df, method)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    # keep desired row order
    summary_df["Method"] = pd.Categorical(
        summary_df["Method"],
        categories=["ROSHAMBO2", "ECFP4", "GeoDiff 2D", "GeoDiff 3D", "GeoDiff Concat", "MoLFormer"],
        ordered=True,
    )
    summary_df = summary_df.sort_values("Method").reset_index(drop=True)

    table3_latex = build_table3_latex(summary_df)
    table4_latex = build_table4_latex(summary_df)

    print("=" * 80)
    print("Table 3 LaTeX")
    print("=" * 80)
    print(table3_latex)
    print()
    print("=" * 80)
    print("Table 4 LaTeX")
    print("=" * 80)
    print(table4_latex)
    print()
    print("=" * 80)
    print("Debug summary")
    print("=" * 80)
    print(summary_df[["Method", "N_valid", "Invalid_SMILES", "UniqueScaffold", "UniqueScaffoldRatio"]])


if __name__ == "__main__":
    main()
