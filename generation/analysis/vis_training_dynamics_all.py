from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from paths import RESULTS_ROOT, OUT_DIR

# =========================
# File paths
# =========================
DATA_ROOT = RESULTS_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# All published runs were consolidated under PED_results/ (2026-11-11), including
# the two ECFP4 runs, moved there from their submission directory on 2026-11-12.
RUN_ROOT = DATA_ROOT / "PED_results"

RUN_DIRS = {
    "REINVENT": {
        "ROSHAMBO2": RUN_ROOT / "reinvent/bmsbtk_roshambo2/res/progress.log",
        "ECFP4": RUN_ROOT / "reinvent/bmsbtk_ecfp4/res/progress.log",
        "GeoDiff 2D": RUN_ROOT / "reinvent/bmsbtk_geodiff_2D/res/progress.log",
        "GeoDiff 3D": RUN_ROOT / "reinvent/bmsbtk_geodiff_3D/res/progress.log",
        "GeoDiff Concat": RUN_ROOT / "reinvent/bmsbtk_geodiff_concat_norm/res/progress.log",
        "MoLFormer": RUN_ROOT / "reinvent/bmsbtk_molformer/res/progress.log",
    },
    "LibINVENT": {
        "ROSHAMBO2": RUN_ROOT / "libinvent/bmsbtk_roshambo2_libinvent/res/progress.log",
        "ECFP4": RUN_ROOT / "libinvent/bmsbtk_ecfp4_libinvent/res/progress.log",
        "GeoDiff 2D": RUN_ROOT / "libinvent/bmsbtk_geodiff_2D_libinvent/res/progress.log",
        "GeoDiff 3D": RUN_ROOT / "libinvent/bmsbtk_geodiff_3D_libinvent/res/progress.log",
        "GeoDiff Concat": RUN_ROOT / "libinvent/bmsbtk_geodiff_concat_norm_libinvent/res/progress.log",
        "MoLFormer": RUN_ROOT / "libinvent/bmsbtk_molformer_libinvent/res/progress.log",
    },
}

MODEL_TITLES = ["REINVENT", "LibINVENT", "SynFormer"]

FILE_SUFFIXES = {
    "ROSHAMBO2": "roshambo2",
    "ECFP4": "ecfp4",
    "GeoDiff 2D": "geodiff_2d",
    "GeoDiff 3D": "geodiff_3d",
    "GeoDiff Concat": "geodiff_concat",
    "MoLFormer": "molformer",
}


def get_synformer_file_path(method_suffix):
    return RUN_ROOT / "synformer" / "scaffold_memory" / f"synformer_{method_suffix}.csv"


# =========================
# Colors
# =========================
method_colors = {
    "ROSHAMBO2": "#15751a",
    "ECFP4": "#949494",
    "GeoDiff 3D": "#8ac0f2",
    "GeoDiff 2D": "#438cd0",
    "GeoDiff Concat": "#235787",
    "MoLFormer": "#d15353",
}


# =========================
# Plot settings
# =========================
fontsize = 13
figsize = (15, 4.8)
linewidth = 2.0
alpha_band = 0.18
smooth_window = 5


def get_step_col(df, file_path):
    for col in ("Step", "step"):
        if col in df.columns:
            return col

    raise ValueError(f"{file_path} is missing required step column: Step or step")


def find_event_file(run_dir):
    run_path = Path(run_dir)
    event_files = sorted(run_path.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")

    # A crashed attempt can leave a near-empty event file next to the real one
    # (the ECFP4 REINVENT directory has one); take the largest, not the last.
    return max(event_files, key=lambda f: f.stat().st_size)


def load_event_step_stats(run_dir):
    event_file = find_event_file(run_dir)
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    tag = "Average score"
    available_tags = ea.Tags().get("scalars", [])
    if tag not in available_tags:
        raise ValueError(f"{event_file} is missing TensorBoard scalar tag: {tag}")

    events = ea.Scalars(tag)
    step_stats = pd.DataFrame(
        {
            "step": [event.step for event in events],
            "mean": [event.value for event in events],
            "std": 0.0,
        }
    )

    return step_stats.sort_values("step")


def load_csv_step_stats(file_path):
    df = pd.read_csv(file_path)

    step_col = get_step_col(df, file_path)

    required_cols = ["SMILES", "total_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path} is missing required columns: {missing_cols}")

    df = df[[step_col, "SMILES", "total_score"]].copy()
    df = df.rename(columns={step_col: "step"})
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
    df = df.dropna(subset=["step", "SMILES", "total_score"])

    step_stats = (
        df.groupby("step")["total_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("step")
    )
    step_stats["std"] = step_stats["std"].fillna(0.0)

    return step_stats


def moving_average(values, window):
    if window <= 1:
        return values

    return pd.Series(values).rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean().values


fig, axes = plt.subplots(
    1,
    len(MODEL_TITLES),
    figsize=figsize,
    sharex=False,
    sharey=True,
)

legend_handles = []
legend_labels = []

for ax, model_title in zip(axes, MODEL_TITLES):
    for method, suffix in FILE_SUFFIXES.items():
        if model_title == "SynFormer":
            file_path = get_synformer_file_path(suffix)
            if not file_path.exists():
                print(f"[{model_title}] {method}: SKIPPED (missing {file_path})")
                continue
            step_stats = load_csv_step_stats(file_path)
        else:
            run_dir = Path(RUN_DIRS[model_title][method])
            if not run_dir.is_dir():
                print(f"[{model_title}] {method}: SKIPPED (missing {run_dir})")
                continue
            step_stats = load_event_step_stats(run_dir)

        x = step_stats["step"].values
        if smooth_window > 1:
            y = moving_average(step_stats["mean"].values, smooth_window)
            y_std = moving_average(step_stats["std"].values, smooth_window)
        else:
            y = step_stats["mean"].values
            y_std = step_stats["std"].values

        color = method_colors[method]

        (line,) = ax.plot(
            x,
            y,
            label=method,
            color=color,
            linewidth=linewidth,
        )

        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=color,
            alpha=alpha_band,
            linewidth=0,
        )

        if method not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(method)

    ax.set_title(model_title, fontsize=fontsize)
    ax.set_xlabel("Step", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

axes[0].set_ylabel("Average Score", fontsize=fontsize)

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=len(legend_labels),
    fontsize=fontsize,
    frameon=True,
    bbox_to_anchor=(0.5, 1.02),
)

plt.tight_layout(rect=(0, 0, 1, 0.92))

out_path = OUT_DIR / "training_dynamics_all.pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
