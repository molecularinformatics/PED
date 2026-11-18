"""Dump the 'Average score' curve of every REINVENT/LibINVENT run to one CSV.

vis_training_dynamics_all.py reads the same TensorBoard logs but keeps nothing on
disk, which makes any number quoted in the paper impossible to check. This writes
training_curves_avg_score.csv so the text can cite it. The paths and the
largest-event-file rule are kept identical to that script.
"""
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from paths import RESULTS_ROOT

RUN_ROOT = RESULTS_ROOT

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


def find_event_file(run_dir):
    event_files = sorted(Path(run_dir).glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    return max(event_files, key=lambda f: f.stat().st_size)


def load_event_step_stats(run_dir):
    ea = event_accumulator.EventAccumulator(
        str(find_event_file(run_dir)),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    events = ea.Scalars("Average score")
    return pd.DataFrame(
        {"step": [e.step for e in events], "mean": [e.value for e in events]}
    ).sort_values("step")


if __name__ == "__main__":
    rows = []
    for model, methods in RUN_DIRS.items():
        for method, run_dir in methods.items():
            try:
                df = load_event_step_stats(run_dir)
            except FileNotFoundError as exc:
                print(f"[skip] {model}/{method}: {exc}", flush=True)
                continue
            df["Model"] = model
            df["Method"] = method
            rows.append(df)
            print(f"[ok] {model}/{method}: {len(df)} steps, "
                  f"final={df['mean'].iloc[-1]:.4f}", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(str(Path(__file__).resolve().parent / "training_curves_avg_score.csv"), index=False)
    print("saved training_curves_avg_score.csv", out.shape)
