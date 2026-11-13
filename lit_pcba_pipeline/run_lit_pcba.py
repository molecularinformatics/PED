"""End-to-end LIT-PCBA pipeline: prepare, embed, distances, EF1, stability.

Runs one target, several, or all 15. Completed stages are stamped under the output
folder and skipped on a rerun unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
ANALYSIS = os.path.join(HERE, "analysis")
REPO_ROOT = os.path.dirname(HERE)

sys.path.insert(0, SCRIPTS)
from lit_pcba_utils import DEFAULT_REFS_CSV  # noqa: E402

TARGETS = ["ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A",
           "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]
STAGES = ["prepare", "embed", "distance", "ef1", "stability"]


def run(cmd: Sequence[str], cwd: Optional[str] = None) -> None:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def stamp_path(out_root: str, target: str, stage: str) -> str:
    return os.path.join(out_root, target, ".stamps", f"{stage}.json")


def done(out_root: str, target: str, stage: str) -> bool:
    return os.path.exists(stamp_path(out_root, target, stage))


def mark(out_root: str, target: str, stage: str, info: dict) -> None:
    path = stamp_path(out_root, target, stage)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)


def target_stages(args, target: str) -> None:
    tdir = os.path.join(args.lit_pcba_root, target)
    out = os.path.join(args.out_root, target)
    prepared = os.path.join(out, "prepared")
    py = args.python

    if "prepare" in args.stages and (args.force or not done(args.out_root, target, "prepare")):
        run([py, os.path.join(SCRIPTS, "prepare_lit_pcba_inputs.py"),
             "--target_dir", tdir, "--out_dir", prepared,
             "--refs_csv", args.refs_csv, "--target", target])
        mark(args.out_root, target, "prepare", {"refs_csv": args.refs_csv})

    sides = (("refs", "refs.csv"), ("library", "library.csv"))

    for model in args.models:
        if "embed" in args.stages and (args.force or not done(args.out_root, target, f"embed_{model}")):
            for which, csv in sides:
                emb_dir = os.path.join(out, which, "embeddings")
                if model == "molformer":
                    run([py, os.path.join(SCRIPTS, "get_molformer_embedding.py"),
                         "--csv_path", os.path.join(prepared, csv),
                         "--cache_dir", os.path.join(emb_dir, "molformer"),
                         "--smiles_col", "smiles",
                         "--model_name", args.molformer_model,
                         "--batch_size", args.batch_size])
                else:
                    packed = os.path.join(out, which, "packed")
                    run([py, os.path.join(SCRIPTS, "pack_geodiff_data.py"),
                         "--csv", os.path.join(prepared, csv), "--smiles_col", "smiles",
                         "--out_dir", packed, "--ref_confs", args.ref_confs,
                         "--seed", args.seed, "--n_workers", args.n_workers])
                    run([py, os.path.join(SCRIPTS, "get_geodiff_embedding.py"),
                         "--ckpt", os.path.abspath(args.geodiff_ckpt),
                         "--packed_dir", packed, "--out_dir", emb_dir,
                         "--batch_size", args.batch_size, "--seed", args.seed,
                         "--device", args.device])
            mark(args.out_root, target, f"embed_{model}", {"model": model})

        if "distance" in args.stages and (args.force or not done(args.out_root, target, f"distance_{model}")):
            sub = ["molformer"] if model == "molformer" else []
            cmd = [py, os.path.join(SCRIPTS, "compute_pcba_distances.py"),
                   "--embedding_backend", model,
                   "--refs_csv", os.path.join(prepared, "refs.csv"),
                   "--library_csv", os.path.join(prepared, "library.csv"),
                   "--refs_emb_root", os.path.join(out, "refs", "embeddings", *sub),
                   "--library_emb_root", os.path.join(out, "library", "embeddings", *sub),
                   "--out_csv", os.path.join(out, "distance",
                                             f"library_reference_{model}_distances.csv"),
                   "--distance_type", *args.distance_type]
            if model == "geodiff":
                cmd += ["--modes", *args.modes]
            run(cmd)
            mark(args.out_root, target, f"distance_{model}", {"model": model})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lit_pcba_root", required=True, help="Folder holding one subfolder per target")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--targets", nargs="+", default=["all"],
                    help="Target names, or 'all' for the 15 benchmark targets")
    ap.add_argument("--models", nargs="+", default=["molformer", "geodiff"],
                    choices=["molformer", "geodiff"])
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--distance_type", nargs="+", default=["cosine", "euclidean"],
                    choices=["cosine", "euclidean"])
    ap.add_argument("--modes", nargs="+", default=["2D", "3D", "concat"],
                    choices=["2D", "3D", "concat"], help="GeoDiff embedding modes")
    ap.add_argument("--refs_csv", default=DEFAULT_REFS_CSV)
    ap.add_argument("--geodiff_ckpt",
                    default=os.path.join(REPO_ROOT, "external", "checkpoints", "geodiff",
                                         "model", "checkpoints", "drugs_default.pt"))
    ap.add_argument("--molformer_model", default="ibm/MoLFormer-XL-both-10pct")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--ref_confs", type=int, default=1)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Rerun stages even if stamped done")
    args = ap.parse_args()

    targets: List[str] = TARGETS if args.targets == ["all"] else args.targets
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown targets: {unknown}")

    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {target}", flush=True)
        target_stages(args, target)

    results_dir = os.path.join(args.out_root, "ef1")
    if "ef1" in args.stages:
        for model in args.models:
            run([args.python, os.path.join(ANALYSIS, "ef1.py"),
                 "--out_root", args.out_root, "--backend", model,
                 "--distance_type", *args.distance_type,
                 "--results_dir", results_dir])

    if "stability" in args.stages:
        for distance_type in args.distance_type:
            run([args.python, os.path.join(ANALYSIS, "stability.py"),
                 "--results_dir", results_dir, "--distance_type", distance_type])


if __name__ == "__main__":
    main()
