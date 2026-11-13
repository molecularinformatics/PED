"""End-to-end embedding pipeline: prepare, embed, distances, merge.

Runs one pretrained model, several, or all five. Completed stages are stamped under
the output folder and skipped on a rerun unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
REPO_ROOT = os.path.dirname(HERE)

MODELS = ["geodiff", "molformer", "sde", "unimol", "ecfp"]
STAGES = ["prepare", "embed", "distance", "merge"]

# Uni-Mol2 needs its own environment: unimol_tools conflicts with the PyG stack the
# other models use.
NEEDS_OWN_ENV = {"unimol"}


def run(cmd: Sequence[str], env: Optional[Dict[str, str]] = None) -> None:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True,
                   env={**os.environ, **(env or {})})


def stamp(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, ".stamps", f"{name}.json")


def done(out_dir: str, name: str) -> bool:
    return os.path.exists(stamp(out_dir, name))


def mark(out_dir: str, name: str, info: dict) -> None:
    path = stamp(out_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)


def python_for(args, model: str) -> str:
    return args.python_unimol if model in NEEDS_OWN_ENV and args.python_unimol else args.python


def embed(args, model: str, out_dir: str, prepared: Dict[str, str]) -> None:
    py = python_for(args, model)
    for side, csv in prepared.items():
        emb = os.path.join(out_dir, side, "embeddings")
        if model == "molformer":
            run([py, os.path.join(SCRIPTS, "get_molformer_embedding.py"),
                 "--csv_path", csv, "--cache_dir", os.path.join(emb, "molformer"),
                 "--smiles_col", "smiles", "--model_name", args.molformer_model,
                 "--batch_size", args.batch_size])
        elif model == "ecfp":
            run([py, os.path.join(SCRIPTS, "get_ecfp.py"),
                 "--input_csv", csv, "--output_dir", os.path.join(emb, "ecfp"),
                 "--smiles_col", "smiles", "--radius", args.ecfp_radius,
                 "--nbits", args.ecfp_nbits, "--n_workers", args.n_workers])
        elif model == "unimol":
            run([py, os.path.join(SCRIPTS, "get_unimol_embedding.py"),
                 "--input_csv", csv, "--output_dir", os.path.join(emb, "unimol"),
                 "--smiles_col", "smiles", "--batch_size", args.batch_size,
                 "--model_name", "unimolv2", "--model_size", args.unimol_size,
                 "--use_cuda", "--use_gpu", args.gpu])
        elif model == "geodiff":
            packed = os.path.join(out_dir, side, "geodiff_packed")
            run([py, os.path.join(SCRIPTS, "pack_geodiff_data.py"),
                 "--csv", csv, "--smiles_col", "smiles", "--out_dir", packed,
                 "--ref_confs", args.ref_confs, "--seed", args.seed,
                 "--n_workers", args.n_workers])
            run([py, os.path.join(SCRIPTS, "get_geodiff_embedding.py"),
                 "--ckpt", os.path.abspath(args.geodiff_ckpt), "--packed_dir", packed,
                 "--out_dir", emb, "--batch_size", args.batch_size,
                 "--seed", args.seed, "--device", args.device])
        elif model == "sde":
            packed = os.path.join(out_dir, side, "sde_prepared")
            run([py, os.path.join(SCRIPTS, "pack_sde_data.py"),
                 "--csv", csv, "--smiles_col", "smiles", "--out_dir", packed,
                 "--num_confs", args.ref_confs, "--seed", args.seed,
                 "--n_workers", args.n_workers])
            run([py, os.path.join(SCRIPTS, "get_sde_embedding.py"),
                 "--ckpt", os.path.abspath(args.sde_ckpt), "--prepared_dir", packed,
                 "--out_dir", emb, "--batch_size", args.batch_size,
                 "--device", args.device])


def distance(args, model: str, out_dir: str, prepared: Dict[str, str]) -> None:
    sub = {"geodiff": "", "sde": ""}.get(model, model)
    emb_root = {side: os.path.join(out_dir, side, "embeddings", sub) if sub
                else os.path.join(out_dir, side, "embeddings")
                for side in prepared}
    cmd = [python_for(args, model), os.path.join(SCRIPTS, "compute_smiles_reference_distances.py"),
           "--refs_csv", prepared["refs"], "--library_csv", prepared["library"],
           "--refs_emb_root", emb_root["refs"], "--library_emb_root", emb_root["library"],
           "--out_csv", os.path.join(out_dir, "distance",
                                     f"smiles_reference_{model}_distances.csv"),
           "--model", model, "--distance_type", *args.distance_type]
    cmd += ["--modes", *args.modes] if model in ("geodiff", "sde") else ["--modes", "single"]
    run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library_csv", required=True, help="CSV holding the library SMILES")
    ap.add_argument("--reference_smiles", required=True, help="Single reference SMILES")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--job_name", default=None, help="Output subfolder; default: library stem")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--models", nargs="+", default=["all"], choices=MODELS + ["all"])
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--distance_type", nargs="+", default=["cosine", "euclidean"],
                    choices=["cosine", "euclidean"])
    ap.add_argument("--modes", nargs="+", default=["2D", "3D", "concat", "concat_norm"],
                    help="Dual-encoder modes for GeoDiff and MoleculeSDE")
    ap.add_argument("--geodiff_ckpt",
                    default=os.path.join(REPO_ROOT, "external", "geodiff", "checkpoints",
                                         "drugs_default.pt"))
    ap.add_argument("--sde_ckpt",
                    default=os.path.join(REPO_ROOT, "external", "moleculesde", "checkpoints",
                                         "vp.pth"))
    ap.add_argument("--molformer_model", default="ibm/MoLFormer-XL-both-10pct")
    ap.add_argument("--unimol_size", default="164m")
    ap.add_argument("--ecfp_radius", type=int, default=2)
    ap.add_argument("--ecfp_nbits", type=int, default=2048)
    ap.add_argument("--base_csv", default=None,
                    help="Reference similarity table for the merge stage")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--python_unimol", default=None,
                    help="Interpreter for the Uni-Mol2 environment")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--ref_confs", type=int, default=1)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models: List[str] = MODELS if "all" in args.models else args.models
    job = args.job_name or os.path.splitext(os.path.basename(args.library_csv))[0]
    out_dir = os.path.join(os.path.abspath(args.out_root), job)
    prep_dir = os.path.join(out_dir, "prepared")
    prepared = {"refs": os.path.join(prep_dir, "refs.csv"),
                "library": os.path.join(prep_dir, "library.csv")}

    if "prepare" in args.stages and (args.force or not done(out_dir, "prepare")):
        run([args.python, os.path.join(SCRIPTS, "prepare_smiles_reference_inputs.py"),
             "--library_csv", os.path.abspath(args.library_csv),
             "--reference_smiles", args.reference_smiles,
             "--smiles_col", args.smiles_col, "--out_dir", prep_dir])
        mark(out_dir, "prepare", {"library_csv": os.path.abspath(args.library_csv)})

    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model}", flush=True)
        if model in NEEDS_OWN_ENV and not args.python_unimol:
            print(f"  [warn] {model} usually needs its own environment; "
                  f"pass --python_unimol to point at it", flush=True)
        if "embed" in args.stages and (args.force or not done(out_dir, f"embed_{model}")):
            embed(args, model, out_dir, prepared)
            mark(out_dir, f"embed_{model}", {"model": model})
        if "distance" in args.stages and (args.force or not done(out_dir, f"distance_{model}")):
            distance(args, model, out_dir, prepared)
            mark(out_dir, f"distance_{model}", {"model": model})

    if "merge" in args.stages:
        cmd = [args.python, os.path.join(SCRIPTS, "merge_dist.py"),
               "--prepared_library_csv", prepared["library"],
               "--distance_dir", out_dir,
               "--out_csv", os.path.join(out_dir, "similarity_distance.csv")]
        if args.base_csv:
            cmd += ["--base_csv", os.path.abspath(args.base_csv)]
        run(cmd)


if __name__ == "__main__":
    main()
