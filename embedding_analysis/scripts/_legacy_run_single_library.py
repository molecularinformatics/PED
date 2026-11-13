import os
import argparse
import subprocess
from pathlib import Path


# def run_cmd(cmd, cwd=None, env=None):
#     print("Running:", " ".join(cmd))
#     subprocess.run(cmd, check=True, cwd=cwd, env=env)


def run_cmd(cmd, cwd=None, env=None):
    myenv = os.environ.copy()

    conda_prefix = myenv.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        old_ld = myenv.get("LD_LIBRARY_PATH", "")
        myenv["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}" if old_ld else conda_lib

    if env is not None:
        myenv.update(env)

    print("Running:", " ".join(cmd))
    print("LD_LIBRARY_PATH =", myenv.get("LD_LIBRARY_PATH", ""))
    subprocess.run(cmd, check=True, cwd=cwd, env=myenv)


def normalize_gpu_ids(devices):
    gpu_ids = []
    for d in devices:
        if not d.startswith("cuda:"):
            raise ValueError(f"Expected device like cuda:0, got: {d}")
        gpu_ids.append(d.split(":", 1)[1])
    return gpu_ids


def run_molformer_embed(script_path, csv_path, out_dir, model_name, batch_size, devices):
    gpu_ids = normalize_gpu_ids(devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    if len(gpu_ids) == 1:
        cmd = [
            "python", str(script_path),
            "--csv_path", str(csv_path),
            "--cache_dir", str(out_dir),
            "--smiles_col", "smiles",
            "--model_name", model_name,
            "--batch_size", str(batch_size),
        ]
    else:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node", str(len(gpu_ids)),
            str(script_path),
            "--csv_path", str(csv_path),
            "--cache_dir", str(out_dir),
            "--smiles_col", "smiles",
            "--model_name", model_name,
            "--batch_size", str(batch_size),
        ]

    run_cmd(cmd, env=env)


def run_unimol_embed(script_path, csv_path, out_dir, batch_size, model_size, device):
    gpu_id = normalize_gpu_ids([device])[0]
    cmd = [
        "python", str(script_path),
        "--input_csv", str(csv_path),
        "--output_dir", str(out_dir),
        "--smiles_col", "smiles",
        "--batch_size", str(batch_size),
        "--model_name", "unimolv2",
        "--model_size", str(model_size),
        "--use_cuda",
        "--use_gpu", str(gpu_id),
    ]
    run_cmd(cmd)


def run_ecfp_embed(script_path, csv_path, out_dir, n_workers, radius, nbits):
    cmd = [
        "python", str(script_path),
        "--input_csv", str(csv_path),
        "--output_dir", str(out_dir),
        "--smiles_col", "smiles",
        "--radius", str(radius),
        "--nbits", str(nbits),
        "--n_workers", str(n_workers),
    ]
    run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run packing, embedding, and distance pipeline for one library CSV and one reference SMILES"
    )
    parser.add_argument("--library_csv", required=True, help="Input CSV that contains the library SMILES column")
    parser.add_argument("--reference_smiles", required=True, help="Single reference SMILES string")
    parser.add_argument("--out_root", required=True, help="Root output folder")
    parser.add_argument("--job_name", default=None, help="Output subfolder name. Default: stem of library_csv")
    parser.add_argument("--smiles_col", default="Smiles", help="SMILES column name in the input CSV")
    parser.add_argument("--ref_confs", type=int, default=1, help="Number of conformers per molecule")
    parser.add_argument("--n_workers", type=int, default=45, help="Workers for the library packing step")
    parser.add_argument("--chunksize", type=int, default=256)
    parser.add_argument("--shard_size", type=int, default=100000)
    parser.add_argument("--failed_shard_size", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None, help="Multiple GPUs for library embedding, e.g. cuda:0 cuda:1 cuda:2")
    parser.add_argument("--distance_type", nargs="+", default=["cosine", "euclidean"], choices=["cosine", "euclidean"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["2D", "3D", "concat", "concat_norm"], choices=["2D", "3D", "concat", "concat_norm"])

    parser.add_argument("--run_geodiff", action="store_true", help="Run GeoDiff embedding generation and distance computation")
    parser.add_argument("--run_molformer", action="store_true", help="Run MolFormer embedding generation and distance computation")
    parser.add_argument("--run_unimol", action="store_true", help="Run UniMol embedding generation and distance computation")
    parser.add_argument("--run_ecfp", action="store_true", help="Run ECFP fingerprint generation and distance computation")
    parser.add_argument("--run_sde", action="store_true", help="Run MoleculeSDE")

    parser.add_argument("--geodiff_batch_size", type=int, default=256, help="Per-GPU batch size for get_geodiff_embedding.py")
    parser.add_argument("--molformer_batch_size", type=int, default=256)
    parser.add_argument("--unimol_batch_size", type=int, default=256)
    parser.add_argument("--sde_batch_size", type=int, default=256)
    
    parser.add_argument("--geodiff_ckpt", type=str, default="./geodiff/log/model/checkpoints/drugs_default.pt", help="GeoDiff checkpoint path")
    parser.add_argument("--sde_ckpt", type=str, default="./MoleculeSDE/model_complete.pth", help="MoleculeSDE checkpoint path")
    parser.add_argument("--molformer_model_name", type=str, default="ibm/MoLFormer-XL-both-10pct")
    parser.add_argument("--unimol_model_size", type=str, default="164m", choices=["84m", "164m", "1.1B"])
    parser.add_argument("--ecfp_radius", type=int, default=2)
    parser.add_argument("--ecfp_nbits", type=int, default=2048)

    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_pack_ref", action="store_true")
    parser.add_argument("--skip_pack_lib", action="store_true")
    parser.add_argument("--skip_geodiff_ref", action="store_true")
    parser.add_argument("--skip_geodiff_lib", action="store_true")
    parser.add_argument("--skip_geodiff_distance", action="store_true")
    parser.add_argument("--skip_molformer_ref", action="store_true")
    parser.add_argument("--skip_molformer_lib", action="store_true")
    parser.add_argument("--skip_molformer_distance", action="store_true")
    parser.add_argument("--skip_unimol_ref", action="store_true")
    parser.add_argument("--skip_unimol_lib", action="store_true")
    parser.add_argument("--skip_unimol_distance", action="store_true")
    parser.add_argument("--skip_sde_prepare_ref", action="store_true")
    parser.add_argument("--skip_sde_prepare_lib", action="store_true")
    parser.add_argument("--skip_sde_ref", action="store_true")
    parser.add_argument("--skip_sde_lib", action="store_true")
    parser.add_argument("--skip_sde_distance", action="store_true")
    parser.add_argument("--skip_ecfp_ref", action="store_true")
    parser.add_argument("--skip_ecfp_lib", action="store_true")
    parser.add_argument("--skip_ecfp_distance", action="store_true")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    library_csv = Path(args.library_csv).resolve()
    job_name = args.job_name or library_csv.stem
    out_dir = Path(args.out_root).resolve() / job_name

    prep_dir = out_dir / "prepared"
    refs_csv = prep_dir / "refs.csv"
    prepared_library_csv = prep_dir / "library.csv"

    refs_pack_dir = out_dir / "refs" / "geodiff_packed"
    lib_pack_dir = out_dir / "library" / "geodiff_packed"
    refs_sde_prep_dir = out_dir / "refs" / "sde_prepared"
    lib_sde_prep_dir = out_dir / "library" / "sde_prepared"

    refs_emb_dir = out_dir / "refs" / "embeddings"
    lib_emb_dir = out_dir / "library" / "embeddings"

    refs_molformer_dir = refs_emb_dir / "molformer"
    lib_molformer_dir = lib_emb_dir / "molformer"

    refs_unimol_dir = refs_emb_dir / "unimol"
    lib_unimol_dir = lib_emb_dir / "unimol"

    refs_ecfp_dir = refs_emb_dir / "ecfp"
    lib_ecfp_dir = lib_emb_dir / "ecfp"

    geodiff_distance_csv = out_dir / "distance" / "smiles_reference_geodiff_distances.csv"
    molformer_distance_csv = out_dir / "distance" / "smiles_reference_molformer_distances.csv"
    unimol_distance_csv = out_dir / "distance" / "smiles_reference_unimol_distances.csv"
    ecfp_distance_csv = out_dir / "distance" / "smiles_reference_ecfp_distances.csv"
    sde_distance_csv = out_dir / "distance" / "smiles_reference_sde_distances.csv"

    devices = args.devices if args.devices else [args.device]

    if not args.skip_prepare:
        run_cmd([
            "python", str(script_dir / "prepare_smiles_reference_inputs.py"),
            "--library_csv", str(library_csv),
            "--reference_smiles", str(args.reference_smiles),
            "--smiles_col", str(args.smiles_col),
            "--out_dir", str(prep_dir),
        ])

    if args.run_molformer:
        molformer_script = script_dir / "get_molformer_embedding.py"

        if not args.skip_molformer_ref:
            run_molformer_embed(
                script_path=molformer_script,
                csv_path=refs_csv,
                out_dir=refs_molformer_dir,
                model_name=args.molformer_model_name,
                batch_size=args.molformer_batch_size,
                devices=[devices[0]],
            )

        if not args.skip_molformer_lib:
            run_molformer_embed(
                script_path=molformer_script,
                csv_path=prepared_library_csv,
                out_dir=lib_molformer_dir,
                model_name=args.molformer_model_name,
                batch_size=args.molformer_batch_size,
                devices=devices,
            )

        if not args.skip_molformer_distance:
            run_cmd([
                "python", str(script_dir / "compute_smiles_reference_distances.py"),
                "--refs_csv", str(refs_csv),
                "--library_csv", str(prepared_library_csv),
                "--refs_emb_root", str(refs_molformer_dir),
                "--library_emb_root", str(lib_molformer_dir),
                "--out_csv", str(molformer_distance_csv),
                "--model", "molformer",
                "--modes", "single",
                "--distance_type", *args.distance_type,
            ])

    if args.run_unimol:
        unimol_script = script_dir / "get_unimol_embedding.py"

        if not args.skip_unimol_ref:
            run_unimol_embed(
                script_path=unimol_script,
                csv_path=refs_csv,
                out_dir=refs_unimol_dir,
                batch_size=args.unimol_batch_size,
                model_size=args.unimol_model_size,
                device=devices[0],
            )

        if not args.skip_unimol_lib:
            run_unimol_embed(
                script_path=unimol_script,
                csv_path=prepared_library_csv,
                out_dir=lib_unimol_dir,
                batch_size=args.unimol_batch_size,
                model_size=args.unimol_model_size,
                device=devices[0],
            )

        if not args.skip_unimol_distance:
            run_cmd([
                "python", str(script_dir / "compute_smiles_reference_distances.py"),
                "--refs_csv", str(refs_csv),
                "--library_csv", str(prepared_library_csv),
                "--refs_emb_root", str(refs_unimol_dir),
                "--library_emb_root", str(lib_unimol_dir),
                "--out_csv", str(unimol_distance_csv),
                "--model", "unimol",
                "--modes", "single",
                "--distance_type", *args.distance_type,
            ])

    if args.run_ecfp:
        ecfp_script = script_dir / "get_ecfp.py"

        if not args.skip_ecfp_ref:
            run_ecfp_embed(
                script_path=ecfp_script,
                csv_path=refs_csv,
                out_dir=refs_ecfp_dir,
                n_workers=1,
                radius=args.ecfp_radius,
                nbits=args.ecfp_nbits,
            )

        if not args.skip_ecfp_lib:
            run_ecfp_embed(
                script_path=ecfp_script,
                csv_path=prepared_library_csv,
                out_dir=lib_ecfp_dir,
                n_workers=args.n_workers,
                radius=args.ecfp_radius,
                nbits=args.ecfp_nbits,
            )

        if not args.skip_ecfp_distance:
            run_cmd([
                "python", str(script_dir / "compute_smiles_reference_distances.py"),
                "--refs_csv", str(refs_csv),
                "--library_csv", str(prepared_library_csv),
                "--refs_emb_root", str(refs_ecfp_dir),
                "--library_emb_root", str(lib_ecfp_dir),
                "--out_csv", str(ecfp_distance_csv),
                "--model", "ecfp",
                "--modes", "single",
                "--distance_type", *args.distance_type,
            ])

    if args.run_sde:
        if not args.skip_sde_prepare_ref:
            run_cmd([
                "python", str(script_dir / "pack_sde_data.py"),
                "--csv", str(refs_csv),
                "--smiles_col", "smiles",
                "--out_dir", str(refs_sde_prep_dir),
                "--n_workers", "1",
                "--chunksize", "1",
                "--shard_size", "1000",
                "--num_confs", str(args.ref_confs),
                "--seed", str(args.seed),
            ])

        if not args.skip_sde_prepare_lib:
            run_cmd([
                "python", str(script_dir / "pack_sde_data.py"),
                "--csv", str(prepared_library_csv),
                "--smiles_col", "smiles",
                "--out_dir", str(lib_sde_prep_dir),
                "--n_workers", str(args.n_workers),
                "--chunksize", str(args.chunksize),
                "--shard_size", str(args.shard_size),
                "--num_confs", str(args.ref_confs),
                "--seed", str(args.seed),
            ])

        if not args.skip_sde_ref:
            run_cmd([
                "python", str(script_dir / "get_sde_embedding.py"),
                "--ckpt", os.path.abspath(args.sde_ckpt),
                "--prepared_dir", str(refs_sde_prep_dir),
                "--out_dir", str(refs_emb_dir),
                "--batch_size", str(args.sde_batch_size),
                "--device", devices[0],
            ])

        if not args.skip_sde_lib:
            cmd = [
                "python", str(script_dir / "get_sde_embedding.py"),
                "--ckpt", os.path.abspath(args.sde_ckpt),
                "--prepared_dir", str(lib_sde_prep_dir),
                "--out_dir", str(lib_emb_dir),
                "--batch_size", str(args.sde_batch_size),
            ]
            if len(devices) == 1:
                cmd.extend(["--device", devices[0]])
            else:
                cmd.extend(["--devices", *devices])
            run_cmd(cmd)

        if not args.skip_sde_distance:
            run_cmd([
                "python", str(script_dir / "compute_smiles_reference_distances.py"),
                "--refs_csv", str(refs_csv),
                "--library_csv", str(prepared_library_csv),
                "--refs_emb_root", str(refs_emb_dir),
                "--library_emb_root", str(lib_emb_dir),
                "--out_csv", str(sde_distance_csv),
                "--model", "sde",
                "--distance_type", *args.distance_type,
                "--modes", *args.modes,
            ])


    if args.run_geodiff:
        if not args.skip_pack_ref:
            run_cmd([
                "python", str(script_dir / "pack_geodiff_data.py"),
                "--csv", str(refs_csv),
                "--smiles_col", "smiles",
                "--out_dir", str(refs_pack_dir),
                "--ref_confs", str(args.ref_confs),
                "--seed", str(args.seed),
                "--n_workers", "1",
                "--chunksize", "1",
                "--shard_size", str(args.shard_size),
                "--failed_shard_size", str(args.failed_shard_size),
            ])

        if not args.skip_pack_lib:
            run_cmd([
                "python", str(script_dir / "pack_geodiff_data.py"),
                "--csv", str(prepared_library_csv),
                "--smiles_col", "smiles",
                "--out_dir", str(lib_pack_dir),
                "--ref_confs", str(args.ref_confs),
                "--seed", str(args.seed),
                "--n_workers", str(args.n_workers),
                "--chunksize", str(args.chunksize),
                "--shard_size", str(args.shard_size),
                "--failed_shard_size", str(args.failed_shard_size),
            ])

        if not args.skip_geodiff_ref:
            run_cmd([
                "python", str(script_dir / "get_geodiff_embedding.py"),
                "--ckpt", os.path.abspath(args.geodiff_ckpt),
                "--packed_dir", str(refs_pack_dir),
                "--out_dir", str(refs_emb_dir),
                "--batch_size", str(args.geodiff_batch_size),
                "--seed", str(args.seed),
                "--device", "cuda:0",
            ])

        if not args.skip_geodiff_lib:
            cmd = [
                "python", str(script_dir / "get_geodiff_embedding.py"),
                "--ckpt", os.path.abspath(args.geodiff_ckpt),
                "--packed_dir", str(lib_pack_dir),
                "--out_dir", str(lib_emb_dir),
                "--batch_size", str(args.geodiff_batch_size),
                "--seed", str(args.seed),
            ]
            if len(devices) == 1:
                cmd.extend(["--device", devices[0]])
            else:
                cmd.extend(["--devices", *devices])
            run_cmd(cmd)

        if not args.skip_geodiff_distance:
            run_cmd([
                "python", str(script_dir / "compute_smiles_reference_distances.py"),
                "--refs_csv", str(refs_csv),
                "--library_csv", str(prepared_library_csv),
                "--refs_emb_root", str(refs_emb_dir),
                "--library_emb_root", str(lib_emb_dir),
                "--out_csv", str(geodiff_distance_csv),
                "--model", "geodiff",
                "--distance_type", *args.distance_type,
                "--modes", *args.modes,
            ])

    print("Pipeline finished")
    print(f"Outputs are under: {out_dir}")


if __name__ == "__main__":
    main()
