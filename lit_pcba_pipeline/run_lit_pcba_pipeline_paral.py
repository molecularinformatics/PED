import os
import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None, env=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


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



def main():
    parser = argparse.ArgumentParser(description="Run the full LIT-PCBA pipeline with GeoDiff and optional MolFormer embeddings")
    parser.add_argument("--target_dir", required=True, help="Path to one target folder under LIT-PCBA")
    parser.add_argument("--out_root", required=True, help="Root output folder")
    parser.add_argument("--ref_confs", type=int, default=1, help="Number of conformers per reference and library molecule")
    parser.add_argument("--n_workers", type=int, default=45)
    parser.add_argument("--chunksize", type=int, default=256)
    parser.add_argument("--shard_size", type=int, default=100000)
    parser.add_argument("--failed_shard_size", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None, help="Multiple GPUs for embedding, e.g. cuda:0 cuda:1 cuda:2")
    parser.add_argument("--geodiff_batch_size", type=int, default=128, help="Per-GPU batch size for get_geodiff_embedding.py")
    parser.add_argument("--distance_type", nargs="+", default=["cosine", "euclidean"], choices=["cosine", "euclidean"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["2D", "3D", "concat"], choices=["2D", "3D", "concat", "concat_norm"])

    parser.add_argument("--run_geodiff", action="store_true", help="Run GeoDiff embedding generation and distance computation")
    parser.add_argument("--run_molformer", action="store_true", help="Run MoLFormer embedding generation and distance computation")
    
    parser.add_argument("--geodiff_ckpt", type=str, default="./geodiff/log/model/checkpoints/drugs_default.pt", help="GeoDiff checkpoint path")
    parser.add_argument("--molformer_model_name", type=str, default="ibm/MoLFormer-XL-both-10pct")
    parser.add_argument("--molformer_batch_size", type=int, default=256)

    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_pack_ref", action="store_true")
    parser.add_argument("--skip_pack_lib", action="store_true")
    parser.add_argument("--skip_geodiff_ref", action="store_true")
    parser.add_argument("--skip_geodiff_lib", action="store_true")
    parser.add_argument("--skip_geodiff_distance", action="store_true")
    parser.add_argument("--skip_molformer_ref", action="store_true")
    parser.add_argument("--skip_molformer_lib", action="store_true")
    parser.add_argument("--skip_molformer_distance", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    target_dir = Path(args.target_dir).resolve()
    target_name = target_dir.name
    out_dir = Path(args.out_root).resolve() / target_name

    prep_dir = out_dir / "prepared"
    refs_csv = prep_dir / "refs.csv"
    library_csv = prep_dir / "library.csv"

    refs_pack_dir = out_dir / "refs" / "packed"
    lib_pack_dir = out_dir / "library" / "packed"

    refs_emb_dir = out_dir / "refs" / "embeddings"
    lib_emb_dir = out_dir / "library" / "embeddings"

    refs_molformer_dir = refs_emb_dir / "molformer"
    lib_molformer_dir = lib_emb_dir / "molformer"

    geodiff_distance_csv = out_dir / "distance" / "library_reference_geodiff_distances.csv"
    molformer_distance_csv = out_dir / "distance" / "library_reference_molformer_distances.csv"

    devices = args.devices if args.devices else [args.device]

    if not args.skip_prepare:
        run_cmd([
            "python", str(script_dir / "prepare_lit_pcba_inputs.py"),
            "--target_dir", str(target_dir),
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
                csv_path=library_csv,
                out_dir=lib_molformer_dir,
                model_name=args.molformer_model_name,
                batch_size=args.molformer_batch_size,
                devices=devices,
            )

        if not args.skip_molformer_distance:
            run_cmd([
                "python", str(script_dir / "compute_pcba_distances.py"),
                "--embedding_backend", "molformer",
                "--refs_csv", str(refs_csv),
                "--library_csv", str(library_csv),
                "--refs_emb_root", str(refs_molformer_dir),
                "--library_emb_root", str(lib_molformer_dir),
                "--out_csv", str(molformer_distance_csv),
                "--distance_type", *args.distance_type,
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
                "--n_workers", str(args.n_workers),
                "--chunksize", str(args.chunksize),
                "--shard_size", str(args.shard_size),
                "--failed_shard_size", str(args.failed_shard_size),
            ])
        if not args.skip_pack_lib:
            run_cmd([
                "python", str(script_dir / "pack_geodiff_data.py"),
                "--csv", str(library_csv),
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
            run_cmd([
                "python", str(script_dir / "get_geodiff_embedding.py"),
                "--ckpt", os.path.abspath(args.geodiff_ckpt),
                "--packed_dir", str(lib_pack_dir),
                "--out_dir", str(lib_emb_dir),
                "--batch_size", str(args.geodiff_batch_size),
                "--seed", str(args.seed),
                "--devices", *devices,
            ])

        if not args.skip_geodiff_distance:
            run_cmd([
                "python", str(script_dir / "compute_pcba_distances.py"),
                "--embedding_backend", "geodiff",
                "--refs_csv", str(refs_csv),
                "--library_csv", str(library_csv),
                "--refs_emb_root", str(refs_emb_dir),
                "--library_emb_root", str(lib_emb_dir),
                "--out_csv", str(geodiff_distance_csv),
                "--distance_type", *args.distance_type,
                "--modes", *args.modes,
            ])

    print(f"Pipeline finished for target: {target_name}")
    print(f"Outputs are under: {out_dir}")


if __name__ == "__main__":
    main()
