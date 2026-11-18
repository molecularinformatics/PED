"""Emit the REINVENT and LibINVENT run configurations, one per reward."""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATION = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(GENERATION)

# Paths are written relative to the repository root so the committed configurations
# carry no machine-specific path; pass --absolute to materialise them for a run.
PREFIX = ""


def path(*parts: str) -> str:
    return os.path.join(PREFIX, *parts) if PREFIX else os.path.join(*parts)


EXTERNAL = ("generation", "external")
REFERENCE_SDF_PARTS = ("data", "bmsBTK_6O8I_bioactive.sdf")
REFERENCE_SMILES = "C/C=C/C(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1"

# Distance ranges used for the BTK runs, keyed by generator and reward.
PED_BOUNDS = {
    "reinvent": {"geodiff_2d": (0.1, 1.2), "geodiff_3d": (1, 10),
                 "geodiff_concat": (0.1, 0.58), "molformer": (5, 17)},
    "libinvent": {"geodiff_2d": (0.1, 0.9), "geodiff_3d": (1, 10),
                  "geodiff_concat": (0.1, 0.5), "molformer": (6, 15)},
}
GEODIFF_MODES = {"geodiff_2d": "2D", "geodiff_3d": "3D", "geodiff_concat": "concat_norm"}


def reverse_sigmoid(low: float, high: float, k: float = 0.25) -> dict:
    return {"transformation_type": "reverse_sigmoid", "low": low, "high": high, "k": k}


def component(model: str, reward: str) -> dict:
    """The one scoring component that distinguishes the arms."""
    if reward == "ecfp4":
        # The component defaults to radius 3 with count and feature invariants
        # (FCFP6 counts), so binary ECFP4 has to be requested explicitly.
        return {
            "component_type": "tanimoto_similarity",
            "name": "ECFP4 similarity to reference",
            "weight": 1,
            "specific_parameters": {
                "smiles": [REFERENCE_SMILES],
                "radius": 2,
                "use_counts": False,
                "use_features": False,
            },
        }
    if reward == "roshambo2":
        return {
            "component_type": "roshambo2_similarity",
            "name": "ROSHAMBO2 combination similarity",
            "weight": 1,
            "specific_parameters": {
                "reference_path": path(*REFERENCE_SDF_PARTS),
                "num_confs": 25,
                "num_isomers": 2,
                "num_processes": 24,
                "combination_param": 0.5,
                "backend": "cpp",
                "random_seed": 42,
            },
        }
    if reward == "molformer":
        return {
            "component_type": "molformer_distance",
            "name": "MoLFormer Euclidean PED",
            "weight": 1,
            "specific_parameters": {
                "reference_smiles": REFERENCE_SMILES,
                "model_name": "ibm/MoLFormer-XL-both-10pct",
                "distance_metric": "euclidean",
                "transformation": reverse_sigmoid(*PED_BOUNDS[model][reward]),
            },
        }
    if reward in GEODIFF_MODES:
        mode = GEODIFF_MODES[reward]
        return {
            "component_type": "geodiff_distance",
            "name": f"GeoDiff {mode} Euclidean PED",
            "weight": 1,
            "specific_parameters": {
                "reference_smiles": REFERENCE_SMILES,
                "ckpt_path": path("external", "checkpoints", "geodiff",
                                  "model", "checkpoints", "drugs_default.pt"),
                "embedding_mode": mode,
                "distance_metric": "euclidean",
                "transformation": reverse_sigmoid(*PED_BOUNDS[model][reward]),
            },
        }
    raise ValueError(f"unknown reward {reward!r}")


def scoring_function(model: str, reward: str) -> dict:
    return {
        "name": "custom_sum",
        "parallel": False,
        "parameters": [component(model, reward)],
    }


def reinvent_config(reward: str, out_dir: str, n_steps: int) -> dict:
    prior = path(*EXTERNAL, "priors", "augmented.prior")
    return {
        "version": 3,
        "run_type": "reinforcement_learning",
        "logging": {
            "sender": "http://127.0.0.1",
            "recipient": "local",
            "logging_frequency": 10,
            "logging_path": os.path.join(out_dir, "progress.log"),
            "result_folder": os.path.join(out_dir, "results"),
            "job_name": f"reinvent_{reward}",
            "job_id": f"reinvent_{reward}",
        },
        "parameters": {
            "scoring_function": scoring_function("reinvent", reward),
            "diversity_filter": {
                "name": "NoFilter",
                "bucket_size": 25,
                "minscore": 0.0,
                "minsimilarity": 0.4,
            },
            "inception": {"smiles": [], "memory_size": 100, "sample_size": 10},
            "reinforcement_learning": {
                "prior": prior,
                "agent": prior,
                "n_steps": n_steps,
                "sigma": 128,
                "learning_rate": 0.0001,
                "batch_size": 128,
                "margin_threshold": 50,
            },
        },
    }


def libinvent_config(reward: str, out_dir: str, n_steps: int, scaffold: str) -> dict:
    prior = path(*EXTERNAL, "priors", "library_design.prior")
    return {
        "version": 3,
        "run_type": "lib_invent_reinforcement_learning",
        "logging": {
            "logging_path": os.path.join(out_dir, "progress.log"),
            "result_folder": os.path.join(out_dir, "results"),
            "job_name": f"libinvent_{reward}",
            "job_id": f"libinvent_{reward}",
            "sender": "http://127.0.0.1",
            "recipient": "local",
        },
        "parameters": {
            "actor": prior,
            "critic": prior,
            "scaffolds": [scaffold],
            "n_steps": n_steps,
            "learning_rate": 0.0001,
            "batch_size": 128,
            "randomize_scaffolds": False,
            "learning_strategy": {
                "name": "dap",
                "parameters": {"sigma": 120},
            },
            "scoring_strategy": {
                "name": "lib_invent",
                "scoring_function": scoring_function("libinvent", reward),
                "diversity_filter": {
                    "name": "NoFilterWithPenalty",
                    "bucket_size": 25,
                    "minscore": 0.0,
                    "minsimilarity": 0.4,
                },
                "reaction_filter": {"type": "selective", "reactions": []},
            },
        },
    }


REWARDS = ["ecfp4", "roshambo2", "geodiff_2d", "geodiff_3d", "geodiff_concat", "molformer"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=HERE)
    ap.add_argument("--results_root", default=None,
                    help="default: results/generation, relative to the repository root")
    ap.add_argument("--absolute", action="store_true",
                    help="write absolute paths for this machine instead of repo-relative ones")
    ap.add_argument("--rewards", nargs="+", default=REWARDS, choices=REWARDS)
    ap.add_argument("--reinvent_steps", type=int, default=10000)
    ap.add_argument("--libinvent_steps", type=int, default=5000)
    ap.add_argument("--scaffold", default="CC=CC([*:0])=O",
                    help="LibINVENT decoration scaffold with attachment points")
    args = ap.parse_args()

    global PREFIX
    if args.absolute:
        PREFIX = REPO_ROOT
    results_root = args.results_root or path("results", "generation")

    os.makedirs(args.out_dir, exist_ok=True)
    for reward in args.rewards:
        for model, builder, steps in (
            ("reinvent", reinvent_config, args.reinvent_steps),
            ("libinvent", libinvent_config, args.libinvent_steps),
        ):
            out_dir = os.path.join(results_root, f"{model}_{reward}")
            cfg = (builder(reward, out_dir, steps) if model == "reinvent"
                   else builder(reward, out_dir, steps, args.scaffold))
            out_path = os.path.join(args.out_dir, f"{model}_{reward}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"wrote {os.path.relpath(out_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
