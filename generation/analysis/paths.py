"""Where the generation runs are.

One run directory per generator, each holding `scaffold_memory/` with one CSV per
reward. Set PED_GENERATION_RESULTS to point elsewhere.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.environ.get("PED_GENERATION_RESULTS",
                                   REPO_ROOT / "results" / "generation"))
OUT_DIR = Path(os.environ.get("PED_GENERATION_FIGS", REPO_ROOT / "figs"))

# Reward name used in the paper -> the suffix of its run file.
REWARDS = {
    "ROSHAMBO2": "roshambo2",
    "ECFP4": "ecfp4",
    "GeoDiff 2D": "geodiff_2d",
    "GeoDiff 3D": "geodiff_3d",
    "GeoDiff Concat": "geodiff_concat",
    "MoLFormer": "molformer",
}
