#!/usr/bin/env bash
#
# Reproduce one reinforcement-learning run of SynFormer-ED on the BTK reference
# compound. One run per reward; everything else is held fixed.
#
#   Usage:  scripts/run_rl.sh <oracle>
#   where <oracle> is one of:
#
#     ecfp4       ECFP4 (Morgan r=2, 2048 bit) Tanimoto to the reference
#     molformer   PED-MolFormer : pretrained MolFormer embedding distance
#     geodiff3d   PED-GeoDiff   : pretrained GeoDiff 3D embedding distance
#     geodiff2d   PED-GeoDiff   : 2D (graph) embedding variant
#     geodiff     PED-GeoDiff   : concatenated 2D+3D embedding variant
#     roshambo2   Roshambo2 3D shape + pharmacophore overlap
#
# Run it from the repository root, with the environment of README.md active.
#
set -euo pipefail

ORACLE="${1:-}"
if [ -z "$ORACLE" ]; then
    echo "usage: $0 {ecfp4|molformer|geodiff3d|geodiff2d|geodiff|roshambo2}" >&2
    exit 2
fi

# --- the target: BTK inhibitor, bound ligand of PDB 6O8I --------------------
REFERENCE_SMILES="C/C=C/C(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1"
REFERENCE_SDF="${REFERENCE_SDF:-../../data/bmsBTK_6O8I_bioactive.sdf}"

MODEL="${MODEL:-data/trained_weights/sf_ed_default.ckpt}"

# --- shared optimisation settings (identical for every reward) -------------
MAX_CALL="${MAX_CALL:-10000}"      # oracle-call budget: unique molecules scored
MAX_STEPS="${MAX_STEPS:-1000}"     # safety cap only; the budget stops the run
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
SIGMA="${SIGMA:-500.0}"

case "$ORACLE" in
    ecfp4|molformer|geodiff|geodiff2d|geodiff3d)
        ORACLE_ARG="${ORACLE}:${REFERENCE_SMILES}"
        ;;
    roshambo2)
        if [ ! -f "$REFERENCE_SDF" ]; then
            echo "Missing reference ligand: $REFERENCE_SDF" >&2
            echo "See README.md, \"Oracles and their dependencies\"." >&2
            exit 1
        fi
        ORACLE_ARG="roshambo2:${REFERENCE_SDF}"
        ;;
    *)
        echo "unknown oracle: $ORACLE" >&2
        exit 2
        ;;
esac

if [ ! -f "$MODEL" ]; then
    echo "Missing pretrained checkpoint: $MODEL" >&2
    echo "See README.md, \"Download the pretrained model and building-block data\"." >&2
    exit 1
fi

echo "oracle        : $ORACLE"
echo "budget        : $MAX_CALL oracle calls"
echo "batch size    : $BATCH_SIZE"
echo "output        : logs/$ORACLE/"

python -m synformer.molopt \
    --model "$MODEL" \
    --oracle "$ORACLE_ARG" \
    --reference-smiles "$REFERENCE_SMILES" \
    --max-call "$MAX_CALL" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --sigma "$SIGMA" \
    --use-prior \
    --use-replay-buffer \
    --log-dir "logs/$ORACLE"
