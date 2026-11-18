# SynFormer reinforcement-learning runs

The SynFormer arm of the generation benchmark: reinforcement-learning runs on the BTK reference compound under
each of the four rewards used in the comparison.

The code here is a trimmed fork of [SynFormer](https://github.com/wenhao-gao/synformer),
kept under its own Apache-2.0 `LICENSE`; only modules reachable from
`python -m synformer.molopt` are included. The rewards are the same four used
for REINVENT and LibINVENT, with the same transformations, and the two PED
oracles call the same scoring components as those runs
([`../scoring_components/`](../scoring_components/)).

| Reward | What it measures | Extra artefacts needed |
|---|---|---|
| **ECFP4** | Morgan fingerprint (radius 2, 2048 bit) Tanimoto similarity to the reference | none |
| **PED-MolFormer** | distance between pretrained MolFormer embeddings of query and reference | MolFormer weights (HuggingFace) |
| **PED-GeoDiff** | distance between pretrained GeoDiff embeddings of query and reference | GeoDiff checkpoint |
| **Roshambo2** | 3D shape + pharmacophore overlap with the reference ligand | Roshambo2 package, reference SDF |

Only the RL method is included. Other baselines reported in the paper
(e.g. the genetic-algorithm runs) are not part of this package.

The reference compound is the BTK inhibitor co-crystallised in PDB **6O8I**:

```
C/C=C/C(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1
```

## What the experiment does

SynFormer-ED is an encoder–decoder model over *synthesis programs*: the decoder
emits building blocks and reaction templates, which are executed to give a
molecule together with the route that makes it. Every molecule it proposes is
therefore synthesisable by construction.

Each optimisation step:

1. **Encode** — the reference SMILES is tokenised and used as the encoder input
   for every row of the batch (`--reference-smiles`), so generation is a
   projection of the target into synthesisable space rather than unconditional
   sampling.
2. **Decode** — the decoder samples a batch of synthesis programs, which are
   executed into molecules plus their routes.
3. **Reward** — the batch is scored by one of the four oracles above.
4. **Update** — the policy is updated with the REINVENT-style augmented
   likelihood `loss = (ll_agent − (ll_prior + σ · score))²`, where `ll_prior`
   comes from a frozen copy of the pretrained model (`--use-prior`), plus a
   second pass over high-scoring past experiences (`--use-replay-buffer`).

The budget is **10,000 oracle calls**, counted as unique molecules scored, and
is identical for all four rewards. Everything else is held fixed: batch size
128, Adam at 1e-4, σ = 500.

## Install

```bash
conda env create -f env.yml -n synformer
conda activate synformer
pip install --no-deps -e .
```

The ECFP4 reward runs with this environment alone. The two PED rewards and the
Roshambo2 reward additionally need the pretrained encoders and Roshambo2 that
the REINVENT and LibINVENT runs use, installed as described in
[`../README.md`](../README.md):

```bash
cd ../external && ./setup.sh                 # Roshambo2 and reinvent-scoring, pinned
cd ../scoring_components && python install.py
```


## Download the pretrained model and building-block data

None of these files are redistributed here. Download them from the SynFormer
release (<https://huggingface.co/whgao/synformer>) and place them exactly as
shown — the checkpoint records these relative paths internally:

```
data/
├── trained_weights/
│   └── sf_ed_default.ckpt              # 229M-parameter SynFormer-ED weights
└── processed/
    └── comp_2048/
        ├── fpindex.pkl                 # building-block fingerprint index
        └── matrix.pkl                  # reactant × reaction-template matrix
```

`fpindex.pkl` and `matrix.pkl` are derived from Enamine's building-block
catalog, which is available from Enamine on request; the SynFormer release
provides the preprocessed form for research use. No training is performed
here — the pretrained checkpoint is only fine-tuned by the RL loop.

## Run

```bash
scripts/run_rl.sh ecfp4        # ECFP4 Tanimoto
scripts/run_rl.sh molformer    # PED-MolFormer
scripts/run_rl.sh geodiff3d    # PED-GeoDiff
scripts/run_rl.sh roshambo2     # Roshambo2 3D overlap
```

Each writes to `logs/<oracle>/`. The wrapper is a thin one around the module
entry point, which can equally be called directly:

```bash
python -m synformer.molopt \
    --model data/trained_weights/sf_ed_default.ckpt \
    --oracle "ecfp4:C/C=C/C(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1" \
    --reference-smiles "C/C=C/C(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1" \
    --max-call 10000 --max-steps 1000 --batch-size 128 \
    --lr 1e-4 --sigma 500.0 --use-prior --use-replay-buffer \
    --log-dir logs/ecfp4
```

`MAX_CALL`, `MAX_STEPS`, `BATCH_SIZE`, `LR`, `SIGMA` and `MODEL` can be
overridden as environment variables, e.g. `MAX_STEPS=2 scripts/run_rl.sh ecfp4`
for a quick smoke test. `--max-steps` is only a runaway guard; the oracle-call
budget is what ends a run.


## Outputs

A run directory contains:

| File | Contents |
|---|---|
| `molecules.csv` | **every** generated molecule in order: `step, smiles, score, synthesis` |
| `oracle_log.txt` | top-1 / top-10 / top-100 mean, SA, diversity, logged every 100 calls; AUC at the end |
| `oracle_*.yaml` | score buffer snapshots (`smiles: [score, call index]`) |
| `synthesis.pkl` | executed synthesis stacks, keyed by product SMILES |
| `config.yaml` | the exact settings the run used |
| `model.pt` | the fine-tuned policy |

`synthesis` is the route as `reagent;reagent;R<template id>;...`, the full
sequence that builds the molecule. Rows with an empty `smiles` are programs
that failed to execute; they score 0.0.

To derive the two reported tables from a finished run:

```bash
python scripts/collect_results.py logs/ecfp4 --name synformer_rl_btk_ecfp4
```

which writes to `results/`:

* `*_generated.csv` — every generation, in order, duplicates kept. Use this for
  anything per-step: it is the policy's actual output distribution.
* `*_explored.csv` — one row per unique molecule, score-sorted. Use this for
  what the 10,000-call budget was spent on.

In the explored file, `step` is the step at which a molecule was **first**
generated; one generated again is recorded only at that first appearance.

## Oracles and their dependencies

All four oracles live in `synformer/molopt/oracles.py` and share one interface —
`score_batch(list[str]) -> list[float]` returning scores in [0, 1], higher
better, 0.0 for an unparseable SMILES. They are selected by the
`--oracle <name>:<argument>` string, dispatched in
`synformer/molopt/oracle.py::get_oracle_evaluator`.

**ECFP4** (`ecfp4:<smiles>`) needs only RDKit, so this run reproduces from
public artefacts alone. Radius 2 is what makes the fingerprint ECFP4; radius 3
would be ECFP6. The score is the raw Tanimoto, with no transformation applied.
Note that the fingerprint is stereo-blind by default, so a stereoisomer of the
reference also scores 1.0; `ECFP4Oracle(..., use_chirality=True)` changes that.

**PED-MolFormer** (`molformer:<smiles>`) embeds query and reference with
`ibm/MoLFormer-XL-both-10pct` from the HuggingFace hub and maps the Euclidean
distance through a sigmoid (high 17, low 5, k −0.25).

**PED-GeoDiff** (`geodiff3d:<smiles>`) embeds with a pretrained GeoDiff model
(the public `drugs_default.pt` DRUGS checkpoint) and maps the distance through
a sigmoid (high 10, low 1, k −0.25). `geodiff3d` uses the 3D (geometry)
embedding and is the variant reported; `geodiff2d` (graph topology, sigmoid
high 1.2 / low 0.1) and `geodiff` (concatenated, normalised) are also
available. All variants generate RDKit conformers as GNN input.

The two PED oracles reach their pretrained encoders through `reinvent_scoring`
scoring components (`MolformerDistance`, `GeoDiffDistance`), which is why that
package is imported lazily. The dependency is narrow and documented: any object
with `calculate_score(list[rdkit.Chem.Mol])` returning an object whose
`total_score` is a sequence of floats can be substituted. The exact
`specific_parameters` used for the reported numbers are in each oracle's
`_get_scorer` method, so the reward is reproducible from the two public
pretrained models and those settings.

**Roshambo2** (`roshambo2:<path to SDF>`) enumerates up to 4 stereoisomers and 25
conformers per molecule (ETKDGv3, MMFF-optimised, fixed random seed) and
returns `tanimoto_combination`, the shape-plus-pharmacophore overlap with the
reference, reduced over conformers and maximised over stereoisomers. It needs
the Roshambo2 package to be importable and a reference ligand, which defaults to
[`../../data/bmsBTK_6O8I_bioactive.sdf`](../../data/bmsBTK_6O8I_bioactive.sdf),
the bioactive conformer of the ligand in PDB entry 6O8I. Override it with the
`REFERENCE_SDF` environment variable.

## Repository contents

```
synformer/molopt/        the RL loop: entry point, policy update, oracles, replay buffer
synformer/models/        SynFormer encoder-decoder
synformer/chem/          molecules, reactions, building-block index, synthesis stacks
synformer/data/          batch/token definitions used by the model
scripts/run_rl.sh        one RL run per reward
scripts/collect_results.py   raw run output -> the two reported CSVs
```

Only modules reachable from `python -m synformer.molopt` are included; training
and sampling utilities not used by these experiments have been left out.

## Citation

SynFormer: <https://arxiv.org/abs/2410.03494>
