# PED-guided molecular generation

Reinforcement-learning generation with REINVENT and LibINVENT, where the reward is a
similarity to one reference ligand. Four reward families are provided so the arms
differ only in that term: a 2D fingerprint (ECFP4), a 3D shape-and-pharmacophore
overlay (Roshambo2), and pretrained embedding distances from GeoDiff and MoLFormer.

```
external/setup.sh        pinned clones of REINVENT, reinvent-scoring, Roshambo2 + public priors
scoring_components/      the custom components, and install.py to register them
configs/                 one run configuration per model and reward
```

## Set up

```bash
cd generation/external
./setup.sh                                   # repos, patch and priors (~110 MB)
./setup.sh verify

cd ../scoring_components
python install.py                            # register into the active reinvent-scoring
python install.py --check
```

`install.py` copies the components into
`reinvent_scoring/scoring/score_components/ped/` and adds their names to the
component enum and the component factory. It is idempotent and `--uninstall`
reverses it. This replaces editing those files by hand.

## Pinned versions

| Piece | Source | Commit |
|---|---|---|
| REINVENT 3.2 (also runs LibINVENT) | [MolecularAI/Reinvent](https://github.com/MolecularAI/Reinvent) `v3.2` | `b7324d222a49d18b08335a01649abdb0ac66a734` |
| reinvent-scoring | [MolecularAI/reinvent-scoring](https://github.com/MolecularAI/reinvent-scoring) `reinvent.3.2` | `9e405b5adddd5bf58721465dfd7c31cc432649bc` |
| Roshambo2 | [molecularinformatics/roshambo2](https://github.com/molecularinformatics/roshambo2) | `7ff8237c4f21372e8fbfb4e842b1b5eee87a7aeb` |
| Priors | [MolecularAI/ReinventCommunity](https://github.com/MolecularAI/ReinventCommunity) `reinvent.3.0` | `e4c1b19c6d9bf7d74f8b8cffb239ca9cc28b4729` |

Both LibINVENT and de novo REINVENT run from the same REINVENT 3.2 checkout, as the
run types `lib_invent_reinforcement_learning` and `reinforcement_learning`.

The priors are the public ones: `augmented.prior` for de novo REINVENT and
`library_design.prior` for LibINVENT.

`roshambo2.patch` makes one change: the preparation loop in `roshambo2/prepare.py`
drops its `tqdm` wrapper. The progress bar writes one line per chunk per worker,
which floods the log when the component is called once per RL step.

## Rewards

| Reward | Component | Note |
|---|---|---|
| ECFP4 | `tanimoto_similarity` (stock) | binary ECFP4 must be requested explicitly: the component defaults to radius 3 with count and feature invariants, i.e. FCFP6 counts |
| Roshambo2 | `roshambo2_similarity` | combined shape-and-colour Tanimoto, maximised over conformers and stereoisomers |
| GeoDiff PED | `geodiff_distance` | 2D, 3D or concatenated embedding; Euclidean distance through a reverse sigmoid |
| MoLFormer PED | `molformer_distance` | Euclidean distance through a reverse sigmoid |

ECFP4 and Roshambo2 similarities already lie in $[0, 1]$ and are used untransformed.
The embedding distances are unbounded, so each is mapped to a reward by a reverse
sigmoid whose bounds are calibrated to that embedding space; the bounds in the
configurations are the ones reported in the appendix.

## Run

The committed configurations carry repository-relative paths, so they record the
setup without embedding anyone's directory layout. Materialise them for a run with
`--absolute`, then launch from the repository root:

```bash
python generation/configs/make_configs.py --absolute --out_dir /tmp/ped-configs
python generation/external/reinvent/input.py /tmp/ped-configs/reinvent_molformer.json
python generation/external/reinvent/input.py /tmp/ped-configs/libinvent_molformer.json
```

`make_configs.py` takes `--rewards`, `--reinvent_steps`, `--libinvent_steps` and
`--scaffold`, and writes the results under `results/generation/<model>_<reward>/`.
The scaffold for LibINVENT, and the reference ligand, are the ones used in the
paper; `data/reference.sdf` holds the reference conformer that Roshambo2 overlays
against and `data/6O8I.pdb` its crystallographic source.

Each environment is separate: the generators need the REINVENT stack, Roshambo2
needs its own build, and the PED components need the GeoDiff checkpoint from
[`../external/`](../external). See the top-level README.
