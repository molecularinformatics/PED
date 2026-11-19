# Advancing Ligand-based Virtual Screening and Molecular Generation with Pretrained Molecular Embedding Distance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/molecularinformatics/PED/blob/main/LICENSE)
[![ArXiv](https://img.shields.io/badge/cs.LG-arXiv%3A2604.24474-B31B1B.svg)](https://arxiv.org/abs/2604.24474)
[![ICML](https://img.shields.io/badge/ICML-AI4Science-green
)](https://openreview.net/forum?id=HbfrCipfNl)

The official implementation of Advancing Ligand-based Virtual Screening and Molecular Generation with Pretrained Molecular Embedding Distance (ICML 2026 AI4Science).

![cover](figs/PED-Main.svg)


## Environments

Four environments, split where the dependencies genuinely conflict rather than per
model. Each pipeline's README says which one it needs.

| Environment | Covers | Created with |
|---|---|---|
| `ped_torch` | GeoDiff, MoLFormer, MoleculeSDE, ECFP, every analysis, and the LIT-PCBA pipeline | `conda env create -f environment.yml` |
| `ped-unimol` | Uni-Mol2 only: `unimol_tools` conflicts with the PyG stack the other models need | see below |
| `ped-generation` | REINVENT 3.2, LibINVENT and their reward components | `conda env create -f generation/environment.yml` |
| `synformer` | the SynFormer runs | `conda env create -f generation/synformer/env.yml` |

Boltz-2 needs no environment file; `generation/boltz2/README.md` gives the one
pinned pip command.

```bash
git clone https://github.com/molecularinformatics/PED.git
cd PED

conda env create -f environment.yml      # ped_torch
conda activate ped_torch
```

For Uni-Mol2, in a separate environment:

```bash
conda create -n ped-unimol python=3.10 -y && conda activate ped-unimol
pip install unimol_tools==0.1.5 numba addict
```

`run_embedding.py --python_unimol /path/to/that/env/bin/python` then uses it for the
Uni-Mol2 stages only, with `--model_name unimolv2 --model_size 164m`.

The MoLFormer checkpoint is pinned by revision
(`ibm-research/MoLFormer-XL-both-10pct` at `361063d0ad524ef77cf39b08469f6be770dc550f`).
IBM moved the repository from `ibm/` to `ibm-research/`; both names resolve, and
`--revision` overrides the pin. Because the checkpoint is loaded with
`trust_remote_code=True`, the revision fixes its modelling code as well as its
weights.


## External models

GeoDiff and MoleculeSDE are used as libraries and have to be cloned at a fixed
commit and patched before their encoders can be imported. This is scripted:

```bash
cd external
PED_CKPT_SRC=/path/to/checkpoints ./setup.sh
PED_PYTHON=$(which python) ./setup.sh verify
```

`setup.sh` pins each repository to a recorded commit, applies the patch kept beside
it, places the checkpoints and verifies their SHA-256. It is idempotent. See
[external/](external/) for the pinned commits and for what each patch changes.

MoLFormer needs no repository, only the
[HuggingFace checkpoint](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct);
Uni-Mol2 comes from
[unimol-tools](https://unimol-tools.readthedocs.io/en/latest/train.html#uni-mol-representation).


## Pipelines

Each pipeline has one end-to-end entry point that takes a subset or everything, and
resumes rather than repeats on a rerun.

- [embedding_analysis](embedding_analysis/): library-to-reference distances under five
  representations, and the analyses built on them.
  `python embedding_analysis/run_embedding.py --models all`

- [lit_pcba_pipeline](lit_pcba_pipeline/): LIT-PCBA virtual screening, from raw target
  folders to EF1% tables and figures.
  `python lit_pcba_pipeline/run_lit_pcba.py --targets all`

- [generation](generation/): PED-based rewards for molecular generation with
  REINVENT, LibINVENT and SynFormer, Boltz-2 affinity prediction for the generated
  molecules, and the analysis that turns a run into the reported tables and figures.
  `python generation/configs/make_configs.py --absolute --out_dir /tmp/ped-configs`
  then `python generation/external/reinvent/input.py /tmp/ped-configs/reinvent_molformer.json`

- [external](external/): pinned clones and patches for GeoDiff and MoleculeSDE.
  Set up through `external/setup.sh`, described above.

Pipeline outputs go to `results/`, which is gitignored.


## Data

### In this repository

[`data/`](data/) holds the target files the pipelines read and the result tables
behind the paper's tables and figures.

| File | Notes |
|---|---|
| `lit_pcba_<encoder>_<metric>_per_reference.csv` | EF1% for each LIT-PCBA reference ligand |
| `lit_pcba_<encoder>_<metric>_pooled.csv` | best-pooled EF1% per LIT-PCBA target |
| `baseline_ef1.csv` | the published baselines: 2D ECFP4, ROCS, docking and ROSHAMBO2 |
| `lit-pcba-reference.csv` | the 120 corrected LIT-PCBA reference ligand SMILES |
| `6O8I.pdb` | PDB entry 6O8I, BTK in complex with BMS-986195 |
| `btk_bms.fasta` | the SEQRES of chain A of 6O8I, 269 residues |
| `bmsBTK_6O8I_bioactive.sdf` | the bound ligand of 6O8I with hydrogens added, the reference for 3D scoring |
| `boltz2_top1k_<generator>.csv` | the top 1,000 generated molecules per reward arm, with Boltz-2 predicted affinity |

`<encoder>` is `geodiff` or `molformer`, `<metric>` is `cosine` or `euclidean`,
and `<generator>` is `reinvent`, `libinvent` or `synformer`.

### On Zenodo

[10.5281/zenodo.24374864](https://doi.org/10.5281/zenodo.24374864) carries
everything in `data/` together with the two sets that are too large for the
repository:

| File | Notes |
|---|---|
| `similarity_distance_200k.csv` | the AmpC study set: ROCS and ECFP4 similarity to the reference ligand, and the distance to it in every embedding space tested, as cosine and Euclidean |
| `top5k_<generator>.csv` | the top 5,000 generated molecules per reward arm |

The record's `README.md` documents every file and column.

### External datasets

- [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/). The reference ligand SMILES
  in `data/lit-pcba-reference.csv` are the corrected ones from
  [Data Leakage and Redundancy in the LIT-PCBA Benchmark](https://arxiv.org/abs/2507.21404) after deduplication.
- [AmpC](https://zenodo.org/records/10277786). The 200k study set is a uniform
  sample of it.


## Citation

```
@misc{wa2026advancingligandbasedvirtualscreening,
      title={Advancing Ligand-based Virtual Screening and Molecular Generation with Pretrained Molecular Embedding Distance}, 
      author={Shiyun Wa and Yifei Wang and Simone Sciabola and Ye Wang},
      year={2026},
      eprint={2604.24474},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.24474}, 
}
```


## Acknowledgements

This work builds on pretrained models and generative frameworks released by their
authors. We thank the teams behind them.

Pretrained encoders:

- [GeoDiff](https://github.com/MinkaiXu/GeoDiff), a diffusion model for molecular conformation generation
- [MoLFormer](https://github.com/IBM/molformer), a SMILES Transformer pretrained with masked language modeling
- [MoleculeSDE](https://github.com/chao1224/MoleculeSDE), a joint 2D and 3D multimodal pretraining framework
- [Uni-Mol2](https://github.com/deepmodeling/Uni-Mol), a large-scale 3D molecular foundation model

Generative frameworks:

- [REINVENT](https://github.com/MolecularAI/Reinvent), reinforcement learning over SMILES
- [LibINVENT](https://github.com/MolecularAI/Lib-INVENT), scaffold decoration under a fixed core
- [SynFormer](https://github.com/wenhao-gao/synformer), generation of synthesizable molecules through synthesis programs

Scoring:

- [ROSHAMBO2](https://github.com/molecularinformatics/roshambo2), 3D shape and pharmacophore overlay against a reference ligand
- [Boltz-2](https://github.com/jwohlwend/boltz), co-folding of a protein-ligand complex with an affinity prediction
