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


## Data

For [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/), we used the [preprocessed reference ligand smiles](https://github.com/sievestack/lit-pcba-audit/blob/main/pdb_to_ligand_mapping_with_smiles.csv) provided by [Data Leakage and Redundancy in the LIT-PCBA Benchmark
](https://arxiv.org/abs/2507.21404). Our original and best-pooled EF1% results for each reference ligand are in the [data](data/) folder.

The AmpC dataset can be found [here](https://zenodo.org/records/10277786).


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
  reinforcement learning, the run configurations, and Boltz-2 affinity prediction
  for the generated molecules.

- [external](external/): pinned clones and patches for GeoDiff and MoleculeSDE.

Pipeline outputs go to `results/`, which is gitignored.


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
