# Advancing Ligand-based Virtual Screening and Molecular Generation with Pretrained Molecular Embedding Distance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/molecularinformatics/PED/blob/main/LICENSE)
[![ArXiv](https://img.shields.io/badge/cs.LG-arXiv%3A2604.24474-B31B1B.svg)](https://arxiv.org/abs/2604.24474)
[![ICML](https://img.shields.io/badge/ICML-AI4Science-green
)](https://ai4sciencecommunity.github.io/icml26.html)

The official implementation of Advancing Ligand-based Virtual Screening and Molecular Generation with Pretrained Molecular Embedding Distance (XXX 2026 ****).

![cover](figs/PED-Main.svg)


## Environments

Dependencies for MoLFormer and GeoDiff:

```bash
# Clone the environment
git clone https://github.com/molecularinformatics/PED.git
cd PED

conda env create -f environment.yml
conda activate ped_torch
```

To use MoleculeSDE, futher install:

```bash
pip install ase
pip install ogb==1.2.1
```

To use UniMol without PyG and torch conflicts, install from source:

```bash
git clone https://github.com/deepmodeling/Uni-Mol.git
cd Uni-Mol/unimol_tools

pip install --no-deps -e .
pip install numba addict
```


## Import models

[GeoDiff](https://github.com/MinkaiXu/GeoDiff/tree/pretrain) and [MoleculeSDE](https://github.com/chao1224/MoleculeSDE) should be cloned and put into this directory manually.

```bash
git clone https://github.com/MinkaiXu/GeoDiff.git
git checkout pretrain

git clone https://github.com/chao1224/MoleculeSDE.git
```

Then, add lacked `__init__.py` files to each folder, and modify the internal module import via a relative path format. For example:

```python
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
```

Particularly, in `geodiff/utils/chem.py`, comment out `from torchvision.transforms.functional import to_tensor`, and replace `to_tensor()` with `torch.tensor()` in the script.

Also download model weights and put in the correct path.

UniMol and MoLFormer can be used via the [unimol-tools](https://unimol-tools.readthedocs.io/en/latest/train.html#uni-mol-representation) and [huggingface](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct).


## Data

For [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/), we used the [preprocessed reference ligand smiles]((https://github.com/sievestack/lit-pcba-audit/blob/main/pdb_to_ligand_mapping_with_smiles.csv)) provided by [Data Leakage and Redundancy in the LIT-PCBA Benchmark
](https://arxiv.org/abs/2507.21404). Our original and best-pooled EF1% results for each reference ligand are in the [data](data/) folder.

The AmpC dataset can be found [here](https://zenodo.org/records/10277786).


## Pipelines

- [smiles_reference_distance_pipeline](smiles_reference_distance_pipeline/): Library molecules—reference ligand distance pipeline

- [lit_pcba_pipeline](lit_pcba_pipeline/): LIT-PCBA PED pipeline

- [canonical_shuffle_test](canonical_shuffle_test/): SMILES representation's impact on different pretrained embeddings

- [scoring_function](scoring_function/): PED-based scoring function for molecular generation with reinforcement learning


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
