# External models

GeoDiff and MoleculeSDE are used as libraries, not as packages, so each has to be
cloned at a fixed commit and patched before its encoders can be imported. `setup.sh`
does that and verifies the result; nothing here needs to be done by hand.

MoLFormer needs no repository, only its HuggingFace checkpoint, and Uni-Mol2 is
installed as `unimol_tools` in its own environment.

## Use

```bash
cd external
PED_CKPT_SRC=/path/to/checkpoints ./setup.sh        # repos + checkpoints
PED_PYTHON=/path/to/env/bin/python ./setup.sh verify
```

| Stage | Does |
|---|---|
| `repos` | clone at the pinned commit, apply the patch |
| `checkpoints` | place the checkpoints and check their SHA-256 |
| `verify` | re-check the SHAs and import both encoders |
| `all` (default) | `repos` then `checkpoints` |

`setup.sh` is idempotent: rerunning it re-pins the commit and reports an
already-patched tree rather than failing. Checkpoints come from `$PED_CKPT_SRC`
(a local directory) when set, otherwise they are downloaded from `$PED_CKPT_URL`.

## Pinned versions

| Repository | Branch | Commit |
|---|---|---|
| [GeoDiff](https://github.com/MinkaiXu/GeoDiff) | `pretrain` | `d76991dd2edfda98631f001e542ecb933d310803` |
| [MoleculeSDE](https://github.com/chao1224/MoleculeSDE) | `main` | `06f84c873ba8196e32b4fd4f7436c2bbc3511915` |

## What the patches change

`geodiff.patch`, 14 files:

- absolute to relative imports (`from utils.chem` becomes `from ..utils.chem`) in the
  nine modules that the encoders pull in, so the clone can be imported as a package
  rather than only run from its own directory;
- four empty `__init__.py` files, for the same reason;
- `utils/chem.py`: `torchvision.transforms.functional.to_tensor` becomes
  `torch.tensor`, dropping an unnecessary torchvision dependency;
- `utils/datasets.py`: conformers are stored as a molblock string instead of a
  deep-copied RDKit molecule, which is what makes packing a 350k-molecule library
  affordable.

`moleculesde.patch`, 6 files:

- absolute to relative imports in `Geom3D/datasets`;
- `Geom3D/models/{molecule_gnn_model,schnet}.py`: drop imports that no longer exist
  in current `torch_geometric` (`GlobalAttention`, `Set2Set`, `add_self_loops`,
  `torch_scatter.scatter_add`, `data.makedirs`). None are used by the 2D or 3D
  encoder, so removing them only widens the version range.

Neither patch touches READMEs, configs or training scripts.

## Checkpoints

| File | Used by |
|---|---|
| `checkpoints/geodiff/model/checkpoints/drugs_default.pt` | GeoDiff, pretrained on GEOM-Drugs |
| `checkpoints/geodiff/model/drugs_default.yml` | its config, read from the parent directory |
| `checkpoints/moleculesde/model_complete.pth` | MoleculeSDE, variance-preserving |

The GeoDiff loader resolves its config by globbing `*.yml` one directory above the
checkpoint, so `model/drugs_default.yml` beside `model/checkpoints/` is required, not
cosmetic.

`checksums.txt` is verified on every run. The cloned repositories and the
checkpoints are gitignored; this folder tracks only `setup.sh`, the two patches,
`checksums.txt` and this file.
