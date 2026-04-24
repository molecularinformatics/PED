# SMILES–Reference Similarity / Distance Pipeline

This pipeline computes similarity or distance between a library of SMILES and a single reference SMILES using multiple molecular representations:

- GeoDiff
- MoLFormer
- UniMol
- MoleculeSDE
- ECFP (Morgan fingerprint with Tanimoto similarity)


## Overview

**Input:**
- A CSV file with a `smiles` column (or specified column)
- One reference SMILES string

**Output:**
- A CSV with:
  - `smiles`
  - GeoDiff distance columns:
    - `geodiff_2D_{dist_type}`
    - `geodiff_3D_{dist_type}`
    - `geodiff_concat_{dist_type}`
    - `geodiff_concat_norm_{dist_type}`
  - MoLFormer distance columns:
    - `molformer_{dist_type}`
  - UniMol distance columns:
    - `unimol_{dist_type}`
  - MoleculeSDE distsance columns:
    - `sde_2D_{dist_type}`
    - `sde_3D_{dist_type}`
    - `sde_concat_{dist_type}`
    - `sde_concat_norm_{dist_type}`
  - ECFP similarity column:
    - `ecfp_tanimoto`


## Pipeline structure

The pipeline consists of 4 steps:

### 1. Prepare inputs
- Clean and canonicalize SMILES
- Save:
  - `refs.csv` (single reference)
  - `library.csv`


### 2. Embedding generation

#### GeoDiff
- Reference: serial
- Library: parallel (multi-process + multi-GPU)
- Outputs:
  - 2D embeddings
  - 3D embeddings

#### MoLFormer
- Reference: serial
- Library: parallel (multi-GPU)
- Outputs:
  - MoLFormer embeddings

#### UniMol
- Reference: serial
- Library: parallel (multi-GPU, but recommend serial for stability)
- Outputs:
  - UniMol embeddings

#### MoleculeSDE
- Reference: serial
- Library: parallel (multi-process + multi-GPU)
- Outputs:
  - 2D embeddings
  - 3D embeddings

#### ECFP
- CPU parallel computation
- Morgan fingerprint (radius=2, nbits=2048)
- Outputs:
  - `emb.npy` (masked feature vectors: invalid SMILES → zero vector)
  - `smiles.npy`



### 3. Similarity / Distance computation

#### GeoDiff and MoleculeSDE modes
- `2D`
- `3D`
- `concat` = concat(2D, 3D)
- `concat_norm` = concat(norm(2D), norm(3D))


#### ECFP metric
- Tanimoto similarity


### 4. Output
- One CSV combining a specific model's computed metrics
- Use the [script](merge_dist.py) to merge all models' metrics together with a SMILES column

## Usage


```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=/home/path/PED:$PYTHONPATH

python smiles_reference_distance_pipeline/run_smiles_reference_pipeline.py \
  --library_csv /path/200k_input.csv \
  --reference_smiles "OC(=O)C1=C(C=CS1)S(=O)(=O)NC1=CC=C(Cl)C=C1" \
  --smiles_col smiles \
  --out_root /path/embeddings/ \
  --job_name 200k_emb \
  --ref_confs 1 \
  --n_workers ${SLURM_CPUS_PER_TASK} \
  --devices cuda:0 \
  --unimol_model_size 164m \
  --run_geodiff \
  --run_molformer \
  --run_unimol \
  --run_sde \
  --run_ecfp \

```

## Output Structure

```text
output_root/
  job_name/
    prepared/
      refs.csv
      library.csv

    refs/
      embeddings/
        geodiff_2D/
        geodiff_3D/
        molformer/
        unimol/
        sde_2D/
        sde_3D/
        ecfp/

    library/
      embeddings/
        geodiff_2D/
        geodiff_3D/
        molformer/
        unimol/
        sde_2D/
        sde_3D/
        ecfp/

    distance/
      smiles_reference_*_distances.csv
```