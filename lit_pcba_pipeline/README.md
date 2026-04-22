# LIT-PCBA PED pipeline

This folder contains a pipeline for LIT-PCBA with data preprocessing, embedding extraction, and distance calculation.

## Procedure

For one LIT-PCBA target folder, it will:

- read all `*_ligand.mol2` files as reference ligands
- read `active.smi` and `inactive.smi` and use the first column as SMILES
- create `refs.csv` and `library.csv`
- pack reference and library molecules into GeoDiff shard pkl files
- generate embeddings and save them under separate folders
- compute reference-to-library distances for different embedding modes
- write one final CSV with library metadata and one distance column per reference ligand

## Expected target folder structure

```text
TARGET/
  active.smi
  inactive.smi
  1ABC_ligand.mol2
  2XYZ_ligand.mol2
```

## Final distance CSV columns

The final CSV contains the library metadata plus columns like:

- `{mode}_{dist_type}_dist_3PBL_ligand`
- `{mode}_{dist_type}_dist_min`

`mode = ["2D", "3D", "concat", "molformer"]`

`dist_type = ["cosine", "euclidean"]`

`*_min` is the best pooled distance across all reference ligands for that target.



## Usage

The [main script](run_lit_pcba_pipeline_paral.py) calls all the sub-scripts to conduct the above procedures for one specific LIT-PCBA target

```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=/home/path/PED:$PYTHONPATH

python lit_pcba_pipeline/run_lit_pcba_pipeline_paral.py \
  --target_dir /mnt/depts/dept01/misd/molml/lit_pcba/ADRB2 \
  --out_root /path/lit-pcba/embeddings/ \
  --n_workers ${SLURM_CPUS_PER_TASK} \
  --devices cuda:0 \
  --run_molformer \
  --run_geodiff
```

## Output structure

```text
out_root/
  ADRB2/
    prepared/
      refs.csv
      library.csv
      prep_meta.json
    refs/
      packed/
      embeddings/
        2D/
          emb.npy
          smiles.npy
        3D/
        molformer/
    library/
      packed/
      embeddings/
        2D/
          emb.npy
          smiles.npy
        3D/
        molformer/
    distance/
      library_reference_{model}_distances.csv
```

