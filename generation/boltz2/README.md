# Boltz-2 binding-affinity prediction

Predicted BTK affinities for the generated molecules. Boltz-2 co-folds the BTK kinase domain with one ligand and returns an
affinity score; the molecules scored in the paper are deposited in
[`data/`](../../data/) as `boltz2_top1k_<model>.csv`.

## Install

```bash
pip install "boltz[cuda]==2.2.1"     # released 2025-09-08
```

The weights (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, ~4 GB) download on first use
into `~/.boltz`, or `$BOLTZ_CACHE` if set. Prediction needs one GPU.

## Target

`btk_6O8I.yaml` is the input template: the BTK kinase domain from chain A of PDB
6O8I with one ligand, and an `affinity` property on the ligand chain.

The protein sequence is the 269-residue SEQRES of chain A, identical to
[`data/btk_bms.fasta`](../../data/btk_bms.fasta). It is taken from SEQRES rather
than from the ATOM records because residues 434-441 are unresolved in the crystal
(REMARK 465), so an ATOM-derived sequence would be missing an 8-residue stretch.
The 6O8I coordinates themselves are not used: Boltz-2 predicts the complex from
sequence and SMILES. Co-folding is non-covalent, so proximity between an acrylamide
warhead and Cys481 is a property of the prediction and not a constraint.

The reference ligand BMS-986195 is [`data/bmsBTK_6O8I_bioactive.sdf`](../../data/bmsBTK_6O8I_bioactive.sdf).

## Run

```bash
python run_boltz2.py \
    --input ../../data/boltz2_top1k_reinvent.csv \
    --out_dir results/reinvent \
    --smiles_col SMILES --name_col Name
```

Three steps, each skipped when its output is already there, so an interrupted run
resumes:

1. **One alignment for the target.** `btk_6O8I.a3m` in this directory is the
   alignment used here, 13,676 sequences from the ColabFold MMseqs2 server, and is
   used by default so that a rerun needs no network access and scores every
   molecule against the same alignment. `--refetch_msa` queries the server instead
   and writes to `<out_dir>/msa/`. Boltz queries the server once per input file, so
   one shared alignment is the difference between one query and one per molecule.
2. **One input per molecule** in `<out_dir>/inputs/`, named by `--name_col`, which
   must be unique and filename-safe.
3. **`boltz predict`**, then the affinities are collected into
   `<out_dir>/affinity.csv`.

Defaults are the Boltz-2 defaults apart from `--seed 42`, which Boltz leaves unset.
`--diffusion_samples_affinity` is 5, `--devices` is 1. `--limit N` scores the first
N molecules, and `--collect_only` rebuilds the CSV from finished predictions.

## Output

`affinity.csv` carries one row per input molecule:

| Column | Meaning |
|---|---|
| `affinity_pred_value` | Boltz-2 affinity head output |
| `affinity_probability_binary` | Boltz-2 binary binding probability |
| `predicted_pIC50` | `8.184 - 1.364 * affinity_pred_value`, the transform used in the paper |

A molecule whose prediction fails leaves these empty rather than dropping the row.
