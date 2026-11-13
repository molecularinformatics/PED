# LIT-PCBA PED pipeline

Virtual screening on [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/) with
pretrained embedding distance, from raw target folders to EF1% tables and figures.

```
run_lit_pcba.py      end-to-end entry point
scripts/             prepare -> embed -> distance
analysis/            EF1% aggregation, tables, figures, statistics
```

## Run it

```bash
python run_lit_pcba.py \
  --lit_pcba_root /path/to/LIT-PCBA \
  --out_root      /path/to/output \
  --targets all \
  --models molformer geodiff
```

`--targets` takes target names or `all` (the 15 benchmark targets). `--models` takes
`molformer`, `geodiff` or both. `--stages` restricts the run to
`prepare embed distance ef1 stability`. Each stage writes a stamp under
`<out_root>/<TARGET>/.stamps/`, so a rerun resumes rather than repeats; pass
`--force` to redo stamped stages.

GeoDiff needs its repository and checkpoint; see [`external/`](../external) for the
pinned setup. MoLFormer needs only the HuggingFace checkpoint.

## Reference ligands

References default to [`data/lit-pcba-reference.csv`](../data/lit-pcba-reference.csv):
120 ligands over 15 targets, with the corrected canonical SMILES from the
[LIT-PCBA audit](https://github.com/sievestack/lit-pcba-audit) of
[Huang et al.](https://arxiv.org/abs/2507.21404). Pass `--refs_from_mol2` to
`prepare_lit_pcba_inputs.py` to parse the target folder's `.mol2` files instead.

Ligands that appear under several PDB entries with identical SMILES are collapsed to
one reference. Without that, the reference merge expands, duplicate output columns
overwrite one another, and the pooled minimum is taken over more columns than the
file reports.

## Outputs

```
<out_root>/<TARGET>/prepared/{refs,library}.csv, prep_meta.json
<out_root>/<TARGET>/{refs,library}/embeddings/...
<out_root>/<TARGET>/distance/library_reference_{model}_distances.csv
<out_root>/ef1/lit_pcba_{model}_{cosine,euclidean}_{per_reference,pooled}.csv
<out_root>/ef1/stability_{cosine,euclidean}.csv, loo_stability_*.pdf
```

Distance CSVs carry the library metadata plus one column per reference,
`{mode}_{dist_type}_dist_{pdb_id}_ligand`, and the pooled
`{mode}_{dist_type}_dist_min`, where `mode` is `2D`, `3D`, `concat` or `molformer`.

## Analysis

| Script | Produces |
|---|---|
| `analysis/ef1.py` | per-reference and pooled EF1%, plus the EF1% granularity table |
| `analysis/stability.py` | per-target differences, leave-one-target-out, paired tests, the 2x2 figure |
| `analysis/visual_EF_*.py` | the per-target EF1% boxplots |
| `analysis/table_EF_*.py` | the LaTeX comparison tables |

`stability.py` compares a PED variant against the published baselines in
[`data/baseline_ef1.csv`](../data/baseline_ef1.csv) (2D ECFP4, ROCS and docking from
the LIT-PCBA paper; ROSHAMBO2 from its own). It reports the mean per-target
difference with a percentile bootstrap interval over targets, the number of targets
won and lost, a Wilcoxon signed-rank test with Holm correction, and which single
target removals reverse the sign. With 15 paired targets the test resolves only
large and consistent effects, so a non-significant result means the data do not
separate the two measures, not that they are equivalent.
