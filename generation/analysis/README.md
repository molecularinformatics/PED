# Generation analysis

The tables and figures of the generation experiments, from the run output of
REINVENT, LibINVENT and SynFormer.

## Input

One directory per generator, each with a `scaffold_memory/` folder holding one CSV
per reward, named `<generator>_<reward>.csv`:

```
<results root>/
  reinvent/scaffold_memory/reinvent_{ecfp4,geodiff_2d,geodiff_3d,geodiff_concat,molformer,roshambo2}.csv
  libinvent/scaffold_memory/libinvent_*.csv
  synformer/scaffold_memory/synformer_*.csv
```

`paths.py` resolves the root to `results/generation/` in the repository.
`PED_GENERATION_RESULTS` overrides it, and `PED_GENERATION_FIGS` overrides where
figures are written (`figs/` by default).

## Scripts

| Script | Produces |
|---|---|
| `get_top1k_each.py <generator>` | `top1000_all_methods.csv`, the top 1,000 per reward |
| `table_unique_generations.py` | unique molecules and unique scaffold ratio per arm |
| `table_top_5000.py [generator]` | the top-5,000 scaffold, physicochemical and drug-likeness tables |
| `table_boltz_all.py` | the predicted-pIC50 table |
| `dump_training_curves.py` | `training_curves_avg_score.csv` |
| `summarize_training_curves.py` | the timing and convergence summary |
| `vis_training_dynamics_all.py` | `training_dynamics_all.pdf` |
| `vis_unique_scaffold_score_all.py` | `scaffold_unique_ratio_by_score_all.pdf` |
| `vis_box_boltz_scaffold_all.py` | `top1k_pIC50_boxplot_scaffold_balanced_sample500_all.pdf` |

The Boltz-2 predictions the last two read are produced by
[`../boltz2/`](../boltz2/).

Run them from this directory, with the `ped-generation` environment active:

```bash
cd generation/analysis
python table_top_5000.py reinvent
python vis_training_dynamics_all.py
```
