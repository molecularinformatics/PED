# Embedding analysis

Distances between a library of SMILES and one reference ligand, under five
molecular representations, plus the analyses built on them.

```
run_embedding.py     end-to-end entry point
scripts/             prepare -> embed -> distance -> merge
analysis/            correlation, embedding geometry, scaling, modality
analysis/shuffle/    sensitivity to SMILES canonicalization and atom-order
```

| Model | Interface |
|---|---|
| GeoDiff | repository + checkpoint, see [`external/`](../external) |
| MoLFormer | HuggingFace checkpoint only, no repository needed |
| MoleculeSDE | repository + checkpoint, see [`external/`](../external) |
| Uni-Mol2 | `unimol_tools`, **separate environment** |
| ECFP | RDKit Morgan fingerprint with Tanimoto similarity |

## Run it

```bash
python run_embedding.py \
  --library_csv       /path/to/library.csv \
  --reference_smiles  "CC(=O)Oc1ccccc1C(=O)O" \
  --out_root          ../results/ampc200k \
  --models all
```

`--models` takes `geodiff molformer sde unimol ecfp` or `all`; `--stages`
restricts the run to `prepare embed distance merge`. Each stage writes a stamp
under `<out_root>/<job>/.stamps/`, so a rerun resumes rather than repeats; pass
`--force` to redo stamped stages.

Uni-Mol2 does not share an environment with the others, because `unimol_tools`
conflicts with the PyG stack GeoDiff and MoleculeSDE need. Point
`--python_unimol` at that environment's interpreter and the runner will use it for
the Uni-Mol2 stages only.

## Outputs

```
<out_root>/<job>/prepared/{refs,library}.csv
<out_root>/<job>/{refs,library}/embeddings/<model>/{emb,smiles}.npy
<out_root>/<job>/distance/smiles_reference_{model}_distances.csv
<out_root>/<job>/similarity_distance.csv
```

Distance columns are `{model}_{dist_type}` for single-encoder models and
`{model}_{2D,3D,concat,concat_norm}_{dist_type}` for the dual-encoder ones;
`dist_type` is `cosine` or `euclidean`. ECFP is reported as `ecfp_tanimoto`.

The merge step joins every per-model distance table onto a reference similarity
table carrying the ROCS and ECFP4 columns; pass it with `--base_csv`.

## Analysis

| Script | Question it answers |
|---|---|
| `analysis/vis_corr_matrix.py` | how each embedding distance correlates with the traditional similarities |
| `analysis/vis_dist_rocs.py` | binned distance against ROCS shape, color and combination |
| `analysis/shuffle/` | how far embeddings move under SMILES canonicalization and atom-order randomization |

Distances are computed on the embeddings exactly as the pretrained models emit
them. Cosine distance normalizes each vector, and the concatenated mode
L2-normalizes each branch before concatenation; both are computed from a single
molecule.
