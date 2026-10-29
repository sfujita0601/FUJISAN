# FUJISAN

FUJISAN (**FU**nctional identity of protein prediction by **J**o**I**ning **S**equence **AN**d structure feature) is a LightGBM-based binary classifier that predicts whether two proteins share an identical function, i.e. catalyze the same enzymatic reaction (same Rhea ID).
For each protein pair it combines full-length sequence similarity (BLAST) with structural similarity between domains of AlphaFold models (TM-align); the published model also uses similarity between pocket-forming residues.

## Publication

Fujita S., Terada T. (2024)
Enhanced prediction of protein functional identity through the integration of sequence and structural features.
*Computational and Structural Biotechnology Journal* 23, 4124–4130.
https://doi.org/10.1016/j.csbj.2024.11.028

## Documentation

- [Feature-generation pipeline](docs/feature_generation.md): the `fujisan` command (`pair`, `search`, `batch`,
  `cache-info`), BLAST settings, AlphaFold DB retrieval, domain segmentation and TM-align, cache and storage,
  output files and metadata, and how to use the models with precomputed features.
- [Validation and reproducibility](docs/validation.md): reproduction of historical features, sensitivity of the
  predictions, AlphaFold DB v4 vs v6, BLAST semantics and the package-installation test.
- [Reproducing the published FUJISAN results](docs/paper_reproduction.md): the `paper2024` model, the test
  dataset, published vs repository metrics and threshold handling.

## Available models

| Model key | Model file | Features | Pocket features | Purpose |
|---|---|---|:---:|---|
| `paper2024` | `model/fujisan_paper2024_evalue_pocket_lgbm.txt` | 7: log10 E-value + full-length sequence identity + domain (3) + pocket (2) | Yes | Reproduction of the published FUJISAN model |
| `updated` | `model/fujisan_updated_blastbitscore_nopocket_lgbm.txt` | 5: BLAST bit score + full-length sequence identity + domain (3) | No | Post-publication model (added 2025-05-29, retrained 2026-10-29) |

- **`paper2024`** is the model used in the published study (E-value, sequence, domain and pocket features). Use it
  to reproduce the results of the paper ([docs/paper_reproduction.md](docs/paper_reproduction.md)). It is the
  default of `run_FUJISAN.py`.
- **`updated`** is a post-publication model with a different feature set: the raw BLAST bit score (`BitScore`)
  instead of the E-value, `Full_SeqID`, the three domain features, and no pocket features. As discussed in the paper
  (Section 3.6), the bit score is intended for applying FUJISAN to proteins that are not in the Swiss-Prot database.
  This model is **not** the model of the published study and should not be used to reproduce the published results.
  Its performance has not been reported in the paper, and the test data in this repository cannot be used to evaluate
  it because it has no `BitScore` column; for reference, on the same 10,400 test pairs as `paper2024` (with BLAST bit
  scores added) it gives AUROC 0.9440 and AUPR 0.9434. **No validated classification threshold is
  defined for it** (`threshold: None`); only probabilities are reported. The feature-generation CLI (`fujisan`)
  targets this model.

`BitScore` is the raw bit score reported by `blastp` (BLAST+ 2.17.0, default scoring parameters, `-max_hsps 1`) for
the best HSP of the full-length sequence pair, with the first protein as the query; `Full_SeqID` is the `pident` of
the same HSP. The bit score itself is not normalised by the database size, but the search context can change which
alignment (HSP) BLAST reports, so the pipeline recomputes both features with pairwise BLAST for each protein pair;
the database searches of `search` and `batch` are used only to identify candidate targets
([details](docs/feature_generation.md#blast-settings-shared-by-all-commands)).

The model keys, file paths, feature names and thresholds are defined in `MODEL_CONFIGS` in [FUJISAN/models.py](FUJISAN/models.py) (also imported by `FUJISAN/run_FUJISAN.py`).
When a model is loaded, its feature names (stored in the LightGBM model file) are checked against `MODEL_CONFIGS`, and before prediction the input columns are checked against the model's features (names and order). A mismatch raises a `ValueError` instead of producing silent predictions.

## Installation

Requirements: Python ≥ 3.9 and [uv](https://docs.astral.sh/uv/) (versions are pinned in `uv.lock`).

```bash
git clone https://github.com/sfujita0601/FUJISAN.git
cd FUJISAN
uv sync                       # then: uv run fujisan ...
```

Or install the package (the models are included, and `fujisan` then works from any directory):

```bash
python3 -m pip install .
fujisan --help
```

The feature-generation CLI also needs **BLAST+** (`blastp`, `makeblastdb`) and **TM-align** (`TMalign`), which you
install yourself; `search` and `batch` need a protein BLAST database that you prepare (e.g. Swiss-Prot). See
[docs/feature_generation.md](docs/feature_generation.md#requirements).

## Quick start

Features and FUJISAN probability (`updated` model) for one protein pair from the AlphaFold Protein Structure Database:

```bash
uv run fujisan pair \
  --afdb-id1 P00509 \
  --afdb-id2 Q22067 \
  --output pair.tsv
```

Search one query against your BLAST database and rank the hits:

```bash
uv run fujisan search \
  --query-afdb-id P00509 \
  --blast-db /path/to/swissprot \
  --max-hits 100 \
  --output results/
```

Several queries (one output directory each):

```bash
uv run fujisan batch \
  --queries queries.tsv \
  --blast-db /path/to/swissprot \
  --max-hits 100 \
  --output batch_results/
```

Local structure / PAE files, `queries.tsv` format, output files and all options:
[docs/feature_generation.md](docs/feature_generation.md).
For reproduction of the published analysis, see [docs/paper_reproduction.md](docs/paper_reproduction.md).

## Pipeline overview

```
protein pair
    │
    ├─ BLAST (pairwise, best HSP) ──────────────► BitScore, Full_SeqID
    │
    └─ AlphaFold DB structure + PAE
          ↓
       domains (pae_to_domains, pLDDT ≥ 70, ≥ 30 residues)
          ↓
       TM-align of all domain pairs (-a T)
          ↓
       Dmn_TM, Dmn_RMSD, Dmn_SeqID (best domain pair)
          ↓
       updated FUJISAN model ──► FUJISAN_probability
```

## Key limitations

- The feature-generation CLI is for the `updated` 5-feature model only; it does not generate the pocket features of
  the `paper2024` model.
- The `updated` model is not the published model, and it has no validated binary classification threshold.
- Model development used AlphaFold DB v4, whereas automatic retrieval uses the current AFDB release, and the
  historical features used stochastic domain segmentation without a fixed seed (the pipeline fixes `seed=0`); both
  can change individual FUJISAN scores ([docs/validation.md](docs/validation.md)).
- Large searches and batch analyses download many AlphaFold DB files and can take long and use substantial disk
  space ([cache and storage](docs/feature_generation.md#cache-and-storage-requirements)).

Full list: [docs/feature_generation.md](docs/feature_generation.md#limitations).

## Citation

If you use FUJISAN, please cite:

```bibtex
@article{fujita2024fujisan,
  author  = {Fujita, Suguru and Terada, Tohru},
  title   = {Enhanced prediction of protein functional identity through the integration of sequence and structural features},
  journal = {Computational and Structural Biotechnology Journal},
  volume  = {23},
  pages   = {4124--4130},
  year    = {2024},
  doi     = {10.1016/j.csbj.2024.11.028}
}
```

GitHub's "Cite this repository" uses [CITATION.cff](CITATION.cff).

## License

Copyright (c) 2024 Computational Biology Lab., The University of Tokyo.

- Source code: [MIT License](LICENSE).
- Model files (`model/`) and data (`test/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  They are derived from the following resources, which are also distributed under CC BY 4.0:
  - UniProtKB/Swiss-Prot (Release 2022_04), https://www.uniprot.org
  - AlphaFold Protein Structure Database (v4), https://alphafold.ebi.ac.uk
  - Rhea, https://www.rhea-db.org

  Please cite the original resources as well as the FUJISAN paper when using them.

## Changelog

- 0.2.0: Added the `fujisan` command (`pair`, `search`, `batch`, `cache-info`) that generates the five features of the `updated` model (retrained on raw BLAST bit scores) and predicts with it. The model registry moved from `run_FUJISAN.py` to `FUJISAN/models.py` (imported by `run_FUJISAN.py`; behaviour unchanged). The model files are included in the installed package. Tests in `tests/` (`uv run pytest`); validation in [docs/validation.md](docs/validation.md). Documentation reorganized into `docs/`.

- 2025-05-29: Added the post-publication model without pocket features (`updated`).
- improve-reproducibility-v1: Renamed the model files (previous names: `LightGBMmodel_best.txt` → `fujisan_paper2024_evalue_pocket_lgbm.txt`, `LightGBMmodel_0102BitScore.txt` → `fujisan_updated_bitscore_nopocket_lgbm.txt`). `run_FUJISAN.py` uses `paper2024` by default again and selects the model with `--model`. In `LightGBMRes_.csv`, `F1` is now the F1 at the fixed threshold (previously the maximum F1 over test-set thresholds, now reported as `F1_max_test_diagnostic`).
- 2026-10-29: Retrained `updated` with BLAST bit scores (`fujisan_updated_blastbitscore_nopocket_lgbm.txt`). The previous `updated` model (`fujisan_updated_bitscore_nopocket_lgbm.txt`, still in the git history) was trained on −log2(E-value) − log2(query length) instead of the BLAST bit score, so it gave wrong scores when given BLAST bit scores. Hyperparameters were tuned with stratified, shuffled 5-fold cross-validation on the training data only; the test data was not used for early stopping.
