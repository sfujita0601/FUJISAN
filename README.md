# FUJISAN

FUJISAN (**FU**nctional identity of protein prediction by **J**o**I**ning **S**equence **AN**d structure feature) is a LightGBM-based binary classifier that predicts whether two proteins share an identical function, i.e. catalyze the same enzymatic reaction (same Rhea ID).
For each protein pair it combines full-length sequence similarity (BLAST), structural similarity between domains of AlphaFold2 models (TM-align), and similarity between pocket-forming residues.

## Publication

Fujita S., Terada T. (2024)
Enhanced prediction of protein functional identity through the integration of sequence and structural features.
*Computational and Structural Biotechnology Journal* 23, 4124–4130.
https://doi.org/10.1016/j.csbj.2024.11.028

## Available models

| Model key | Model file | Features | Pocket features | Purpose |
|---|---|---|:---:|---|
| `paper2024` | `model/fujisan_paper2024_evalue_pocket_lgbm.txt` |
| `updated` | `model/fujisan_updated_bitscore_nopocket_lgbm.txt` |

The model keys, file paths, feature names and thresholds are defined in `MODEL_CONFIGS` in [FUJISAN/run_FUJISAN.py](FUJISAN/run_FUJISAN.py).
When a model is loaded, its feature names (stored in the LightGBM model file) are checked against `MODEL_CONFIGS`, and before prediction the input columns are checked against the model's features (names and order). A mismatch raises a `ValueError` instead of producing silent predictions.

## Paper version and updated version

The repository contains two FUJISAN models.

- `paper2024` is the model used in the published study. Use it to reproduce the results of the paper. It is the default of `run_FUJISAN.py`.
- `updated` is a post-publication model with a different feature set (BLAST bit score instead of E-value, no pocket features). As discussed in the paper (Section 3.6), the bit score is intended for applying FUJISAN to proteins that are not in the Swiss-Prot database. This model is **not** the model of the published study and should not be used to reproduce the published results. Its performance has not been reported in the paper, and the test data in this repository cannot be used to evaluate it because it has no `BitScore` column. No validated classification threshold is defined for it (`threshold: None`).

## Input format

Each row is a protein pair. The columns must be exactly the features of the selected model, in this order.

### `paper2024` (7 features)

| Column | Meaning |
|---|---|
| `Full_log10(E_value)` | BLAST E-value between the full-length sequences |
| `Full_SeqID` | BLAST sequence identity between the full-length sequences |
| `Dmn_RMSD` | Cα RMSD (TM-align) between the domain pair with the highest average TM-score |
| `Dmn_SeqID` | Sequence identity in the TM-align alignment of that domain pair |
| `Dmn_TM` | Average TM-score of that domain pair |
| `Pckt_RMSD` | Cα RMSD (PyMOL `super`) between the largest pockets in that domain pair |
| `Pckt_SeqID` | Sequence identity between those pockets |

Definitions follow Section 2.2 of the paper.

### `updated` (5 features)

| Column | Meaning |
|---|---|
| `BitScore` | BLAST bit score between the full-length sequences |
| `Full_SeqID` | same as `paper2024` |
| `Dmn_RMSD` | same as `paper2024` |
| `Dmn_SeqID` | same as `paper2024` |
| `Dmn_TM` | same as `paper2024` |

## Quick start

Requirements: Python ≥ 3.9 and [uv](https://docs.astral.sh/uv/) (versions are pinned in `uv.lock`).

```bash
git clone https://github.com/sfujita0601/FUJISAN.git
cd FUJISAN
uv sync
uv run python FUJISAN/run_FUJISAN.py                 # evaluates the paper2024 model on test/
```

Without uv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
python3 FUJISAN/run_FUJISAN.py
```

Run the script from the repository root (it reads `./test/` and writes `./Analysis/` and `./Figure/`).

Output:
- `Analysis/`: predictions on the test data (`Testdata_y_pred.csv`, `MLdata_LightGBM.csv`), ROC / PR / F1 curve data, and metrics (`LightGBMRes_.csv`)
- `Figure/`: ROC and PR curves (FUJISAN vs. E-value), confusion matrix, feature importance, SHAP plots

### Prediction for your own protein pairs

```python
import pandas as pd
from FUJISAN.run_FUJISAN import MODEL_CONFIGS, load_model, predict_proba

config = MODEL_CONFIGS['paper2024']
model = load_model('paper2024')
X = pd.read_csv('my_pairs.csv')[config['features']]   # columns as in "Input format"
prob = predict_proba(model, X)                          # probability of identical function
pred = (prob >= config['threshold']).astype(int)
```

## Reproducing the published results

`uv run python FUJISAN/run_FUJISAN.py` (model `paper2024`) evaluates the published model on the test sub-dataset in `test/` (10,400 pairs; 5,200 with identical and 5,200 with different functions).

| Metric | Published (Table 1) | This repository (`Analysis/LightGBMRes_.csv`) |
|---|---|
| AUROC | 0.9427 | 0.9427 |
| AUPR | 0.9413 | 0.9413 |
| MCC | 0.7421 | 0.7421 |
| FPR (%) | 13.35 | 13.35 |
| ACC (%) | 87.05 | 87.11 |
| PRE (%) | 87.24 | 86.77 |
| REC (%) | 87.05 | 87.56 |
| F1 | 0.8728 | 0.8716 |

The E-value baseline gives AUROC 0.8890 (published 0.8890) and AUPR 0.7667 (published 0.7767).
The published values are written here for comparison only and are not hard-coded in the code; the figures and result files show values computed at run time.

Classification threshold: `paper2024` uses a fixed threshold of 0.5 (`PAPER2024_THRESHOLD`). This reproduces the published MCC and FPR. The paper states that the threshold was set to maximize the F1 score (Section 2.4), and the small differences in ACC, PRE, REC and F1 above are not explained by the code in this repository. The script also reports the F1-maximizing threshold and F1 on the test data (`F1_max_test_diagnostic`, `Threshold_F1max_test_diagnostic`) as diagnostics only; they are not used for classification because choosing a threshold on test labels would leak them. The validation sub-dataset is not included in this repository, so a validation-optimized threshold cannot be computed here.

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

The source code is released under the [MIT License](LICENSE).

The model files (`model/`) and data (`test/`) are derived from Swiss-Prot (Release 2022_04), the AlphaFold Protein Structure Database (v4) and Rhea. Their use may be subject to the terms of these original data sources, and the MIT License may not cover them. TODO: state the license of the model files and data explicitly.

## Changelog

- 2025-05-29: Added the post-publication model without pocket features (`updated`).
- improve-reproducibility-v1: Renamed the model files (previous names: `LightGBMmodel_best.txt` → `fujisan_paper2024_evalue_pocket_lgbm.txt`, `LightGBMmodel_0102BitScore.txt` → `fujisan_updated_bitscore_nopocket_lgbm.txt`). `run_FUJISAN.py` uses `paper2024` by default again and selects the model with `--model`. In `LightGBMRes_.csv`, `F1` is now the F1 at the fixed threshold (previously the maximum F1 over test-set thresholds, now reported as `F1_max_test_diagnostic`).
