# Reproducing the published FUJISAN results

This page describes how to reproduce the results of the published study with the **`paper2024`** model:

Fujita S., Terada T. (2024). Enhanced prediction of protein functional identity through the integration of
sequence and structural features. *Computational and Structural Biotechnology Journal* 23, 4124–4130.
https://doi.org/10.1016/j.csbj.2024.11.028

## The `paper2024` model

`paper2024` (`model/fujisan_paper2024_evalue_pocket_lgbm.txt`) is the model used in the published study. It uses
7 features: log10 E-value and sequence identity of the full-length sequences, three domain features and two pocket
features (see [feature_generation.md](feature_generation.md#paper2024-7-features) for the column definitions).
The `updated` model is a different, post-publication model and should not be used to reproduce the published
results. The feature-generation CLI (`fujisan`) does not compute the pocket features of `paper2024`.

## Running the evaluation

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
`paper2024` is the default of `run_FUJISAN.py`; the model is selected with `--model`.

Output:
- `Analysis/`: predictions on the test data (`Testdata_y_pred.csv`, `MLdata_LightGBM.csv`), ROC / PR / F1 curve data, and metrics (`LightGBMRes_.csv`)
- `Figure/`: ROC and PR curves (FUJISAN vs. E-value), confusion matrix, feature importance, SHAP plots

## Published vs repository metrics

`uv run python FUJISAN/run_FUJISAN.py` (model `paper2024`) evaluates the published model on the test sub-dataset in `test/` (10,400 pairs; 5,200 with identical and 5,200 with different functions).

| Metric | Published (Table 1) | This repository (`Analysis/LightGBMRes_.csv`) |
|---|---|---|
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

## Classification threshold

`paper2024` uses a fixed threshold of 0.5 (`PAPER2024_THRESHOLD`). This reproduces the published MCC and FPR. The paper states that the threshold was set to maximize the F1 score (Section 2.4), and the small differences in ACC, PRE, REC and F1 above are not explained by the code in this repository. The script also reports the F1-maximizing threshold and F1 on the test data (`F1_max_test_diagnostic`, `Threshold_F1max_test_diagnostic`) as diagnostics only; they are not used for classification because choosing a threshold on test labels would leak them. The validation sub-dataset is not included in this repository, so a validation-optimized threshold cannot be computed here.
