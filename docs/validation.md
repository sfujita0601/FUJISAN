# Validation of the feature-generation pipeline (`fujisan`)

Checks done for release 0.2.0 (2026-10-29) with the `updated` model retrained on raw BLAST bit scores
(`model/fujisan_updated_blastbitscore_nopocket_lgbm.txt`). They compare the features computed by the pipeline with
the historical FUJISAN feature values in `test/same.tsv` and `test/diff.tsv`, and measure how differences affect the
probability of the `updated` model. Only summary statistics are recorded here; downloaded AlphaFold DB files and
per-pair tables are not part of the repository.

## Software and data

| Item | Version / source |
|---|---|
| BLAST+ | 2.16.0+ (pipeline runs); 2.17.0+ for the version comparison |
| TM-align | 20210520 |
| pae_to_domains | vendored, upstream commit `f407c60` (igraph Leiden, `pae_power=1`, `pae_cutoff=5`, `resolution=1`, seed 0) |
| python-igraph / LightGBM / gemmi | 1.0.0 / 4.5.0 / 0.7.5 (Python 3.9.20) |
| Model | `model/fujisan_updated_blastbitscore_nopocket_lgbm.txt` (unchanged) |
| Structures | AFDB v4 models and PAE (local copies); AFDB v6 (current API) where stated |
| Reference features | `test/same.tsv`, `test/diff.tsv` (historical FUJISAN features; no `BitScore` column) |
| BLAST features | `blastp` of each pair (first protein as query), `-evalue 10 -max_hsps 1`, no `-dbsize` |

Because the reference tables have no `BitScore`, every probability comparison below uses the pipeline `BitScore`
for both sides; the comparisons measure the effect of `Full_SeqID` and the three domain features.

The model file was saved by a newer LightGBM than the 4.5.0 pinned in `uv.lock` (it warns about the unknown
parameter `bagging_by_query`). Predictions for 2,000 random feature vectors were identical with LightGBM 4.5.0 and
4.6.0 (maximum difference 0).

## 1. Historical feature reproduction (200 pairs)

Random sample (fixed seed) of 100 `same` + 100 `diff` pairs whose two proteins have local AFDB v4 files.
Features were computed for all 200 pairs.

| Feature agreement with the historical values | Pairs |
|---|---|
| `Dmn_TM`, `Dmn_RMSD`, `Dmn_SeqID` all identical (tolerance 0.001 / 0.006 Å / 0.001) | 184 / 200 (same 90, diff 94) |
| `Full_SeqID` identical (tolerance 0.01) | 190 / 200 |

Tolerances correspond to the precision of the historical tables. Of the 16 pairs with differing domain features,
14 differ in `Dmn_TM`; in 13 of these the pipeline value is higher than the historical one. In an earlier 40-pair sample,
running TM-align on the domain files kept from the original feature generation gave the pipeline's values for both
non-matching pairs, i.e. the historical values came from a different domain split. The original feature generation
used stochastic Leiden clustering without a fixed seed; the pipeline fixes seed 0, so some historical domain
assignments differ. The historical `Full_SeqID` values come from database searches (not single-pair BLAST); in
some of the differing pairs the E-values differ by orders of magnitude, which points to differences in the search
context, BLAST version or sequence version that were not investigated further.

## 2. Prediction sensitivity to historical feature differences (same 200 pairs, updated model)

| Statistic | Value |
|---|---|
| Pearson r | 0.9972 |
| Spearman ρ | 0.9960 |
| \|Δp\| mean / median / max | 0.0065 / 0.0000 / 0.240 |
| pairs with \|Δp\| > 0.05 / > 0.10 / > 0.20 | 10 / 5 / 2 |

Of the 10 pairs with \|Δp\| > 0.05, 8 have differing domain features and 2 have identical domain features but a
different `Full_SeqID`. Largest: Q2FG85–Q8DAN4 (same; `Dmn_TM` 0.696 → 0.926, p 0.639 → 0.880),
Q5HAU5–Q21QP9 (same; 0.941 → 0.955, p 0.696 → 0.906), Q99XW8–Q5KWZ9 (same; 0.725 → 0.966, p 0.809 → 0.962).

## 3. AFDB v4 vs current AFDB (v6), 40 pairs, updated model

Same pipeline; structures from local AFDB v4 files vs AFDB v6 downloaded through the API.

| Feature | \|difference\| mean / median / max | identical |
|---|---|---|
| `BitScore` | 0.17 / 0 / 6.9 | 39 / 40 |
| `Full_SeqID` | 0.0003 / 0 / 0.011 | 39 / 40 |
| `Dmn_TM` | 0.0037 / 0.0017 / 0.0405 | 0 / 40 |
| `Dmn_RMSD` | 0.051 / 0.040 / 0.29 Å | 7 / 40 |
| `Dmn_SeqID` | 0.0020 / 0.0005 / 0.0150 | 20 / 40 |
| FUJISAN probability | 0.021 / 0.004 / 0.140 (Pearson r 0.994, Spearman ρ 0.987) | |

Pairs with \|Δp\| > 0.05 / > 0.10 / > 0.20: 6 / 2 / 0. The only pair whose sequences changed between the versions
accounts for the BLAST differences.

## 4. BLAST semantics (40 pairs)

| Comparison | Result |
|---|---|
| BLAST+ 2.16.0 vs 2.17.0 (`-subject`, one-sequence DB, and a 227k-sequence DB search) | identical bit score, pident and alignment length for 40 / 40 |
| `blastp -subject` vs temporary one-sequence database (`-db`) | identical for 40 / 40 |
| single-pair BLAST vs search of a 227k-sequence database | identical for 39 / 40; P0A6V0–B0SFJ4: bit score 203 / pident 30.242 / length 496 vs 205 / 31.020 / 490 |
| single-pair `Full_SeqID` vs historical | 39 / 40 (P0A6V0–B0SFJ4: 30.242 vs 31.02) |

The updated model was trained on bit scores of single-pair `blastp` runs (BLAST+ 2.17.0, default parameters,
`-max_hsps 1`, first protein as query). The pipeline therefore computes `BitScore` and `Full_SeqID` for each pair
alone in all commands; in `search` / `batch` the database search only selects the candidates. `pair` and `search`
give identical values for the same pair. No `-dbsize` is passed by default.

End-to-end examples (AFDB v6, current API; updated model): P00509–Q22067 `BitScore` 289.0, `Full_SeqID` 39.394,
`Dmn_TM` 0.97069, p = 0.910; P0A6V0–B0SFJ4 203.0, 30.242, 0.89631, p = 0.255.

## 5. Package installation

A wheel built with `uv build --wheel` contains `FUJISAN/model/` (both models), the vendored
`FUJISAN/vendor/pae_to_domains/` (with `LICENSE` and attribution `README.md`), the package `LICENSE` and the
`fujisan` entry point. Installed into a fresh Python 3.9 virtual environment and run from a directory outside the
repository: `fujisan --help`, `fujisan pair|search|batch --help`, `fujisan cache-info`,
`load_model('updated').feature_name()` (model read from `site-packages/FUJISAN/model`) and an AFDB end-to-end
`fujisan pair` worked.

## Known limitations of these checks

- Sample sizes are small (200 and 40 pairs); the numbers above are observations on these samples.
- `BitScore` cannot be compared with historical values (not in the reference tables).
- The exact original BLAST databases and versions of the historical `Full_SeqID` values are not available.
- The reference tables are test-set pairs of the `paper2024` data; the training set of the `updated` model is not in
  the repository.
