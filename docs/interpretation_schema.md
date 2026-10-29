# Interpretation records (`fujisan.interpretation`, schema 1.0.0)

`fujisan explain` and the GUI write the same records, built by `FUJISAN/interpretation/`:
JSON Lines, one object per protein pair. Each line holds its own provenance, so files can be concatenated,
filtered and joined with other per-pair tables line by line.

## Scope

- **Phase 1 (implemented):** per-pair feature contributions (SHAP values) of the model input features.
- **Phase 2 (not connected):** domain pair and residue correspondence from the server-side outputs
  `selected_domain_pairs_41600.tsv` and `residue_alignment_41600.tsv`. Until they are connected, every record has
  `structure.status = "not_connected"`. No domain pair or residue mapping is inferred locally.

SHAP values explain the **model input features** of a pair. They are not assigned to residues; residue
colouring (Phase 2) is a separate structural annotation.

## Computation

| Quantity | How |
|---|---|
| `probability` | `FUJISAN.models.predict_proba` (the prediction function of `pair` / `search` / `batch`), same model file, features in the model's order (`check_features`), `num_iteration = best_iteration` |
| `raw_score` | `Booster.predict(raw_score=True)`, same iteration |
| `base_value`, `contributions` | `Booster.predict(pred_contrib=True)` (LightGBM TreeSHAP; equal to `shap.TreeExplainer` with the default path-dependent algorithm), same iteration |
| `sum_check` | `abs(base_value + Σ contributions − raw_score) ≤ 1e-9` per pair |
| `sigmoid_check` | `abs(sigmoid(raw_score) − probability) ≤ 1e-12` per pair (the models use `objective=binary sigmoid:1`; other objectives are rejected) |

**Units:** `raw_score`, `base_value` and `contributions` are in **log-odds**. A contribution is not a change in
probability, and it is not converted into one: the effect of a feature on the probability depends on the other
features.

**Skipped rows:** rows with a missing, non-numeric or non-finite (`inf`) feature, or without both protein IDs,
get `status = "skipped"` with `skip_reason`; they have no prediction and no contributions. Values are never
filled in. Non-finite values are stored as `null` in `features.values` (JSON has no `inf`).

**Input:** feature columns are selected by name and put in the model's order, so the column order of the input
does not matter. Missing feature columns, feature columns that occur twice (e.g. after `--rename`) and unknown
ID columns are errors (no output is written). Records keep the input order; duplicate pairs are kept and told
apart by `pair.source.row`. Empty input gives an empty file. The CLI treats only empty cells as missing, so IDs
such as `NA` stay text.

## Fields

| Field | Type | Content |
|---|---|---|
| `schema` | string | `"fujisan.interpretation"` |
| `schema_version` | string | `"1.0.0"` |
| `status` | string | `"ok"` or `"skipped"` |
| `skip_reason` | string | only when skipped |
| **`provenance`** | object | |
| `.model_key` | string | `updated` or `paper2024` (`MODEL_CONFIGS`) |
| `.model_file` | string | model file name |
| `.model_sha256` | string | SHA256 of the model file used |
| `.model_sha256_matches_release` | bool | equal to the SHA256 of the released file (`RELEASED_MODEL_SHA256`) |
| `.fujisan_version` | string | package version |
| `.fujisan_commit` / `.fujisan_commit_dirty` | string / bool | git commit of the checkout (`"unknown"` / `null` for an installed package); dirty = uncommitted changes to tracked files |
| `.lightgbm_version` | string | |
| `.created_at` | string | UTC, ISO 8601 |
| **`pair`** | object | |
| `.pair_id` | string or null | `<query_id>_<target_id>`; `null` when an ID is missing (record skipped) |
| `.query_id`, `.target_id` | string or null | as in the input, surrounding whitespace removed (query = protein 1); the orientation is kept, `B_A` is a different key from `A_B` |
| `.source` | object | input `file`, 1-based data `row`, `id_columns` used |
| **`features`** | object | |
| `.names` | list | model feature names in model order |
| `.values` | object | feature → value used for the prediction (`null` if missing) |
| **`prediction`** | object or null | null when skipped |
| `.raw_score`, `.raw_score_scale` | number, string | log-odds, `"log_odds"` |
| `.probability`, `.probability_function` | number, string | |
| `.sigmoid_check` | object | `abs_diff`, `tolerance`, `passed` |
| `.threshold` | number or null | from `MODEL_CONFIGS`; `null` for `updated` |
| `.threshold_source` | string | where the threshold comes from, or why there is none |
| `.call` | 0/1 or null | `probability ≥ threshold`; `null` when there is no threshold (no binary call is made) |
| `.reference_probability` | object | optional: a probability computed elsewhere for the same row (`column`, `value`, `abs_diff`); recorded, not corrected |
| **`shap`** | object or null | null when skipped |
| `.method` | string | `"lightgbm_pred_contrib"` |
| `.scale` | string | `"log_odds"` |
| `.base_value` | number | expected raw score of the model |
| `.contributions` | object | feature → contribution (log-odds) |
| `.sum_check` | object | `base_plus_sum`, `raw_score`, `abs_diff`, `tolerance`, `passed` |
| **`link`** | object | key for joining with server-side tables |
| `.pair_key` | string or null | same as `pair_id`: the `ProteinID1_ProteinID2` convention of the training pair tables. Not unique to split back when an ID itself contains `_` (e.g. local labels); join on `query_id` + `target_id` in that case |
| `.server_manifest` | string or null | name of the server manifest the pair was matched against (null: not matched) |
| `.server_match_status` | string | `"not_checked"` until matched |
| **`structure`** | object | |
| `.status` | string | `"not_connected"` (Phase 1) |
| `.reason` | string | |
| `.selected_domain_pair`, `.is_max_tm_pair`, `.superposition`, `.residue_alignment` | null | reserved for Phase 2 (see below) |

The flat TSV (`--tsv`) has one row per record: IDs, `status`, the feature values, `probability`,
`raw_score_log_odds`, `threshold`, `call`, `base_value_log_odds`, `shap_<feature>_log_odds`, the sum check,
`structure_status` and `model_sha256`.

## Reserved for Phase 2

Filled only from the server outputs, never inferred locally:

- `selected_domain_pair`: the domain pair used for the training-time `Dmn_*` features (from `ml_match.tsv` /
  `selected_domain_pairs_41600.tsv`), with domain IDs and UniProt residue ranges. It is not replaced by the
  pair with the highest TM-score.
- `is_max_tm_pair`: whether that pair also has the highest TM-score among all domain pairs; `false` is shown as is.
- `superposition`: which TM-align run the superposition comes from, and whether it is the one behind `Dmn_TM`.
- `residue_alignment`: aligned residues with UniProt numbering, position within the domain, gaps / missing
  residues and pLDDT; catalytic-residue annotations only with their source and evidence category.

Adding these fields will increase `schema_version` (minor version if only fields are added).

## Example

```bash
uv run fujisan explain -i out/predictions.tsv -o out/explain.jsonl --tsv out/explain.tsv
# GUI table with other column names:
uv run fujisan explain -i merged.csv -o explain.jsonl --id-columns query_id,target_id \
    --rename Full_Bitscore=BitScore --reference-probability FUJISAN_Score
```
