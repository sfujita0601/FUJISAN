# Feature-generation pipeline (`updated` model)

The `fujisan` command computes the five input features of the **`updated`** model
(`BitScore`, `Full_SeqID`, `Dmn_RMSD`, `Dmn_SeqID`, `Dmn_TM`) from your own proteins and
returns the FUJISAN probability. It does **not** generate the pocket features of the
`paper2024` model, and it does not change or retrain either model.

```
protein A, protein B ──► BLASTP ─────────────────────────► BitScore, Full_SeqID
        │
        └─► AlphaFold model + PAE ─► pae_to_domains ─► pLDDT ≥ 70 ─► domains ≥ 30 residues
                                                                        │
                       TM-align of all domain pairs (A_i × B_j) ◄───────┘
                       best pair by average-length TM-score ─► Dmn_TM, Dmn_RMSD, Dmn_SeqID
                                                                        │
                                        updated LightGBM model ◄────────┘ ─► FUJISAN_probability
```

Validation of the pipeline against the historical FUJISAN features is described in
[validation.md](validation.md).

## Contents

- [Requirements](#requirements)
- [External dependencies](#external-dependencies)
- [BLAST database preparation](#blast-database-preparation)
- [Pairwise analysis](#pairwise-analysis)
- [Search from one query](#search-from-one-query)
- [Batch analysis](#batch-analysis)
- [AlphaFold DB automatic retrieval](#alphafold-db-automatic-retrieval)
- [Cache and storage requirements](#cache-and-storage-requirements)
- [Output files](#output-files)
- [Feature definitions](#feature-definitions)
- [Reproducibility metadata](#reproducibility-metadata)
- [Using the models with precomputed features (Python)](#using-the-models-with-precomputed-features-python)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

## Requirements

- Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (`uv sync` installs the Python dependencies, including
  `python-igraph` for pae_to_domains and `gemmi` for reading PDB / mmCIF files).
- `uv sync` in a clone (then `uv run fujisan ...`), or `python3 -m pip install .` (then `fujisan ...` from any
  directory). The model files are included in the installed package (`FUJISAN/model`) and found with
  `importlib.resources`; in a source checkout they are read from `model/`.

## External dependencies

These are **not** installed by FUJISAN; install them yourself and make them available on `PATH`
(or give their paths with `--blastp-bin` / `--tmalign-bin`).

| Tool | Used for | Check |
|---|---|---|
| [BLAST+](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html) (`blastp`, `makeblastdb`) | `BitScore`, `Full_SeqID`, database search | `blastp -version` |
| [TM-align](https://zhanggroup.org/TM-align/) (`TMalign`) | `Dmn_TM`, `Dmn_RMSD`, `Dmn_SeqID` | `TMalign` (prints its version) |

[pae_to_domains](https://github.com/tristanic/pae_to_domains) (MIT License, Tristan Croll) is a single
script without packaging, so an unmodified copy is included in
[`FUJISAN/vendor/pae_to_domains/`](../FUJISAN/vendor/pae_to_domains/) (source commit and license there).

## BLAST database preparation

`fujisan search` and `fujisan batch` need a protein BLAST database that **you** prepare; FUJISAN
does not download sequences or run `makeblastdb`. Hit IDs must contain UniProt accessions so that the
AlphaFold DB models of the hits can be found. Example with Swiss-Prot:

```bash
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
makeblastdb -in uniprot_sprot.fasta -dbtype prot -parse_seqids -out /path/to/swissprot
```

Accepted hit IDs: `sp|P12345|NAME`, `tr|A0A0B4J2F0|NAME`, bare accessions (`P12345`), `UniRef100_P12345`
and AFDB IDs (`AF-P12345-F1`). Isoforms (`P12345-2`) and IDs that are not UniProt accessions are not
guessed; they are listed in `skipped_targets.tsv`.

## Pairwise analysis

```bash
# both proteins from the AlphaFold Protein Structure Database
uv run fujisan pair --afdb-id1 P00509 --afdb-id2 Q22067 --output pair.tsv

# local files (structure: PDB or mmCIF with pLDDT in the B-factor column; PAE: JSON)
uv run fujisan pair \
    --fasta1 A.fasta --structure1 A.cif --pae1 A_pae.json \
    --fasta2 B.fasta --structure2 B.pdb --pae2 B_pae.json \
    --output pair.tsv
```

AFDB IDs and local files can be mixed (`--afdb-id1` with `--structure2 / --pae2`). The FASTA is optional;
without it the sequence of the structure is used. A FASTA whose length differs from the structure is an error.

## Search from one query

```bash
uv run fujisan search --query-afdb-id P00509 --blast-db /path/to/swissprot --max-hits 100 --output results/
uv run fujisan search --query-fasta q.fasta --query-structure q.cif --query-pae q.json \
    --blast-db /path/to/swissprot --max-hits 100 --output results/
```

The query is searched against the database (`blastp`, E-value ≤ 10, one HSP per subject); for each hit
the AlphaFold DB model is fetched, the domain features are computed, and the hits are ranked by FUJISAN
probability. The database search only selects the candidates: `BitScore` and `Full_SeqID` are then computed
for each query–hit pair exactly as in `pair` (the database-search values are kept in `features.tsv`;
see [BLAST settings](#blast-settings-shared-by-all-commands)).
`--max-hits 0` analyses all hits (**can be very slow and use a lot of disk space**). `--workers N` processes
N hits in parallel (default 1; AlphaFold DB requests stay throttled).

## Batch analysis

```bash
uv run fujisan batch --queries queries.tsv --blast-db /path/to/swissprot --max-hits 100 --output batch_results/
```

`queries.tsv` (tab-separated) needs a column `afdb_id` (or `query_id` holding AFDB IDs). For local files use
the columns `query_id`, `structure`, `pae` and optionally `fasta` (paths relative to the TSV). Each query gets
its own directory; a query that fails is recorded in `batch_summary.tsv` and the batch continues.

```
batch_results/
├── P00509/   predictions.tsv  features.tsv  skipped_targets.tsv  run_metadata.json
├── Q22067/   ...
├── batch_summary.tsv
└── run_metadata.json
```

## AlphaFold DB automatic retrieval

`--afdb-id*` and the hits of `search` / `batch` are fetched from the AlphaFold Protein Structure Database.
IDs such as `Q9HBA0`, `q9hba0`, `AF-Q9HBA0-F1` and `AF-Q9HBA0-F1-model_v4` are accepted and normalised to the
UniProt accession. The file URLs come from the official API (`https://alphafold.ebi.ac.uk/api/prediction/<accession>`:
`pdbUrl`, `paeDocUrl`, `latestVersion`), so the current AFDB model is used. Requests time out after 60 s, are
retried at most 3 times (no retry on 404; HTTP 429 honours `Retry-After`), and are throttled to at most two at a
time. A target whose files cannot be obtained is skipped with the reason; the run continues.

### AFDB v4 vs current AFDB

Model development used AFDB v4 models. Automatic retrieval uses the currently available (latest) AFDB version
(v6 at the time of writing), so numerical differences can occur. In our validation sample of 40 test-set pairs
computed from both versions (v4 from local copies, v6 downloaded), the mean / median / maximum absolute differences
were 0.0037 / 0.0017 / 0.041 for `Dmn_TM`, 0.051 / 0.040 / 0.29 Å for `Dmn_RMSD` and 0.0020 / 0.0005 / 0.015 for
`Dmn_SeqID`; the FUJISAN probability (updated model) differed by 0.021 / 0.004 / 0.14 (Pearson r = 0.994,
Spearman ρ = 0.987). Sequences, and hence the BLAST features, were the same for 39 of the 40 pairs. v4 files are
not required; to use them, pass them as local structure / PAE files. Details: [validation.md](validation.md).

## Cache and storage requirements

> [!WARNING]
> **Storage requirements.** FUJISAN may download AlphaFold structures and PAE files for many BLAST hits.
> With the default `--max-hits 100`, a single query can result in many downloaded files (about 0.5 MB for a
> 400-residue protein; the PAE file grows with the square of the length, ≈ 5 MB at 1,500 residues), and batch
> analyses can therefore consume substantial disk space.
> Downloaded AFDB files are cached to avoid repeated downloads.
> Check the cache directory periodically and remove it manually when no longer needed.

- Cache directory: `~/.cache/fujisan` (or `$FUJISAN_CACHE_DIR`, or `--cache-dir PATH`), one directory per
  accession and AFDB model version (`afdb/<accession>/<model>-v<version>/model.pdb`, `pae.json`, `metadata.json`).
  A cached entry is reused and not downloaded again (`--refresh-afdb` forces a new download).
- `--keep-downloads` (default) keeps the AFDB files in the cache; `--no-keep-downloads` downloads into a temporary
  cache that is deleted at the end of the run.
- Domain PDB files, raw TM-align output and BLAST input files are temporary and deleted at the end of the run;
  `--keep-intermediates` keeps them in `<output>/intermediates/` for debugging.
- `search` and `batch` print the number of hits to process and the cache directory at the start, and the cache
  size at the end. `fujisan cache-info` shows the cache directory, the number of AFDB entries and the structure,
  PAE and total sizes. FUJISAN does not delete the cache; remove it yourself when it is no longer needed.

## Output files

| File | Content |
|---|---|
| `predictions.tsv` | `Rank`, `QueryID`, `TargetID`, the five features, `FUJISAN_probability`; sorted by probability (descending) |
| `features.tsv` | the five features and metadata: BLAST E-value / alignment, `SelectedDomain1/2`, `Domain1Length/2`, `TMscore1/2`, `TMscoreMean`, `AlignedLength`, numbers of domains and domain pairs, AFDB model versions, notes |
| `skipped_targets.tsv` | `target_id`, `accession`, `reason`, `detail` for every hit without features (no accession, no AFDB entry, PAE unavailable, no domain ≥ 30 residues, TM-align failed, ...) |
| `run_metadata.json` | see [Reproducibility metadata](#reproducibility-metadata) |
| `pair` | one TSV row (`ProteinID1`, `ProteinID2`, five features, `FUJISAN_probability`, metadata) and `<output>.run_metadata.json` |

No binary prediction is written: the `updated` model has no validated classification threshold.
Missing features are never filled in; such pairs are only listed in `skipped_targets.tsv`.

## Feature definitions

| Feature | Definition |
|---|---|
| `BitScore` | raw BLAST `bitscore` of the best HSP (`-max_hsps 1`) of `blastp` on the protein pair (first protein as query) |
| `Full_SeqID` | `pident` of the same HSP |
| `Dmn_TM` | TM-score normalised by the average length of the two domains (`TMalign -a T`), maximum over all domain pairs |
| `Dmn_RMSD` | RMSD reported by TM-align for **the same** domain pair |
| `Dmn_SeqID` | `Seq_ID=n_identical/n_aligned` reported by TM-align for the same domain pair |

### Domain segmentation

Domains (paper Section 2.1): `pae_to_domains` with its default parameters (`pae_power=1`, `pae_cutoff=5`,
`resolution=1`, igraph Leiden) on the full PAE matrix; within each domain only residues with pLDDT ≥ 70 are
kept, and domains with fewer than 30 such residues are discarded. Row/column *i* of the PAE is the *i*-th
residue of the first chain of the structure (residue numbers are not used for the mapping).

### TM-align and domain-pair selection

Every usable domain of protein A is aligned with every usable domain of protein B (`TMalign A_i.pdb B_j.pdb -a T`).
The domain pair with the highest `Dmn_TM` is selected, and `Dmn_RMSD` and `Dmn_SeqID` are taken from that same pair.

### Notes on matching the training data

- "Average TM-score" is TM-align's TM-score normalised by the average length (`-a T`), as in the `TM-scoreAve`
  column of the training/test data; it is not (TM-score1 + TM-score2) / 2, which is written to `features.tsv`
  as `TMscoreMean` for information only.
- The paper states "pLDDT greater than 70"; the script that built the training-data domains kept pLDDT ≥ 70.
  The pipeline follows the training code. The two differ only for residues with pLDDT exactly 70.00.
- The original feature generation used stochastic Leiden clustering (pae_to_domains / igraph) without a fixed
  seed. The released pipeline fixes `seed=0` for reproducibility, so some historical domain assignments may differ
  from those of the pipeline. In 184 of 200 pairs of our validation sample the three domain features were identical;
  with the updated model the FUJISAN probability changed by more than 0.05 for 10 and by more than 0.20 for 2 of
  the 200 pairs (see [validation.md](validation.md)).

### BLAST settings shared by all commands

- `BitScore` and `Full_SeqID` are computed as for training the updated model: `blastp` of the full-length sequence
  pair with the first protein as query, default scoring parameters, `-evalue 10 -max_hsps 1`; `BitScore` is the raw
  `bitscore` and `Full_SeqID` the `pident` of that single (best) HSP. HSPs are not aggregated. The output columns are
  requested explicitly (`-outfmt "6 qseqid sseqid sacc pident length evalue bitscore qstart qend sstart send qlen
  slen stitle"`).
- The bit score itself is not normalised by the database size, but the search context can change which alignment
  (HSP) BLAST reports. In our validation, P0A6V0–B0SFJ4 gave `BitScore` 203 / `Full_SeqID` 30.242 with single-pair
  BLAST and 205 / 31.020 in a search of a 227k-sequence database. The pipeline therefore always computes the two
  features with pairwise BLAST.
- The second protein is written to a temporary one-sequence database (`makeblastdb`) and searched with `blastp -db`;
  this gave the same bit score, identity and alignment length as `blastp -subject` for all 40 pairs checked.
- `search` / `batch` use the database search only to select candidates and then compute the two features for each
  pair in the same way. The database-search values are written to `features.tsv` (`DBsearch_*`).
- No `-dbsize` is passed by default (BLAST's own database size). `--blast-dbsize N` is an optional override.
- BLAST+ 2.17.0 was used for training-feature generation. Other recent BLAST+ versions may produce small
  differences; BLAST+ 2.16.0 and 2.17.0 gave identical values for the 40 pairs checked (see
  [validation.md](validation.md)).

## Reproducibility metadata

Every run writes `run_metadata.json` with: FUJISAN version; model name, file and md5; feature names and their
definitions; BLAST version, E-value, `-dbsize` (if set), outfmt; TM-align path, version and options; pae_to_domains source commit,
parameters, python-igraph version and Leiden seed; pLDDT rule and minimum domain size; Python version and platform;
timestamp; all command-line arguments, including the BLAST database path, `--blast-db-label`
(e.g. `--blast-db-label "Swiss-Prot 2026_01"`) and `--max-hits`; the AFDB model version of each protein.

## Using the models with precomputed features (Python)

If you already have the features, the models can be used directly. Each row is a protein pair; the columns must
be exactly the features of the selected model, in this order.

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
| `BitScore` | BLAST bit score between the full-length sequences (`bitscore` column of `blastp` output; see [Feature definitions](#feature-definitions)) |
| `Full_SeqID` | same as `paper2024` |
| `Dmn_RMSD` | same as `paper2024` |
| `Dmn_SeqID` | same as `paper2024` |
| `Dmn_TM` | same as `paper2024` |

```python
import pandas as pd
from FUJISAN.models import MODEL_CONFIGS, load_model, predict_proba

config = MODEL_CONFIGS['paper2024']
model = load_model('paper2024')
X = pd.read_csv('my_pairs.csv')[config['features']]   # columns as above
prob = predict_proba(model, X)                          # probability of identical function
pred = (prob >= config['threshold']).astype(int)
```

`MODEL_CONFIGS` is defined in [`FUJISAN/models.py`](../FUJISAN/models.py). For the `updated` model
`config['threshold']` is `None` (no validated threshold); use the probabilities only.

## Limitations

1. The feature-generation pipeline is for the `updated` 5-feature model only.
2. The pocket features of the `paper2024` 7-feature model are not generated.
3. The `updated` model is not the model of the published study.
4. No validated classification threshold is defined for the `updated` model; only probabilities are reported.
5. The BLAST database must be prepared by the user.
6. BLAST hit candidates depend on the content and version of that database.
7. Targets without an AlphaFold DB entry cannot be analysed (they are listed in `skipped_targets.tsv`).
8. Many BLAST hits or batch analyses take long and consume substantial disk space.
9. Foldseek is not used.
10. Replacing the TM-align-derived features with Foldseek (or other) scores would create a distribution mismatch
    with the trained model.
11. Model development used AFDB v4 models. Automatic AFDB retrieval uses the currently available (latest) AFDB
    version (v6 at the time of writing), so numerical differences from the training-time features can occur
    (see [AFDB v4 vs current AFDB](#afdb-v4-vs-current-afdb)). v4 files are not required; they can be given through
    the local-input options.

## Future improvements

- An optional structural pre-filter of BLAST hits (e.g. Foldseek) before the exact TM-align step, to speed up
  large searches. The features passed to the model would still come from TM-align.
