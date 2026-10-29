# Vendored: pae_to_domains

`pae_to_domains.py` and `LICENSE` in this directory are an **unmodified copy** of

- Repository: https://github.com/tristanic/pae_to_domains
- Commit: `f407c6035c825f151a56f28bf803fcb44321b941` (2022-07-28, latest on `main` when vendored)
- Author: Tristan Croll
- License: MIT (see `LICENSE`; Copyright (c) 2021 Tristan Croll)

It is vendored because the upstream project is a single script without
packaging (no `setup.py` / `pyproject.toml`, not on PyPI), so it cannot be
declared as a Python dependency. This is the same file that was used to
build the FUJISAN training data (checked byte-for-byte against the copy used
at the time).

FUJISAN calls `domains_from_pae_matrix_igraph()` with the upstream default
parameters (`pae_power=1.0`, `pae_cutoff=5.0`, `graph_resolution=1.0`,
igraph Leiden); see `FUJISAN/features/domains.py`.
