"""PAE parsing and domain segmentation (Fujita & Terada 2024, Section 2.1).

structure + PAE
    -> pae_to_domains (upstream defaults, igraph Leiden) on the full PAE matrix
    -> keep residues with pLDDT >= 70 within each domain
    -> discard domains with fewer than 30 such residues

Notes on matching the training data of the FUJISAN models:

* pLDDT cut-off. The paper says "pLDDT greater than 70"; the script that
  built the training-data domain files (SplitPDB_by_pae.py) kept atoms with
  ``B-factor >= 70``. We follow the training code (``>= 70``). The two
  differ only for residues whose pLDDT is exactly 70.00.
* Minimum size. Domains with fewer than 30 residues (counted after the pLDDT
  filter) are discarded, as stated in the paper.
* Leiden is randomised. pae_to_domains calls igraph's Leiden without a seed,
  so repeated runs can split the same protein differently. We seed Python's
  ``random`` (which igraph uses) before each call so that results are
  reproducible; the training data were produced without a seed.
* Indexing. Row/column i of the PAE matrix is the i-th residue of the chain
  in file order (``Residue.index``); residue numbers are not used for the
  mapping, so models numbered from anything other than 1 are handled.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import (INVALID_PAE, LENGTH_MISMATCH, NO_CONFIDENT_RESIDUES, NO_DOMAIN, PAE_UNAVAILABLE,
                      SEGMENTATION_FAILED, FujisanError)
from .structure import Structure, write_pdb

# Upstream defaults of pae_to_domains (pae_to_domains.py `_defaults`).
PAE_POWER = 1.0
PAE_CUTOFF = 5.0
GRAPH_RESOLUTION = 1.0
PAE_LIBRARY = "igraph"
PLDDT_MIN = 70.0          # residues with pLDDT >= PLDDT_MIN are kept (see module docstring)
MIN_DOMAIN_RESIDUES = 30
LEIDEN_SEED = 0


def load_pae(path: str | Path) -> np.ndarray:
    """PAE matrix from AlphaFold DB (current and legacy formats), ColabFold or AlphaFold 3 JSON."""
    path = Path(path)
    if not path.exists():
        raise FujisanError(PAE_UNAVAILABLE, str(path))
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise FujisanError(INVALID_PAE, f"{path}: not JSON ({exc})") from exc
    return parse_pae(data, str(path))


def parse_pae(data, source: str = "PAE") -> np.ndarray:
    if isinstance(data, list):
        if not data:
            raise FujisanError(INVALID_PAE, f"{source}: empty list")
        data = data[0]
    if not isinstance(data, dict):
        raise FujisanError(INVALID_PAE, f"{source}: unexpected JSON structure")
    if "predicted_aligned_error" in data:                 # AFDB (2022-) / AlphaFold 2
        m = np.array(data["predicted_aligned_error"], dtype=np.float64)
    elif "pae" in data:                                   # ColabFold / AlphaFold 3
        m = np.array(data["pae"], dtype=np.float64)
    elif "residue1" in data and "distance" in data:       # legacy AFDB format
        size = int(max(data["residue1"]))
        d = np.array(data["distance"], dtype=np.float64)
        if d.size != size * size:
            raise FujisanError(INVALID_PAE, f"{source}: legacy format with {d.size} values for {size} residues")
        m = d.reshape(size, size)
    else:
        raise FujisanError(INVALID_PAE, f"{source}: no 'predicted_aligned_error' or 'pae' field")
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise FujisanError(INVALID_PAE, f"{source}: matrix is not square ({m.shape})")
    if not np.all(np.isfinite(m)):
        raise FujisanError(INVALID_PAE, f"{source}: non-finite values")
    return m


def cluster_pae(pae: np.ndarray, seed: int = LEIDEN_SEED) -> list:
    """pae_to_domains clusters (lists of 0-based residue indices, largest first)."""
    from ..vendor.pae_to_domains.pae_to_domains import domains_from_pae_matrix_igraph

    random.seed(seed)
    try:
        with np.errstate(divide="ignore"):       # diagonal PAE = 0 -> weight inf, as upstream
            clusters = domains_from_pae_matrix_igraph(pae, pae_power=PAE_POWER, pae_cutoff=PAE_CUTOFF,
                                                      graph_resolution=GRAPH_RESOLUTION)
    except Exception as exc:
        raise FujisanError(SEGMENTATION_FAILED, f"pae_to_domains: {exc}") from exc
    return [sorted(int(i) for i in c) for c in clusters]


@dataclass
class Domain:
    number: int            # 1-based, in pae_to_domains order (largest cluster first)
    n_assigned: int        # residues assigned by pae_to_domains
    residues: list         # Residue objects kept (pLDDT >= PLDDT_MIN)
    usable: bool           # >= MIN_DOMAIN_RESIDUES kept residues
    pdb_path: str = ""     # written for TM-align when usable

    @property
    def n_kept(self) -> int:
        return len(self.residues)


def segment(structure: Structure, pae: np.ndarray, clusters: list | None = None) -> list:
    """Domains of a structure; ``clusters`` (0-based indices) may be given instead of running pae_to_domains."""
    n = len(structure)
    if pae.shape[0] != n:
        raise FujisanError(LENGTH_MISMATCH, f"PAE has {pae.shape[0]} residues, structure has {n}")
    if not any(r.plddt >= PLDDT_MIN for r in structure.residues):
        raise FujisanError(NO_CONFIDENT_RESIDUES, structure.path)
    clusters = cluster_pae(pae) if clusters is None else clusters
    domains = []
    for k, idx in enumerate(clusters, start=1):
        kept = [structure.residues[i] for i in sorted(idx) if structure.residues[i].plddt >= PLDDT_MIN]
        domains.append(Domain(k, len(idx), kept, len(kept) >= MIN_DOMAIN_RESIDUES))
    return domains


def usable_domains(domains: list, workdir: str | Path, label: str) -> list:
    """Write the usable domains as PDB files for TM-align; raise if there is none."""
    out = [d for d in domains if d.usable]
    if not out:
        raise FujisanError(NO_DOMAIN, f"{label}: largest domain has {max((d.n_kept for d in domains), default=0)} "
                                      f"residues with pLDDT >= {PLDDT_MIN:g}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    for d in out:
        d.pdb_path = str(workdir / f"{safe}_domain{d.number}.pdb")
        write_pdb(d.residues, d.pdb_path)
    return out
