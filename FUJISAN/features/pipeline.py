"""Feature generation for the updated FUJISAN model: pair, search and batch.

Per protein:  structure + PAE -> pae_to_domains -> pLDDT >= 70 -> domains >= 30 residues
Per pair:     BLASTP (BitScore, Full_SeqID) + TM-align of all domain pairs
              -> best pair by average-length TM-score (Dmn_TM, Dmn_RMSD, Dmn_SeqID)

The steps are separate functions (prepare_protein, pair_features) so that a
structural pre-filter could later be put in front of TM-align without touching
the feature definitions. TM-align itself must stay: the model was trained on
TM-align values.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import errors as E
from ..errors import FujisanError
from ..prediction import FEATURES, add_probabilities
from . import alphafold as af
from . import blast as bl
from . import domains as dm
from . import tmalign as tm
from .structure import read_fasta, read_structure

PREDICTION_COLUMNS = ["Rank", "QueryID", "TargetID"] + FEATURES + ["FUJISAN_probability"]
FEATURE_COLUMNS = (["QueryID", "TargetID", "BlastSubjectID"] + FEATURES +
                   ["BLAST_evalue", "BLAST_alignment_length", "BLAST_qstart", "BLAST_qend", "BLAST_sstart",
                    "BLAST_send", "QueryLength", "TargetLength",
                    "SelectedDomain1", "SelectedDomain2", "Domain1Length", "Domain2Length", "TMscore1", "TMscore2",
                    "TMscoreMean", "AlignedLength", "NumDomains1", "NumDomains2", "NumDomainPairs",
                    "NumDomainPairsFailed", "Structure1", "Structure2", "DBsearch_bitscore", "DBsearch_pident",
                    "DBsearch_evalue", "Note"])
SKIPPED_COLUMNS = ["target_id", "accession", "reason", "detail"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Tools:
    blastp: str
    tmalign: str
    threads: int = 1
    workers: int = 1
    debug: bool = False
    dbsize: Optional[int] = None    # -dbsize override; None = BLAST default


@dataclass
class ProteinSource:
    label: str
    sequence: str
    structure_path: str
    pae_path: str
    origin: str                 # "afdb:<model>-v<version>" or "local"
    accession: str = ""
    note: str = ""


@dataclass
class Prepared:
    source: ProteinSource
    sequence: str
    n_domains: int
    domains: list = field(default_factory=list)   # usable Domain objects with PDB files


def protein_from_afdb(accession: str, cache, refresh: bool = False) -> ProteinSource:
    entry = af.fetch_entry(accession, cache, refresh)
    return ProteinSource(entry.accession, entry.sequence, str(entry.structure_path), str(entry.pae_path),
                         f"afdb:{entry.model_id}-v{entry.version}" + (" (cache)" if entry.from_cache else ""),
                         entry.accession, entry.note)


def protein_from_local(structure: str, pae: str, fasta: str | None = None, label: str | None = None) -> ProteinSource:
    for p, what in ((structure, "structure"), (pae, "PAE")):
        if not p or not Path(p).exists():
            raise FujisanError(E.STRUCTURE_UNAVAILABLE if what == "structure" else E.PAE_UNAVAILABLE, str(p))
    name, seq = read_fasta(fasta) if fasta else (Path(structure).stem, "")
    return ProteinSource(label or name, seq, structure, pae, "local")


def prepare_protein(src: ProteinSource, workdir: Path) -> Prepared:
    st = read_structure(src.structure_path)
    seq = src.sequence or st.sequence
    if len(seq) != len(st):
        raise FujisanError(E.LENGTH_MISMATCH, f"{src.label}: sequence {len(seq)} aa, structure {len(st)} residues")
    note = src.note
    diff = sum(1 for a, b in zip(seq, st.sequence) if a != b and "X" not in (a, b))
    if diff:
        note = (note + "; " if note else "") + f"{diff} residues differ between sequence and structure"
    pae = dm.load_pae(src.pae_path)
    domains = dm.segment(st, pae)
    usable = dm.usable_domains(domains, workdir, src.label)
    if note != src.note:
        src.note = note
    return Prepared(src, seq, len(domains), usable)


def pair_features(p1: Prepared, p2: Prepared, tools: Tools, db_hit: bl.BlastHit | None = None,
                  keep_dir: Path | None = None) -> dict:
    """Five features (+ metadata) for one protein pair; FujisanError when they cannot be computed.

    BitScore / Full_SeqID always come from BLAST of the pair alone (``blast_pair``), as for the
    training data; ``db_hit`` (search mode) is only kept as metadata.
    """
    hit = bl.blast_pair(p1.sequence, p2.sequence, tools.blastp, dbsize=tools.dbsize)
    results, failures = tm.align_all(p1.domains, p2.domains, tools.tmalign)
    if not results:
        if failures:
            raise FujisanError(E.TMALIGN_FAILED, f"all {len(failures)} domain pairs failed: {failures[0][2]}")
        raise FujisanError(E.NO_DOMAIN_PAIR)
    best = tm.select_best_pair(results)
    r = best.result
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        (keep_dir / f"{p1.source.label}_D{best.domain1}__{p2.source.label}_D{best.domain2}.tmalign.txt").write_text(r.raw)
    notes = [n for n in (p1.source.note, p2.source.note) if n]
    if failures:
        notes.append(f"{len(failures)} domain pairs failed in TM-align")
    return {
        "QueryID": p1.source.label, "TargetID": p2.source.label,
        "BlastSubjectID": db_hit.sseqid if db_hit else "",
        "DBsearch_bitscore": db_hit.bitscore if db_hit else "", "DBsearch_pident": db_hit.pident if db_hit else "",
        "DBsearch_evalue": db_hit.evalue if db_hit else "",
        "BitScore": bl.bitscore_feature(hit), "Full_SeqID": hit.pident,
        "Dmn_RMSD": r.rmsd, "Dmn_SeqID": r.seq_id, "Dmn_TM": r.tm_avg,
        "BLAST_evalue": hit.evalue, "BLAST_alignment_length": hit.length, "BLAST_qstart": hit.qstart,
        "BLAST_qend": hit.qend, "BLAST_sstart": hit.sstart, "BLAST_send": hit.send,
        "QueryLength": hit.qlen, "TargetLength": hit.slen,
        "SelectedDomain1": best.domain1, "SelectedDomain2": best.domain2,
        "Domain1Length": r.len1, "Domain2Length": r.len2, "TMscore1": r.tm1, "TMscore2": r.tm2,
        "TMscoreMean": round((r.tm1 + r.tm2) / 2, 5), "AlignedLength": r.aligned_length,
        "NumDomains1": len(p1.domains), "NumDomains2": len(p2.domains), "NumDomainPairs": len(results),
        "NumDomainPairsFailed": len(failures), "Structure1": p1.source.origin, "Structure2": p2.source.origin,
        "Note": "; ".join(notes),
    }


def write_tsv(path: Path, rows: list, columns: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    tmp.replace(path)


# --------------------------------------------------------------------------- search

class TargetStore:
    """Prepared target proteins, shared by all queries of a run (thread-safe)."""

    def __init__(self, cache, workdir: Path, refresh: bool):
        self.cache, self.workdir, self.refresh = cache, workdir, refresh
        self._done: dict = {}
        self._locks: dict = {}
        self._lock = threading.Lock()

    def get(self, accession: str) -> Prepared:
        with self._lock:
            lock = self._locks.setdefault(accession, threading.Lock())
        with lock:
            if accession not in self._done:
                try:
                    src = protein_from_afdb(accession, self.cache, self.refresh)
                    self._done[accession] = prepare_protein(src, self.workdir / "targets")
                except FujisanError as exc:
                    self._done[accession] = exc
            out = self._done[accession]
        if isinstance(out, FujisanError):
            raise out
        return out


def run_search(query: Prepared, db: str, max_hits: int, tools: Tools, store: TargetStore, outdir: Path,
               keep_dir: Path | None = None) -> dict:
    """BLAST the query against the database, compute features for each hit, predict and write the outputs."""
    outdir.mkdir(parents=True, exist_ok=True)
    hits = bl.blast_search(query.sequence, db, tools.blastp, max_hits, threads=tools.threads, dbsize=tools.dbsize)
    log(f"[{query.source.label}] {len(hits)} BLAST hits")
    skipped, todo, seen = [], [], set()
    for h in hits:
        try:
            acc = bl.hit_accession(h)
        except FujisanError as exc:
            skipped.append({"target_id": h.sseqid, "accession": "", "reason": exc.reason, "detail": exc.detail})
            continue
        if acc in seen:
            skipped.append({"target_id": h.sseqid, "accession": acc, "reason": "Duplicate accession",
                            "detail": "already processed from an earlier hit"})
            continue
        seen.add(acc)
        todo.append((acc, h))

    def one(item):
        acc, h = item
        try:
            target = store.get(acc)
            return pair_features(query, target, tools, db_hit=h, keep_dir=keep_dir), None
        except FujisanError as exc:
            return None, {"target_id": h.sseqid, "accession": acc, "reason": exc.reason, "detail": exc.detail}
        except Exception as exc:                        # unexpected: record, keep going
            if tools.debug:
                traceback.print_exc()
            return None, {"target_id": h.sseqid, "accession": acc, "reason": "Unexpected error",
                          "detail": f"{type(exc).__name__}: {exc}"}

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, tools.workers)) as ex:
        for k, (row, skip) in enumerate(ex.map(one, todo), start=1):
            if row:
                rows.append(row)
            else:
                skipped.append(skip)
            if k % 10 == 0 or k == len(todo):
                log(f"[{query.source.label}] {k}/{len(todo)} targets ({len(rows)} with features)")
    rows = add_probabilities(rows)
    rows.sort(key=lambda r: r["FUJISAN_probability"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["Rank"] = i
    write_tsv(outdir / "predictions.tsv", rows, PREDICTION_COLUMNS)
    write_tsv(outdir / "features.tsv", rows, FEATURE_COLUMNS)
    write_tsv(outdir / "skipped_targets.tsv", skipped, SKIPPED_COLUMNS)
    return {"n_hits": len(hits), "n_predicted": len(rows), "n_skipped": len(skipped),
            "top_target": rows[0]["TargetID"] if rows else "",
            "top_probability": rows[0]["FUJISAN_probability"] if rows else ""}


class WorkDir:
    """Temporary directory for domain PDBs etc.; kept only with --keep-intermediates."""

    def __init__(self, outdir: Path, keep: bool):
        self.keep = keep
        if keep:
            self.path = outdir / "intermediates"
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            self.path = Path(tempfile.mkdtemp(prefix="fujisan_work_"))

    def cleanup(self) -> None:
        if not self.keep and self.path.name.startswith("fujisan_work_"):
            shutil.rmtree(self.path, ignore_errors=True)
