"""BLASTP: full-length sequence features (BitScore, Full_SeqID) and DB search.

Settings follow the construction of the FUJISAN data set (paper Section 2.1):
E-value threshold 10 and one HSP per subject (``-max_hsps 1``); with one HSP
BLAST reports its best-scoring local alignment, and that HSP's ``bitscore``
and ``pident`` are used. No aggregation over HSPs is done.

Features (as for training the updated model: blastp on the full-length sequence
pair, first protein as query, default scoring, ``-max_hsps 1``):
``BitScore`` = raw BLAST ``bitscore`` of the best HSP, ``Full_SeqID`` = ``pident`` of
the same HSP. They are always computed for the pair alone (``blast_pair``): protein B
is written to a temporary one-sequence database and searched with ``blast_search``.
This gives the same values as ``blastp -subject`` (checked on 40 pairs with BLAST+
2.16.0 and 2.17.0). In a search of a large database the alignment can differ slightly
(the search space affects it), so in search mode the database search only finds the
candidates and the features are recomputed for each pair.

Output columns are requested explicitly (outfmt 6, see OUTFMT_FIELDS). ``-dbsize`` is
not passed unless the user asks for it (``--blast-dbsize``).
"""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..errors import (BLAST_DB_NOT_FOUND, BLAST_FAILED, BLAST_NOT_FOUND, NO_ACCESSION, NO_BLAST_HIT,
                      FujisanError)

OUTFMT_FIELDS = ["qseqid", "sseqid", "sacc", "pident", "length", "evalue", "bitscore",
                 "qstart", "qend", "sstart", "send", "qlen", "slen", "stitle"]
EVALUE = 10.0          # as for the FUJISAN data set
MAX_HSPS = 1

BITSCORE_DEFINITION = "raw BLAST bitscore of the best HSP (-max_hsps 1), blastp of the sequence pair (first protein as query)"
FULL_SEQID_DEFINITION = "BLAST pident of the same HSP"


def bitscore_feature(hit: "BlastHit") -> float:
    """Value passed to the model as ``BitScore``: the raw BLAST bit score of the best HSP."""
    return hit.bitscore


@dataclass
class BlastHit:
    qseqid: str
    sseqid: str
    sacc: str
    pident: float      # -> Full_SeqID
    length: int
    evalue: float
    bitscore: float
    qstart: int
    qend: int
    sstart: int
    send: int
    qlen: int
    slen: int
    stitle: str


def find_blastp(binary: str = "blastp") -> str:
    path = shutil.which(binary)
    if not path:
        raise FujisanError(BLAST_NOT_FOUND, f"'{binary}' (install BLAST+ and check with `blastp -version`, "
                                            "or use --blastp-bin)")
    return path


def blast_version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return "unknown"
    return out.splitlines()[0].strip() if out else "unknown"


def check_blast_db(db: str) -> None:
    """A protein BLAST database exists at this prefix (single volume, multi-volume or alias)."""
    patterns = [f"{db}.pin", f"{db}.pal", f"{db}.*.pin", f"{db}.pdb", f"{db}.psq"]
    if not any(glob.glob(p) for p in patterns):
        raise FujisanError(BLAST_DB_NOT_FOUND, f"no protein BLAST database at '{db}' (expected {db}.pin / {db}.pal; "
                                               "build one with makeblastdb -dbtype prot)")


def parse_outfmt6(text: str) -> list:
    hits = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < len(OUTFMT_FIELDS):
            f += [""] * (len(OUTFMT_FIELDS) - len(f))
        hits.append(BlastHit(f[0], f[1], f[2], float(f[3]), int(f[4]), float(f[5]), float(f[6]), int(f[7]),
                             int(f[8]), int(f[9]), int(f[10]), int(f[11]), int(f[12]), f[13]))
    return hits


def _run(cmd: list) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise FujisanError(BLAST_FAILED, str(exc)) from exc
    if proc.returncode != 0:
        raise FujisanError(BLAST_FAILED, (proc.stderr or proc.stdout).strip()[-500:])
    return proc.stdout


def find_makeblastdb(blastp: str) -> str:
    """makeblastdb of the same BLAST+ installation as blastp (else the one on PATH)."""
    sibling = Path(blastp).with_name("makeblastdb")
    if sibling.exists():
        return str(sibling)
    path = shutil.which("makeblastdb")
    if not path:
        raise FujisanError(BLAST_NOT_FOUND, "'makeblastdb' (part of BLAST+; needed by `fujisan pair`)")
    return path


def blast_pair(seq1: str, seq2: str, binary: str, evalue: float = EVALUE, dbsize: Optional[int] = None) -> BlastHit:
    """BLASTP of sequence 1 (query) against a one-sequence database of sequence 2; the best HSP.

    Uses ``blast_search`` (same options, parser and HSP selection as the database search).
    """
    with tempfile.TemporaryDirectory(prefix="fujisan_blastdb_") as tmp:
        fasta = Path(tmp) / "target.fasta"
        fasta.write_text(f">protein2\n{seq2}\n")
        _run([find_makeblastdb(binary), "-in", str(fasta), "-dbtype", "prot", "-out", str(Path(tmp) / "target")])
        hits = blast_search(seq1, str(Path(tmp) / "target"), binary, max_hits=1, evalue=evalue, dbsize=dbsize)
    if not hits:
        raise FujisanError(NO_BLAST_HIT, f"E-value threshold {evalue:g}")
    return hits[0]


def blast_search(sequence: str, db: str, binary: str, max_hits: int, evalue: float = EVALUE,
                 threads: int = 1, dbsize: Optional[int] = None) -> list:
    """BLASTP of one query sequence against a database; one row (best HSP) per subject, BLAST order."""
    max_targets = max_hits if max_hits > 0 else 1_000_000
    with tempfile.TemporaryDirectory(prefix="fujisan_blast_") as tmp:
        q = Path(tmp) / "query.fasta"
        q.write_text(f">query\n{sequence}\n")
        out = _run([binary, "-query", str(q), "-db", db, "-evalue", f"{evalue:g}",
                    "-max_target_seqs", str(max_targets), "-max_hsps", str(MAX_HSPS),
                    "-num_threads", str(threads), "-outfmt", "6 " + " ".join(OUTFMT_FIELDS)]
                   + (["-dbsize", str(dbsize)] if dbsize else []))
    return parse_outfmt6(out)


# --------------------------------------------------------------------------- UniProt accessions

ACCESSION_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
ISOFORM_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})-\d+$")


def is_accession(text: str) -> bool:
    return bool(ACCESSION_RE.match(text))


def parse_uniprot_accession(target_id: str) -> str:
    """UniProt accession from a BLAST subject ID; FujisanError(NO_ACCESSION) if there is none.

    Accepted: ``sp|P12345|NAME_SPECIES``, ``tr|A0A0B4J2F0|NAME``, a bare
    accession (``P12345``, ``P12345.1``), ``UniRef100_P12345`` /
    ``UniRef90_P12345`` (UniRef clusters named after a UniProtKB entry) and
    AFDB IDs (``AF-P12345-F1``). Isoforms (``P12345-2``) are rejected: AFDB
    holds models of canonical sequences only. Anything else is rejected rather
    than guessed.
    """
    tid = target_id.strip()
    parts = tid.split("|")
    if len(parts) >= 2 and parts[0] in ("sp", "tr"):
        cand = parts[1]
    elif tid.startswith(("UniRef100_", "UniRef90_", "UniRef50_")):
        cand = tid.split("_", 1)[1]
    elif tid.startswith("AF-"):
        cand = tid.split("-")[1] if tid.count("-") >= 2 else ""
    else:
        cand = tid
    cand = re.sub(r"\.\d+$", "", cand)            # sequence version suffix
    if ISOFORM_RE.match(cand):
        raise FujisanError(NO_ACCESSION, f"'{target_id}' is an isoform; AFDB has canonical entries only")
    if not ACCESSION_RE.match(cand):
        raise FujisanError(NO_ACCESSION, f"'{target_id}'")
    return cand


def hit_accession(hit: BlastHit) -> str:
    """Accession of a DB hit, from ``sseqid`` (else ``sacc``).

    The first word of the title is used only when the database carries no
    sequence IDs (``BL_ORD_ID`` / empty sseqid, i.e. built without
    -parse_seqids). The reason reported is that of the sequence ID itself.
    """
    ids = [c for c in (hit.sseqid, hit.sacc) if c and not c.startswith(("BL_ORD_ID", "gnl|BL_ORD_ID"))]
    if not ids and hit.stitle:
        ids = [hit.stitle.split()[0]]
    first = None
    for cand in ids:
        try:
            return parse_uniprot_accession(cand)
        except FujisanError as exc:
            first = first or exc
    raise first or FujisanError(NO_ACCESSION, hit.sseqid or "(no subject ID)")
