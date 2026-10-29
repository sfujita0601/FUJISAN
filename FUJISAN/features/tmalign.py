"""TM-align of all domain pairs and selection of the best pair.

Command: ``TMalign domainA.pdb domainB.pdb -a T`` (as for the training data).
TM-align then reports three TM-scores, normalised by the length of
structure 1, of structure 2, and by the average length of the two (``-a T``).

``Dmn_TM`` is the third value, the TM-score normalised by the average chain
length. This is the "average TM-score" of the paper (Section 2.2.2) and the
``TM-scoreAve`` column of the training / test data: in test/same.tsv and
test/diff.tsv it lies between TM-score1 and TM-score2 for every pair but
equals their arithmetic mean for only 13 % of them, so it is *not*
(TM1 + TM2) / 2. The arithmetic mean is still reported as metadata
(``TMscoreMean``).

``Dmn_RMSD`` and ``Dmn_SeqID`` (``Seq_ID=n_identical/n_aligned``) come from
the same domain pair as ``Dmn_TM``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from ..errors import NO_DOMAIN_PAIR, TMALIGN_FAILED, TMALIGN_NOT_FOUND, FujisanError

RE_TM1 = re.compile(r"TM-score=\s*([\d.]+)\s*\((?:if )?normalized by length of (?:Chain|Structure)_1")
RE_TM2 = re.compile(r"TM-score=\s*([\d.]+)\s*\((?:if )?normalized by length of (?:Chain|Structure)_2")
RE_TMAVG = re.compile(r"TM-score=\s*([\d.]+)\s*\((?:if )?normalized by average length")
RE_LEN1 = re.compile(r"Length of (?:Chain_1|Structure_1):\s*(\d+)")
RE_LEN2 = re.compile(r"Length of (?:Chain_2|Structure_2):\s*(\d+)")
RE_ALN = re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+),\s*Seq_ID=n_identical/n_aligned=\s*([\d.]+)")
RE_VERSION = re.compile(r"TM-align \(Version (\w+)\)")


@dataclass
class TMResult:
    tm1: float          # normalised by length of structure 1 (protein A domain)
    tm2: float          # normalised by length of structure 2 (protein B domain)
    tm_avg: float       # normalised by the average length (-a T)  -> Dmn_TM
    len1: int
    len2: int
    aligned_length: int
    rmsd: float         # -> Dmn_RMSD
    seq_id: float       # -> Dmn_SeqID
    raw: str = ""


def find_tmalign(binary: str = "TMalign") -> str:
    path = shutil.which(binary)
    if not path:
        raise FujisanError(TMALIGN_NOT_FOUND, f"'{binary}' (install TM-align and put it on PATH or use --tmalign-bin)")
    return path


def tmalign_version(binary: str) -> str:
    try:
        out = subprocess.run([binary], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return "unknown"
    m = RE_VERSION.search(out)
    return m.group(1) if m else "unknown"


def parse_tmalign(text: str) -> TMResult:
    """Values of one TM-align run with ``-a T``."""
    t1, t2, ta = RE_TM1.search(text), RE_TM2.search(text), RE_TMAVG.search(text)
    if not (t1 and t2):
        raise FujisanError(TMALIGN_FAILED, "cannot parse the TM-scores")
    if not ta:
        raise FujisanError(TMALIGN_FAILED, "no average-length TM-score in the output (TM-align must be run with -a T)")
    l1, l2, aln = RE_LEN1.search(text), RE_LEN2.search(text), RE_ALN.search(text)
    if not (l1 and l2 and aln):
        raise FujisanError(TMALIGN_FAILED, "cannot parse lengths / RMSD / Seq_ID")
    return TMResult(float(t1.group(1)), float(t2.group(1)), float(ta.group(1)), int(l1.group(1)), int(l2.group(1)),
                    int(aln.group(1)),
                    float(aln.group(2)), float(aln.group(3)), text)


def run_tmalign(pdb1: str, pdb2: str, binary: str, timeout: int = 600) -> TMResult:
    try:
        proc = subprocess.run([binary, pdb1, pdb2, "-a", "T"], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FujisanError(TMALIGN_FAILED, f"timeout after {timeout} s") from exc
    if proc.returncode != 0:
        raise FujisanError(TMALIGN_FAILED, (proc.stderr or proc.stdout).strip()[-300:])
    return parse_tmalign(proc.stdout)


@dataclass
class DomainPairResult:
    domain1: int
    domain2: int
    result: TMResult


def align_all(domains1: list, domains2: list, binary: str) -> tuple[list, list]:
    """TM-align every domain of protein A against every domain of protein B.

    Returns (successful pair results, failures as (domain1, domain2, message)).
    """
    results, failures = [], []
    for d1 in domains1:
        for d2 in domains2:
            try:
                results.append(DomainPairResult(d1.number, d2.number, run_tmalign(d1.pdb_path, d2.pdb_path, binary)))
            except FujisanError as exc:
                failures.append((d1.number, d2.number, exc.detail or exc.reason))
    return results, failures


def select_best_pair(results: list) -> DomainPairResult:
    """Domain pair with the highest average-length TM-score (first one on ties).

    Dmn_TM, Dmn_RMSD and Dmn_SeqID are all taken from this one pair.
    """
    if not results:
        raise FujisanError(NO_DOMAIN_PAIR)
    best = results[0]
    for r in results[1:]:
        if r.result.tm_avg > best.result.tm_avg:
            best = r
    return best
