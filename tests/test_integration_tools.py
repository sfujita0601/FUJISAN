"""Uses the real BLAST+ and TM-align binaries; skipped when they are not on PATH."""

import shutil

import pytest

from FUJISAN.features import blast as bl
from FUJISAN.features import tmalign as tm

from helpers import make_pdb

needs_tm = pytest.mark.skipif(not shutil.which("TMalign"), reason="TMalign not on PATH")
needs_blast = pytest.mark.skipif(not shutil.which("blastp"), reason="blastp not on PATH")


@needs_tm
def test_tmalign_runs_and_parses(tmp_path):
    a, b = tmp_path / "a.pdb", tmp_path / "b.pdb"
    a.write_text(make_pdb([90.0] * 40))
    b.write_text(make_pdb([90.0] * 45))
    r = tm.run_tmalign(str(a), str(b), shutil.which("TMalign"))
    assert 0 < r.tm_avg <= 1 and r.len1 == 40 and r.len2 == 45


@needs_blast
def test_blast_pair_identical():
    seq = "MFENITAAPADPILGLADLFRADERPGKINLGIGVYKDETGKTPVLTSVKKAEQYLLENETTKNYLGIDGIPEFGRCTQELLFGKGS"
    h = bl.blast_pair(seq, seq, shutil.which("blastp"))   # temporary one-sequence DB, same call as search
    assert h.pident == 100.0 and h.bitscore > 100 and h.length == len(seq)
