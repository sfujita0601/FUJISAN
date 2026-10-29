from pathlib import Path

import pytest

from FUJISAN.errors import NO_DOMAIN_PAIR, TMALIGN_FAILED, FujisanError
from FUJISAN.features import tmalign as tm

FIXTURE = Path(__file__).parent / "data" / "tmalign_a_T.txt"


def test_parse_real_output():
    r = tm.parse_tmalign(FIXTURE.read_text())
    assert (r.tm1, r.tm2, r.tm_avg) == (0.98303, 0.95915, 0.97094)
    assert (r.len1, r.len2, r.aligned_length) == (396, 406, 396)
    assert (r.rmsd, r.seq_id) == (0.97, 0.384)
    assert r.tm_avg != pytest.approx((r.tm1 + r.tm2) / 2, abs=1e-5)   # -a T value, not the arithmetic mean


def test_missing_average_score_is_an_error():
    text = "\n".join(l for l in FIXTURE.read_text().splitlines() if "average length" not in l)
    with pytest.raises(FujisanError) as e:
        tm.parse_tmalign(text)
    assert e.value.reason == TMALIGN_FAILED


def _pair(d1, d2, avg, rmsd, sid):
    return tm.DomainPairResult(d1, d2, tm.TMResult(avg + 0.01, avg - 0.01, avg, 100, 100, 90, rmsd, sid))


def test_best_pair_and_same_pair_values():
    rs = [_pair(1, 1, 0.55, 2.0, 0.30), _pair(1, 2, 0.81, 3.5, 0.12), _pair(2, 1, 0.80, 0.9, 0.95)]
    best = tm.select_best_pair(rs)
    assert (best.domain1, best.domain2) == (1, 2)
    assert (best.result.tm_avg, best.result.rmsd, best.result.seq_id) == (0.81, 3.5, 0.12)   # not min RMSD / max SeqID


def test_tie_keeps_first_pair():
    best = tm.select_best_pair([_pair(1, 1, 0.7, 1.0, 0.5), _pair(2, 2, 0.7, 0.5, 0.9)])
    assert (best.domain1, best.domain2) == (1, 1)


def test_no_pair():
    with pytest.raises(FujisanError) as e:
        tm.select_best_pair([])
    assert e.value.reason == NO_DOMAIN_PAIR
