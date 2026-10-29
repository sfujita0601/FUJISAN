import json

import numpy as np
import pytest

from FUJISAN.errors import INVALID_PAE, FujisanError
from FUJISAN.features.domains import load_pae, parse_pae

M = [[0.0, 1.5, 20.0], [1.2, 0.0, 18.5], [19.0, 17.0, 0.0]]


def test_afdb_current_format(tmp_path):
    p = tmp_path / "pae.json"
    p.write_text(json.dumps([{"predicted_aligned_error": M, "max_predicted_aligned_error": 31.75}]))
    m = load_pae(p)
    assert m.shape == (3, 3) and m[0, 2] == 20.0 and m[2, 0] == 19.0   # row = aligned residue, not transposed


def test_colabfold_format():
    assert parse_pae({"pae": M, "plddt": [90, 90, 90]})[1, 2] == 18.5


def test_legacy_afdb_format():
    legacy = [{"residue1": [1, 1, 2, 2], "residue2": [1, 2, 1, 2], "distance": [0.0, 3.0, 4.0, 0.0]}]
    m = parse_pae(legacy)
    assert m.tolist() == [[0.0, 3.0], [4.0, 0.0]]


@pytest.mark.parametrize("bad", [{"foo": 1}, [], {"pae": [[1, 2]]}, {"pae": [[0, float("nan")], [1, 0]]}, "x"])
def test_invalid_pae(bad):
    with pytest.raises(FujisanError) as e:
        parse_pae(bad)
    assert e.value.reason == INVALID_PAE


def test_not_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("not json")
    with pytest.raises(FujisanError):
        load_pae(p)
