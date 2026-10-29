import json
import urllib.error

import pytest

from FUJISAN.errors import AFDB_UNAVAILABLE, INVALID_INPUT, PAE_UNAVAILABLE, FujisanError
from FUJISAN.features import alphafold as af
from FUJISAN.utils.cache import Cache

from helpers import make_pdb


@pytest.mark.parametrize("text, acc", [("Q9HBA0", "Q9HBA0"), ("q9hba0", "Q9HBA0"), ("AF-Q9HBA0-F1", "Q9HBA0"),
                                       ("AF-Q9HBA0-F1-model_v4", "Q9HBA0"), ("A0A0B4J2F0", "A0A0B4J2F0")])
def test_normalize(text, acc):
    assert af.normalize_afdb_id(text) == acc


def test_normalize_rejects():
    with pytest.raises(FujisanError) as e:
        af.normalize_afdb_id("not-an-id")
    assert e.value.reason == INVALID_INPUT


def _fake_get(calls, pae=True, status=None):
    def get(url):
        calls.append(url)
        if status:
            raise urllib.error.HTTPError(url, status, "x", {}, None)
        if "api/prediction" in url:
            e = {"uniprotAccession": "P12345", "modelEntityId": "AF-P12345-F1", "latestVersion": 6,
                 "sequence": "AGSL", "uniprotSequence": "AGSL", "uniprotStart": 1, "uniprotEnd": 4,
                 "pdbUrl": "https://x/model.pdb"}
            if pae:
                e["paeDocUrl"] = "https://x/pae.json"
            return json.dumps([e]).encode()
        if url.endswith(".pdb"):
            return make_pdb([90.0] * 4).encode()
        return json.dumps([{"predicted_aligned_error": [[0] * 4] * 4}]).encode()
    return get


def test_download_then_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(af, "_get", _fake_get(calls))
    cache = Cache(tmp_path)
    e1 = af.fetch_entry("AF-P12345-F1", cache)
    assert not e1.from_cache and e1.version == 6 and e1.structure_path.exists() and e1.pae_path.exists()
    n = len(calls)
    e2 = af.fetch_entry("P12345", cache)
    assert e2.from_cache and len(calls) == n                    # no new request
    info = cache.info()
    assert info["afdb_entries"] == 1 and info["structure_bytes"] > 0 and info["pae_bytes"] > 0


def test_404_is_no_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(af, "_get", _fake_get([], status=404))
    with pytest.raises(FujisanError) as e:
        af.fetch_entry("P12345", Cache(tmp_path))
    assert e.value.reason == AFDB_UNAVAILABLE


def test_missing_pae(tmp_path, monkeypatch):
    monkeypatch.setattr(af, "_get", _fake_get([], pae=False))
    with pytest.raises(FujisanError) as e:
        af.fetch_entry("P12345", Cache(tmp_path))
    assert e.value.reason == PAE_UNAVAILABLE


def test_retries_are_bounded(monkeypatch):
    calls = []

    class Boom:
        def __enter__(self):
            calls.append(1)
            raise urllib.error.URLError("down")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(af.urllib.request, "urlopen", lambda *a, **k: Boom())
    monkeypatch.setattr(af.time, "sleep", lambda s: None)
    with pytest.raises(urllib.error.URLError):
        af._get("https://example.invalid/x")
    assert len(calls) == af.MAX_RETRIES == 3
