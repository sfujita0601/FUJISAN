"""AlphaFold Protein Structure Database (AFDB): structure + PAE for a UniProt accession.

The file URLs are taken from the official prediction API
(``https://alphafold.ebi.ac.uk/api/prediction/{accession}``: ``pdbUrl``,
``paeDocUrl``, ``latestVersion``, ``sequence`` ...) rather than from a
hard-coded file name pattern, so the current AFDB version is used. Note that
AFDB serves only its latest model version; the FUJISAN training data used
AFDB v4 models (see README, Limitations).

Downloads are cached per accession and model version (``FUJISAN.utils.cache``)
and are not repeated. Requests are throttled and retried at most
``MAX_RETRIES`` times; a failure for one target is raised as FujisanError so
that the caller can skip the target and record the reason.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..errors import AFDB_NETWORK, AFDB_UNAVAILABLE, INVALID_INPUT, PAE_UNAVAILABLE, STRUCTURE_UNAVAILABLE, FujisanError
from .blast import ACCESSION_RE

API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
TIMEOUT = 60
MAX_RETRIES = 3
MIN_INTERVAL = 0.3          # seconds between requests (all threads)
USER_AGENT = "FUJISAN feature pipeline (https://github.com/sfujita0601/FUJISAN)"

_lock = threading.Lock()
_last_request = [0.0]
_slots = threading.BoundedSemaphore(2)       # at most two AFDB requests at a time


def normalize_afdb_id(text: str) -> str:
    """UniProt accession from ``Q9HBA0``, ``q9hba0``, ``AF-Q9HBA0-F1`` or ``AF-Q9HBA0-F1-model_v4``."""
    t = text.strip()
    m = re.match(r"^AF-([A-Za-z0-9]+)-F\d+(?:-model_v\d+)?(?:\.(?:pdb|cif))?$", t)
    acc = (m.group(1) if m else t).upper()
    if not ACCESSION_RE.match(acc):
        raise FujisanError(INVALID_INPUT, f"'{text}' is not a UniProt accession or AFDB ID")
    return acc


def _get(url: str) -> bytes:
    """HTTP GET with throttling and at most MAX_RETRIES attempts (no retry on 404)."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        with _slots:
            with _lock:
                wait = _last_request[0] + MIN_INTERVAL - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                _last_request[0] = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    return r.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 404, 410):
                    raise
                last = exc
                delay = 2 ** attempt
                if exc.code == 429:                       # rate limited: honour Retry-After (capped)
                    try:
                        delay = min(60, int(exc.headers.get("Retry-After", delay)))
                    except (TypeError, ValueError):
                        pass
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
                delay = 2 ** attempt
        if attempt < MAX_RETRIES:
            time.sleep(delay)
    raise last


@dataclass
class AFDBEntry:
    accession: str
    model_id: str            # e.g. AF-P12345-F1
    version: int
    sequence: str
    structure_path: Path
    pae_path: Path
    metadata_path: Path
    from_cache: bool
    note: str = ""


def fetch_entry(accession: str, cache, refresh: bool = False) -> AFDBEntry:
    """Structure (PDB, pLDDT in B-factors) and PAE of the AFDB model of a UniProt accession."""
    acc = normalize_afdb_id(accession)
    cached = None if refresh else cache.find_afdb(acc)
    if cached:
        meta = json.loads(cached.read_text())
        d = cached.parent
        return AFDBEntry(acc, meta["model_id"], meta["version"], meta["sequence"], d / meta["structure_file"],
                         d / meta["pae_file"], cached, True, meta.get("note", ""))
    try:
        entries = json.loads(_get(API_URL.format(accession=acc)))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FujisanError(AFDB_UNAVAILABLE, f"{acc}: AFDB has no model") from exc
        raise FujisanError(AFDB_NETWORK, f"{acc}: API HTTP {exc.code}") from exc
    except Exception as exc:
        raise FujisanError(AFDB_NETWORK, f"{acc}: API request failed ({exc})") from exc
    if not entries:
        raise FujisanError(AFDB_UNAVAILABLE, f"{acc}: empty API response")
    mine = [e for e in entries if e.get("uniprotAccession", "").upper() == acc] or entries
    entry = next((e for e in mine if str(e.get("modelEntityId", "")).endswith("-F1")), mine[0])
    model_id, version = entry.get("modelEntityId", f"AF-{acc}-F1"), int(entry.get("latestVersion", 0))
    note = ""
    useq = entry.get("uniprotSequence") or ""
    if entry.get("uniprotStart") not in (None, 1) or (useq and entry.get("uniprotEnd") not in (None, len(useq))):
        note = (f"AFDB model covers residues {entry.get('uniprotStart')}-{entry.get('uniprotEnd')} "
                f"of {len(useq)} (fragment)")
    pdb_url, pae_url = entry.get("pdbUrl"), entry.get("paeDocUrl")
    if not pdb_url:
        raise FujisanError(STRUCTURE_UNAVAILABLE, f"{acc}: no pdbUrl in the AFDB entry")
    if not pae_url:
        raise FujisanError(PAE_UNAVAILABLE, f"{acc}: no paeDocUrl in the AFDB entry")
    try:
        pdb = _get(pdb_url)
    except Exception as exc:
        raise FujisanError(STRUCTURE_UNAVAILABLE, f"{acc}: {pdb_url} ({exc})") from exc
    try:
        pae = _get(pae_url)
    except Exception as exc:
        raise FujisanError(PAE_UNAVAILABLE, f"{acc}: {pae_url} ({exc})") from exc
    meta = {"accession": acc, "model_id": model_id, "version": version,
            "sequence": entry.get("sequence") or entry.get("uniprotSequence") or "",
            "structure_file": "model.pdb", "pae_file": "pae.json", "pdb_url": pdb_url, "pae_url": pae_url,
            "downloaded": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
            "api_entry": {k: v for k, v in entry.items() if not isinstance(v, (list, dict))}}
    meta_path = cache.store_afdb(acc, model_id, version, pdb, pae, meta)
    d = meta_path.parent
    return AFDBEntry(acc, model_id, version, meta["sequence"], d / "model.pdb", d / "pae.json", meta_path, False, note)
