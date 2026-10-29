"""On-disk cache of AFDB downloads.

Layout (default ``~/.cache/fujisan``, or ``--cache-dir``, or ``$FUJISAN_CACHE_DIR``)::

    <cache>/afdb/<accession>/<model_id>-v<version>/model.pdb
                                                  /pae.json
                                                  /metadata.json   (URLs, download time, sequence)

An entry is written to a temporary directory and renamed into place, so an
interrupted download never leaves a half-written entry. The cache is only
ever written below ``<cache>/afdb``; FUJISAN never deletes anything outside it.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("FUJISAN_CACHE_DIR") or Path.home() / ".cache" / "fujisan")


class Cache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root else DEFAULT_CACHE
        self.afdb = self.root / "afdb"

    def find_afdb(self, accession: str) -> Path | None:
        """metadata.json of the newest cached model of an accession."""
        d = self.afdb / accession
        if not d.is_dir():
            return None
        metas = []
        for m in d.glob("*/metadata.json"):
            try:
                v = json.loads(m.read_text()).get("version", 0)
            except ValueError:
                continue
            if (m.parent / "model.pdb").exists() and (m.parent / "pae.json").exists():
                metas.append((v, m))
        return max(metas)[1] if metas else None

    def store_afdb(self, accession: str, model_id: str, version: int, pdb: bytes, pae: bytes, meta: dict) -> Path:
        dest = self.afdb / accession / f"{model_id}-v{version}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_", dir=dest.parent))
        (tmp / "model.pdb").write_bytes(pdb)
        (tmp / "pae.json").write_bytes(pae)
        (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
        if dest.exists():                 # another worker stored it meanwhile
            shutil.rmtree(tmp)
        else:
            os.replace(tmp, dest)
        return dest / "metadata.json"

    def info(self) -> dict:
        n, s_bytes, p_bytes, other = 0, 0, 0, 0
        if self.afdb.is_dir():
            n = sum(1 for d in self.afdb.iterdir() if d.is_dir())
        for f in self.root.rglob("*") if self.root.exists() else []:
            if not f.is_file():
                continue
            size = f.stat().st_size
            if f.name == "model.pdb" or f.suffix in (".cif", ".pdb"):
                s_bytes += size
            elif f.name == "pae.json":
                p_bytes += size
            else:
                other += size
        return {"cache_dir": str(self.root), "afdb_entries": n, "structure_bytes": s_bytes, "pae_bytes": p_bytes,
                "other_bytes": other, "total_bytes": s_bytes + p_bytes + other}


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"
