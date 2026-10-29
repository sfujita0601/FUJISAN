"""run_metadata.json: versions, parameters and inputs of one run."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..features import blast as bl
from ..features import domains as dm
from ..models import MODEL_CONFIGS

PAE_TO_DOMAINS_SOURCE = {
    "repository": "https://github.com/tristanic/pae_to_domains",
    "commit": "f407c6035c825f151a56f28bf803fcb44321b941",
    "vendored_file": "FUJISAN/vendor/pae_to_domains/pae_to_domains.py",
    "function": "domains_from_pae_matrix_igraph",
}


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(command: str, args: dict, tools: dict, model_name: str = "updated") -> dict:
    cfg = MODEL_CONFIGS[model_name]
    try:
        import igraph
        igraph_version = igraph.__version__
    except Exception:
        igraph_version = "unavailable"
    return {
        "fujisan_version": __version__,
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"name": model_name, "file": cfg["model_path"], "md5": _md5(cfg["model_path"]),
                  "features": cfg["features"], "threshold": cfg["threshold"]},
        "features": {
            "BitScore": bl.BITSCORE_DEFINITION,
            "Full_SeqID": bl.FULL_SEQID_DEFINITION,
            "Dmn_TM": "TM-align -a T TM-score normalised by the average length, best domain pair",
            "Dmn_RMSD": "TM-align RMSD of the same domain pair",
            "Dmn_SeqID": "TM-align Seq_ID (n_identical/n_aligned) of the same domain pair",
        },
        "blast": {"version": tools.get("blast_version", ""), "evalue": bl.EVALUE, "max_hsps": bl.MAX_HSPS,
                  "outfmt": "6 " + " ".join(bl.OUTFMT_FIELDS),
                  "dbsize": args.get("blast_dbsize") or "not set (BLAST default)",
                  "features_from": "blastp of each protein pair (first protein as query) against a temporary "
                                   "one-sequence database of the second; in search mode the database search only "
                                   "selects the candidates"},
        "tmalign": {"binary": tools.get("tmalign", ""), "version": tools.get("tmalign_version", ""),
                    "options": "-a T"},
        "pae_to_domains": {**PAE_TO_DOMAINS_SOURCE, "pae_power": dm.PAE_POWER, "pae_cutoff": dm.PAE_CUTOFF,
                           "graph_resolution": dm.GRAPH_RESOLUTION, "library": dm.PAE_LIBRARY,
                           "python_igraph": igraph_version, "leiden_seed": dm.LEIDEN_SEED},
        "domain_filter": {"plddt_rule": f"pLDDT >= {dm.PLDDT_MIN:g}", "min_domain_residues": dm.MIN_DOMAIN_RESIDUES},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items()},
    }


def write_metadata(path: Path, meta: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    tmp.replace(path)
