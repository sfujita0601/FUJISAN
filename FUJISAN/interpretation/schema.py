"""Interpretation records (``fujisan.interpretation`` JSONL), shared by the CLI and the GUI.

One JSON object per protein pair and line. See docs/interpretation_schema.md for the
field definitions. Records are self-contained (provenance repeated per line) so that
files can be concatenated, filtered and joined with server outputs line by line.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..models import MODEL_CONFIGS
from . import contributions as C

SCHEMA_NAME = "fujisan.interpretation"
SCHEMA_VERSION = "1.0.0"

# SHA256 of the model files as released; a different file is reported (model_sha256_matches_release=false).
RELEASED_MODEL_SHA256 = {
    "updated": "dccf13277362bcc60add91d0ee3a8881494bbc2604123b8213cad6e3c3fb92a8",
    "paper2024": "9a22806c48d5d45ae6f7ff030ab14befffbb1fe48e12d0278a65cbb23fcd1a8f",
}

# Server-side residue mapping (Phase 2). Until these files are received and connected,
# every record says so; no domain pair or residue correspondence is inferred locally.
STRUCTURE_NOT_CONNECTED = {
    "status": "not_connected",
    "reason": "server outputs (selected_domain_pairs_41600.tsv, residue_alignment_41600.tsv) not connected",
    "selected_domain_pair": None,
    "is_max_tm_pair": None,
    "superposition": None,
    "residue_alignment": None,
}

ID_COLUMN_PAIRS = [("ProteinID1", "ProteinID2"), ("QueryID", "TargetID"), ("query_id", "target_id")]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fujisan_commit() -> dict:
    """Commit of the FUJISAN checkout this code runs from ("unknown" for an installed wheel)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return {"commit": "unknown", "dirty": None}
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True,
                                        check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def provenance(model_key: str) -> dict:
    import lightgbm as lgb
    path = MODEL_CONFIGS[model_key]["model_path"]
    sha = sha256_file(path)
    commit = fujisan_commit()
    return {
        "model_key": model_key,
        "model_file": Path(path).name,
        "model_sha256": sha,
        "model_sha256_matches_release": sha == RELEASED_MODEL_SHA256.get(model_key),
        "fujisan_version": __version__,
        "fujisan_commit": commit["commit"],
        "fujisan_commit_dirty": commit["dirty"],
        "lightgbm_version": lgb.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _id_columns(df: pd.DataFrame, id_cols) -> tuple:
    if id_cols:
        if len(id_cols) != 2 or any(c not in df.columns for c in id_cols):
            raise ValueError(f"ID columns {list(id_cols)} not found (need two existing columns: query, target)")
        return tuple(id_cols)
    for a, b in ID_COLUMN_PAIRS:
        if a in df.columns and b in df.columns:
            return a, b
    raise ValueError(f"no protein ID columns; expected one of {ID_COLUMN_PAIRS} or explicit id_cols")


def _num(x):
    """JSON-safe number: None for missing and non-finite values."""
    return float(x) if x is not None and np.isfinite(x) else None


def _id(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip()


def explain_table(df: pd.DataFrame, model_key: str = "updated", id_cols=None, rename=None,
                  reference_probability_col: str | None = None, source: dict | None = None) -> list:
    """One interpretation record per row of df, in input order.

    Rows with a missing, non-numeric or non-finite feature, or without both protein IDs, get
    status "skipped" (nothing is filled in). Duplicate pairs are kept; source.row tells them apart.

    rename maps input column names to model feature names (e.g. {"Full_Bitscore": "BitScore"} for
    the GUI table). reference_probability_col, if given, is a probability computed elsewhere for the
    same row (e.g. the GUI's FUJISAN_Score); its difference to this run is recorded, not corrected.
    """
    model = C.load_model(model_key)
    feats = list(model.feature_name())
    q_col, t_col = _id_columns(df, id_cols)
    data = df.rename(columns=rename or {})
    missing = [f for f in feats if f not in data.columns]
    if missing:
        raise ValueError(f"input lacks model features {missing}")
    dup = sorted({c for c in data.columns[data.columns.duplicated()] if c in feats})
    if dup:
        raise ValueError(f"feature columns occur more than once (after renaming): {dup}")
    X_all = data[feats].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(X_all.to_numpy(dtype=float))
    ids = [(_id(a), _id(b)) for a, b in zip(df[q_col], df[t_col])]
    has_ids = np.array([bool(q and t) for q, t in ids], dtype=bool)
    ok = finite.all(axis=1) & has_ids
    ex = C.explain(model, X_all[ok].astype(float)) if ok.any() else None
    prov = provenance(model_key)
    thr = C.threshold_info(model_key)
    src = dict(source or {})
    out, k = [], 0
    for i in range(len(df)):
        q, t = ids[i]
        key = f"{q}_{t}" if has_ids[i] else None
        rec = {
            "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
            "provenance": prov,
            "pair": {"pair_id": key, "query_id": q or None, "target_id": t or None,
                     "source": {**src, "row": i + 1, "id_columns": [q_col, t_col]}},
            "features": {"names": feats, "values": {f: _num(X_all[f].iloc[i]) for f in feats}},
            "link": {"pair_key": key, "pair_key_format": "ProteinID1_ProteinID2 (query first)",
                     "server_manifest": None, "server_match_status": "not_checked"},
            "structure": dict(STRUCTURE_NOT_CONNECTED),
        }
        if not ok[i]:
            reasons = [] if has_ids[i] else ["missing protein ID"]
            bad = [f for j, f in enumerate(feats) if not finite[i, j]]
            if bad:
                reasons.append("missing, non-numeric or non-finite feature: " + ", ".join(bad))
            rec.update(status="skipped", skip_reason="; ".join(reasons), prediction=None, shap=None)
        else:
            p, raw = float(ex.probability[k]), float(ex.raw_score[k])
            call = int(p >= thr["value"]) if thr["call_made"] else None
            pred = {"raw_score": raw, "raw_score_scale": C.SHAP_SCALE, "probability": p,
                    "probability_function": "FUJISAN.models.predict_proba",
                    "sigmoid_check": {"abs_diff": float(ex.probability_abs_diff[k]),
                                      "tolerance": C.PROBABILITY_TOLERANCE,
                                      "passed": bool(ex.probability_abs_diff[k] <= C.PROBABILITY_TOLERANCE)},
                    "threshold": thr["value"], "threshold_source": thr["source"], "call": call}
            if reference_probability_col:
                ref = _num(pd.to_numeric(df[reference_probability_col].iloc[i], errors="coerce"))
                pred["reference_probability"] = {
                    "column": reference_probability_col, "value": ref,
                    "abs_diff": None if ref is None else abs(ref - p)}
            s = float(ex.base_value[k] + ex.contributions[k].sum())
            rec.update(status="ok", prediction=pred, shap={
                "method": C.SHAP_METHOD, "scale": C.SHAP_SCALE,
                "base_value": float(ex.base_value[k]),
                "contributions": {f: float(v) for f, v in zip(feats, ex.contributions[k])},
                "sum_check": {"base_plus_sum": s, "raw_score": raw, "abs_diff": float(ex.sum_abs_diff[k]),
                              "tolerance": C.SUM_TOLERANCE, "passed": bool(ex.sum_abs_diff[k] <= C.SUM_TOLERANCE)},
                "note": "contributions explain the model input features in log-odds; "
                        "they are not probabilities and are not assigned to residues",
            })
            k += 1
        out.append(rec)
    return out


REQUIRED_KEYS = ["schema", "schema_version", "provenance", "pair", "features", "status", "prediction", "shap",
                 "link", "structure"]


def validate_record(rec: dict) -> list:
    """Problems found in one record (empty list = valid)."""
    errs = [f"missing key {k}" for k in REQUIRED_KEYS if k not in rec]
    if errs:
        return errs
    if rec["schema"] != SCHEMA_NAME:
        errs.append(f"schema {rec['schema']!r}")
    if rec["status"] == "ok":
        sh, pr = rec["shap"], rec["prediction"]
        if not rec["pair"]["pair_id"] or rec["link"]["pair_key"] != rec["pair"]["pair_id"]:
            errs.append("ok record without pair_id / pair_key")
        if sh["scale"] != C.SHAP_SCALE:
            errs.append(f"shap scale {sh['scale']!r}")
        if set(sh["contributions"]) != set(rec["features"]["names"]):
            errs.append("contribution names differ from feature names")
        diff = abs(sh["base_value"] + sum(sh["contributions"].values()) - pr["raw_score"])
        if diff > C.SUM_TOLERANCE or not sh["sum_check"]["passed"]:
            errs.append(f"sum check failed ({diff:.3g})")
        if pr["threshold"] is None and pr["call"] is not None:
            errs.append("binary call without a threshold")
    elif rec["status"] == "skipped":
        if rec["prediction"] is not None or rec["shap"] is not None:
            errs.append("skipped record carries a prediction")
        if not rec.get("skip_reason"):
            errs.append("skipped record without skip_reason")
    else:
        errs.append(f"status {rec['status']!r}")
    if rec["structure"].get("status") == "not_connected" and rec["structure"].get("residue_alignment") is not None:
        errs.append("residue alignment present while structure is not connected")
    return errs


def write_jsonl(path, records: list) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def records_to_frame(records: list) -> pd.DataFrame:
    """Flat table for display/TSV: features, contributions (shap_<feature>, log-odds) and checks."""
    rows = []
    for r in records:
        row = {"pair_id": r["pair"]["pair_id"], "query_id": r["pair"]["query_id"],
               "target_id": r["pair"]["target_id"], "status": r["status"]}
        row.update(r["features"]["values"])
        if r["status"] == "ok":
            pr, sh = r["prediction"], r["shap"]
            row.update(probability=pr["probability"], raw_score_log_odds=pr["raw_score"],
                       threshold=pr["threshold"], call=pr["call"], base_value_log_odds=sh["base_value"])
            row.update({f"shap_{f}_log_odds": v for f, v in sh["contributions"].items()})
            row.update(sum_check_abs_diff=sh["sum_check"]["abs_diff"], sum_check_passed=sh["sum_check"]["passed"])
            if "reference_probability" in pr:
                row["reference_probability_abs_diff"] = pr["reference_probability"]["abs_diff"]
        else:
            row["skip_reason"] = r.get("skip_reason", "")
        row["structure_status"] = r["structure"]["status"]
        row["model_sha256"] = r["provenance"]["model_sha256"]
        rows.append(row)
    return pd.DataFrame(rows)
