"""Interpretation of FUJISAN predictions (shared by the ``fujisan explain`` CLI and the GUI).

Phase 1: per-pair feature contributions (log-odds SHAP values) with provenance.
Structure / residue information is not connected yet (``structure.status == "not_connected"``).
"""

from .contributions import SHAP_METHOD, SHAP_SCALE, explain, threshold_info
from .schema import (SCHEMA_NAME, SCHEMA_VERSION, explain_table, read_jsonl, records_to_frame, validate_record,
                     write_jsonl)

__all__ = ["SHAP_METHOD", "SHAP_SCALE", "SCHEMA_NAME", "SCHEMA_VERSION", "explain", "explain_table",
           "read_jsonl", "records_to_frame", "threshold_info", "validate_record", "write_jsonl"]
