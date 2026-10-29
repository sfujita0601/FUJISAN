"""FUJISAN probability from the five features of the ``updated`` model.

Only complete feature rows are passed to the model; rows with a missing
feature are never filled in (they are reported as skipped instead). No binary
call is made: the updated model has no validated classification threshold.
"""

from __future__ import annotations

import sys

import pandas as pd

from .errors import MODEL_MISMATCH, FujisanError
from .models import MODEL_CONFIGS, load_model, predict_proba

MODEL_NAME = "updated"
FEATURES = list(MODEL_CONFIGS[MODEL_NAME]["features"])   # BitScore, Full_SeqID, Dmn_RMSD, Dmn_SeqID, Dmn_TM

_model = None


class _StderrLogger:
    """LightGBM messages go to stderr so that CLI tables on stdout stay clean
    (e.g. a model saved by a newer LightGBM version warns about unknown parameters)."""

    def info(self, msg):
        print(msg, file=sys.stderr)

    def warning(self, msg):
        print(msg, file=sys.stderr)


def get_model():
    global _model
    if _model is None:
        import lightgbm as lgb
        lgb.register_logger(_StderrLogger())
        try:
            _model = load_model(MODEL_NAME)
        except ValueError as exc:
            raise FujisanError(MODEL_MISMATCH, str(exc)) from exc
        if _model.feature_name() != FEATURES:
            raise FujisanError(MODEL_MISMATCH, f"model expects {_model.feature_name()}, pipeline makes {FEATURES}")
    return _model


def feature_frame(rows: list) -> pd.DataFrame:
    """Model input in the exact feature order of the model file."""
    X = pd.DataFrame([{f: r[f] for f in FEATURES} for r in rows], columns=FEATURES).astype(float)
    if X.isna().any().any():
        raise FujisanError(MODEL_MISMATCH, "missing feature values must not reach the model")
    return X


def add_probabilities(rows: list) -> list:
    """Return the rows with ``FUJISAN_probability`` added."""
    if not rows:
        return rows
    prob = predict_proba(get_model(), feature_frame(rows))
    return [dict(r, FUJISAN_probability=float(p)) for r, p in zip(rows, prob)]
