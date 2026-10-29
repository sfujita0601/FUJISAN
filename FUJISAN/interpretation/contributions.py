"""Per-pair feature contributions (SHAP values) of a FUJISAN LightGBM model.

The probability is computed with ``models.predict_proba`` (the function used for
prediction everywhere else), with the same model file and feature order. The
contributions come from LightGBM's own TreeSHAP (``Booster.predict(pred_contrib=True)``)
and are in log-odds (raw score) units: for every pair

    base_value + sum(contributions) == raw_score,   sigmoid(raw_score) == probability

Both identities are checked per pair and recorded in the output. Contributions
explain the five input features only; they are not assigned to residues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models import MODEL_CONFIGS, check_features, load_model, predict_proba  # noqa: F401 (load_model re-exported)

SHAP_METHOD = "lightgbm_pred_contrib"      # LightGBM TreeSHAP (path-dependent, tree_path_dependent in shap)
SHAP_SCALE = "log_odds"
SUM_TOLERANCE = 1e-9                       # |base + sum(contributions) - raw_score|
PROBABILITY_TOLERANCE = 1e-12              # |sigmoid(raw_score) - probability|


@dataclass
class Explained:
    """Arrays for n complete feature rows, in the model's feature order."""
    features: list
    X: pd.DataFrame
    probability: np.ndarray
    raw_score: np.ndarray
    base_value: np.ndarray
    contributions: np.ndarray            # shape (n, n_features)
    sum_abs_diff: np.ndarray
    probability_abs_diff: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def check_binary_sigmoid(model) -> None:
    """The log-odds reading of the contributions holds only for objective=binary with sigmoid 1."""
    objective = model.dump_model(num_iteration=0).get("objective", "")
    if not objective.startswith("binary") or "sigmoid:1" not in objective:
        raise ValueError(f"unsupported objective for log-odds contributions: {objective!r}")


def explain(model, X: pd.DataFrame) -> Explained:
    """Probability, raw score and contributions for complete rows of X (columns in model order)."""
    check_features(model, X)
    if X.isna().any().any():
        raise ValueError("missing feature values must not reach the model")
    check_binary_sigmoid(model)
    it = model.best_iteration
    prob = predict_proba(model, X)
    raw = model.predict(X, num_iteration=it, raw_score=True)
    contrib = model.predict(X, num_iteration=it, pred_contrib=True)
    n_feat = len(model.feature_name())
    if contrib.shape != (len(X), n_feat + 1):
        raise ValueError(f"unexpected pred_contrib shape {contrib.shape}")
    phi, base = contrib[:, :n_feat], contrib[:, n_feat]
    return Explained(
        features=list(model.feature_name()), X=X, probability=np.asarray(prob, float), raw_score=np.asarray(raw, float),
        base_value=base, contributions=phi,
        sum_abs_diff=np.abs(base + phi.sum(axis=1) - raw),
        probability_abs_diff=np.abs(_sigmoid(raw) - prob),
    )


def threshold_info(model_key: str) -> dict:
    """Threshold of the model and where it comes from; no threshold -> no binary call."""
    thr = MODEL_CONFIGS[model_key]["threshold"]
    if thr is None:
        return {"value": None, "source": "none: no validated threshold for this model (MODEL_CONFIGS)",
                "call_made": False}
    return {"value": float(thr), "source": "MODEL_CONFIGS: fixed threshold of the published model "
                                           "(Fujita & Terada, 2024), not optimised on the test set",
            "call_made": True}

