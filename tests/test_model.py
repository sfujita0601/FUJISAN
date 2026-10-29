import pytest

from FUJISAN.errors import FujisanError
from FUJISAN.models import MODEL_CONFIGS, load_model
from FUJISAN.prediction import FEATURES, add_probabilities, feature_frame, get_model

ORDER = ["BitScore", "Full_SeqID", "Dmn_RMSD", "Dmn_SeqID", "Dmn_TM"]


def test_feature_order():
    assert FEATURES == ORDER
    row = {"Dmn_TM": 0.97, "BitScore": 289.0, "Dmn_SeqID": 0.38, "Full_SeqID": 39.4, "Dmn_RMSD": 0.97, "extra": 1}
    assert list(feature_frame([row]).columns) == ORDER


def test_model_schema_matches():
    from FUJISAN.models import MODEL_CONFIGS as C
    assert C["updated"]["model_path"].endswith("fujisan_updated_blastbitscore_nopocket_lgbm.txt")
    assert get_model().feature_name() == ORDER
    assert MODEL_CONFIGS["updated"]["features"] == ORDER
    assert MODEL_CONFIGS["updated"]["threshold"] is None        # no validated threshold -> no binary call


def test_paper2024_model_untouched():
    assert load_model("paper2024").feature_name()[-2:] == ["Pckt_RMSD", "Pckt_SeqID"]


def test_missing_value_is_not_filled():
    with pytest.raises(FujisanError):
        feature_frame([{"BitScore": 289.0, "Full_SeqID": 39.4, "Dmn_RMSD": None, "Dmn_SeqID": 0.38, "Dmn_TM": 0.97}])


def test_probability_regression():
    # P00509 vs Q22067 (AFDB v4 models), features as computed by the pipeline; updated model retrained
    # on raw BLAST bit scores (model/fujisan_updated_blastbitscore_nopocket_lgbm.txt)
    row = {"BitScore": 289.0, "Full_SeqID": 39.394, "Dmn_RMSD": 0.97, "Dmn_SeqID": 0.384, "Dmn_TM": 0.97094}
    out = add_probabilities([row])[0]
    assert out["FUJISAN_probability"] == pytest.approx(0.9102717776, abs=1e-8)
    assert "prediction" not in out and "label" not in out
