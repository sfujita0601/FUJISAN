import json

import numpy as np
import pandas as pd
import pytest

from FUJISAN.cli import main
from FUJISAN.interpretation import (SCHEMA_VERSION, explain_table, read_jsonl, records_to_frame, validate_record,
                                    write_jsonl)
from FUJISAN.interpretation.schema import RELEASED_MODEL_SHA256
from FUJISAN.models import MODEL_CONFIGS, load_model, predict_proba
from FUJISAN.prediction import FEATURES, add_probabilities

# Same pair as test_model.test_probability_regression
REF = {"ProteinID1": "P00509", "ProteinID2": "Q22067", "BitScore": 289.0, "Full_SeqID": 39.394,
       "Dmn_RMSD": 0.97, "Dmn_SeqID": 0.384, "Dmn_TM": 0.97094}


def _random_table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ProteinID1": [f"A{i}" for i in range(n)], "ProteinID2": [f"B{i}" for i in range(n)],
        "BitScore": rng.uniform(20, 2000, n), "Full_SeqID": rng.uniform(10, 100, n),
        "Dmn_RMSD": rng.uniform(0.2, 6, n), "Dmn_SeqID": rng.uniform(0.05, 1, n), "Dmn_TM": rng.uniform(0.2, 1, n)})


def test_probability_identical_to_prediction_path():
    rec = explain_table(pd.DataFrame([REF]))[0]
    assert rec["status"] == "ok"
    expected = add_probabilities([{f: REF[f] for f in FEATURES}])[0]["FUJISAN_probability"]
    assert rec["prediction"]["probability"] == expected                 # bit-identical, same function
    assert rec["prediction"]["probability"] == pytest.approx(0.9102717776, abs=1e-8)


def test_sum_and_sigmoid_checks_random():
    df = _random_table()
    recs = explain_table(df)
    model = load_model("updated")
    prob = predict_proba(model, df[FEATURES])
    for r, p in zip(recs, prob):
        assert validate_record(r) == []
        assert r["prediction"]["probability"] == p
        sh = r["shap"]
        assert sh["sum_check"]["passed"] and r["prediction"]["sigmoid_check"]["passed"]
        assert sh["scale"] == "log_odds"
        assert list(sh["contributions"]) == FEATURES


def test_matches_shap_tree_explainer():
    shap = pytest.importorskip("shap")
    df = _random_table(50, seed=1)
    recs = explain_table(df)
    sv = shap.TreeExplainer(load_model("updated")).shap_values(df[FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    ours = np.array([[r["shap"]["contributions"][f] for f in FEATURES] for r in recs])
    assert np.allclose(ours, sv, atol=1e-8)


def test_no_threshold_no_call():
    rec = explain_table(pd.DataFrame([REF]))[0]
    assert MODEL_CONFIGS["updated"]["threshold"] is None
    assert rec["prediction"]["threshold"] is None and rec["prediction"]["call"] is None
    assert rec["prediction"]["threshold_source"].startswith("none")


def test_missing_feature_skipped_not_filled():
    row = dict(REF, Dmn_RMSD=None)
    recs = explain_table(pd.DataFrame([REF, row]))
    assert [r["status"] for r in recs] == ["ok", "skipped"]
    assert recs[1]["prediction"] is None and recs[1]["shap"] is None
    assert "Dmn_RMSD" in recs[1]["skip_reason"]
    assert validate_record(recs[1]) == []
    assert recs[0]["prediction"]["probability"] == explain_table(pd.DataFrame([REF]))[0]["prediction"]["probability"]


def test_column_order_and_rename():
    gui = pd.DataFrame([{"query_id": "P00509", "target_id": "Q22067", "Dmn_TM": REF["Dmn_TM"],
                         "Full_Bitscore": REF["BitScore"], "Dmn_SeqID": REF["Dmn_SeqID"],
                         "Full_SeqID": REF["Full_SeqID"], "Dmn_RMSD": REF["Dmn_RMSD"], "FUJISAN_Score": 0.9102717776}])
    rec = explain_table(gui, rename={"Full_Bitscore": "BitScore"}, reference_probability_col="FUJISAN_Score")[0]
    base = explain_table(pd.DataFrame([REF]))[0]
    assert rec["prediction"]["probability"] == base["prediction"]["probability"]
    assert rec["prediction"]["reference_probability"]["abs_diff"] < 1e-9


def test_provenance_and_structure_placeholder():
    rec = explain_table(pd.DataFrame([REF]))[0]
    pv = rec["provenance"]
    assert rec["schema_version"] == SCHEMA_VERSION
    assert pv["model_sha256"] == RELEASED_MODEL_SHA256["updated"] and pv["model_sha256_matches_release"]
    assert pv["model_key"] == "updated" and pv["lightgbm_version"]
    assert rec["structure"]["status"] == "not_connected"
    assert rec["structure"]["residue_alignment"] is None and rec["structure"]["is_max_tm_pair"] is None
    assert rec["pair"]["pair_id"] == "P00509_Q22067" == rec["link"]["pair_key"]


def test_validate_detects_broken_sum():
    rec = explain_table(pd.DataFrame([REF]))[0]
    rec["shap"]["contributions"]["Dmn_TM"] += 0.01
    assert any("sum check" in e for e in validate_record(rec))


def test_cli_explain_roundtrip(tmp_path):
    inp = tmp_path / "predictions.tsv"
    pd.DataFrame([dict(REF, QueryID="P00509", TargetID="Q22067")]).drop(columns=["ProteinID1", "ProteinID2"]) \
        .to_csv(inp, sep="\t", index=False)
    out, tsv = tmp_path / "explain.jsonl", tmp_path / "explain.tsv"
    assert main(["explain", "-i", str(inp), "-o", str(out), "--tsv", str(tsv)]) == 0
    recs = read_jsonl(out)
    assert len(recs) == 1 and validate_record(recs[0]) == []
    json.dumps(recs[0], allow_nan=False)
    flat = pd.read_csv(tsv, sep="\t")
    assert flat.loc[0, "pair_id"] == "P00509_Q22067"
    assert "shap_BitScore_log_odds" in flat.columns
    write_jsonl(tmp_path / "copy.jsonl", recs)
    assert records_to_frame(read_jsonl(tmp_path / "copy.jsonl")).shape[0] == 1


def test_input_order_ids_and_nonfinite():
    rows = [REF, dict(REF, BitScore=float("inf")), dict(REF, BitScore="abc"), dict(REF, ProteinID1=" "),
            dict(REF, ProteinID1=None), REF]
    recs = explain_table(pd.DataFrame(rows))
    assert [r["status"] for r in recs] == ["ok", "skipped", "skipped", "skipped", "skipped", "ok"]
    assert [r["pair"]["source"]["row"] for r in recs] == [1, 2, 3, 4, 5, 6]          # input order, duplicates kept
    assert "non-finite feature: BitScore" in recs[1]["skip_reason"]
    assert recs[1]["features"]["values"]["BitScore"] is None
    assert recs[3]["skip_reason"] == "missing protein ID" and recs[3]["pair"]["pair_id"] is None
    for r in recs:
        assert validate_record(r) == []
        json.dumps(r, allow_nan=False)


def test_empty_input():
    assert explain_table(pd.DataFrame(columns=list(REF))) == []


def test_column_order_does_not_matter():
    df = pd.DataFrame([REF])
    a = explain_table(df)[0]["prediction"]["probability"]
    assert explain_table(df[list(df.columns)[::-1]])[0]["prediction"]["probability"] == a


@pytest.mark.parametrize("kwargs, df_change, msg", [
    ({}, lambda d: d.drop(columns="Dmn_TM"), "lacks model features"),
    ({}, lambda d: d.drop(columns=["ProteinID1", "ProteinID2"]), "no protein ID columns"),
    ({"id_cols": ["x", "y"]}, lambda d: d, "not found"),
    ({"rename": {"Full_Bitscore": "BitScore"}}, lambda d: d.assign(Full_Bitscore=1.0), "more than once"),
    ({"model_key": "paper2024"}, lambda d: d, "lacks model features"),
])
def test_invalid_inputs_raise(kwargs, df_change, msg):
    with pytest.raises(ValueError, match=msg):
        explain_table(df_change(pd.DataFrame([REF])), **kwargs)


def test_pair_orientation_is_kept():
    fwd = explain_table(pd.DataFrame([REF]))[0]
    rev = explain_table(pd.DataFrame([dict(REF, ProteinID1=REF["ProteinID2"], ProteinID2=REF["ProteinID1"])]))[0]
    assert fwd["pair"]["pair_id"] == "P00509_Q22067" and rev["pair"]["pair_id"] == "Q22067_P00509"
    assert rev["pair"]["query_id"] == "Q22067"


def test_paper2024_threshold_and_call():
    row = {"ProteinID1": "A", "ProteinID2": "B", "Full_log10(E_value)": -50.0, "Full_SeqID": 40.0, "Dmn_RMSD": 1.0,
           "Dmn_SeqID": 0.4, "Dmn_TM": 0.9, "Pckt_RMSD": 1.0, "Pckt_SeqID": 0.5}
    rec = explain_table(pd.DataFrame([row]), model_key="paper2024")[0]
    pr = rec["prediction"]
    assert pr["threshold"] == 0.5 and pr["call"] == int(pr["probability"] >= 0.5)
    assert rec["provenance"]["model_sha256"] == RELEASED_MODEL_SHA256["paper2024"]
    assert validate_record(rec) == []


def test_cli_keeps_text_ids_and_reports_bad_input(tmp_path):
    inp = tmp_path / "f.tsv"
    pd.DataFrame([dict(REF, ProteinID1="NA")]).to_csv(inp, sep="\t", index=False)
    out = tmp_path / "o.jsonl"
    assert main(["explain", "-i", str(inp), "-o", str(out)]) == 0
    assert read_jsonl(out)[0]["pair"]["pair_id"] == "NA_Q22067"
    pd.DataFrame([REF]).drop(columns="Dmn_TM").to_csv(inp, sep="\t", index=False)
    assert main(["explain", "-i", str(inp), "-o", str(tmp_path / "x.jsonl")]) == 2
    assert not (tmp_path / "x.jsonl").exists()
