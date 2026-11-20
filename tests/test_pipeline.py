"""Offline end-to-end test: fake data + mock model. Run with: python -m pytest -q"""
import json

from triage import config as C
from triage import data, pipeline, report
from tests.make_fixture import write


def test_end_to_end_mock(tmp_path, monkeypatch):
    for name, sub in [("DATA_DIR", "data"), ("RAW_DIR", "data/raw"),
                      ("RESULTS_DIR", "results"), ("REVIEW_DIR", "review")]:
        monkeypatch.setattr(C, name, tmp_path / sub)
    monkeypatch.setattr(C, "PER_PRODUCT", 20)
    monkeypatch.setattr(C, "MIN_PRODUCT_SUPPORT", 20)
    C.RAW_DIR.mkdir(parents=True)
    write(C.RAW_DIR / C.EXPORTS[C.DEFAULT_EXPORT])

    data.prepare()
    stats = json.loads((C.DATA_DIR / "prep_stats.json").read_text())
    assert stats["duplicate_template_narratives_removed"] > 0
    assert stats["sample_size"] == 80

    models = ["haiku", "opus"]
    pipeline.triage(models, mode="mock")
    pipeline.latency(models, n=5, mock=True)
    pipeline.run_judge(models, n=10, mode="mock")
    pipeline.review(models)
    import pandas as pd
    rl = pd.read_csv(C.REVIEW_DIR / "risk_labels.csv")
    assert len(rl) == 80 and set(rl.columns) == {"cid", "company", "narrative", "risk", "notes"}
    rl["risk"] = ["high" if i % 3 == 0 else "low" for i in range(len(rl))]
    rl.to_csv(C.REVIEW_DIR / "risk_labels.csv", index=False)
    report.build_report(models)
    summary = json.loads((C.RESULTS_DIR / "summary.json").read_text())
    assert summary["risk_vs_human"]["n"] == 80
    assert set(summary["risk_vs_human"]["models"]) == set(models)
    assert {m["model"] for m in summary["models"]} == set(models)
    assert all(0 <= m["product_acc"] <= 1 for m in summary["models"])


def test_schema_matches_taxonomy():
    from triage.agent import build_schema
    tax = {"A": ["x", "y"], "B": ["y", "z"]}
    s = build_schema(tax)
    assert s["properties"]["product"]["enum"] == ["A", "B"]
    assert s["properties"]["issue"]["enum"] == ["x", "y", "z"]
    assert s["additionalProperties"] is False


def test_report_scores_only_attempted(tmp_path, monkeypatch):
    for name, sub in [("DATA_DIR", "data"), ("RAW_DIR", "data/raw"),
                      ("RESULTS_DIR", "results"), ("REVIEW_DIR", "review")]:
        monkeypatch.setattr(C, name, tmp_path / sub)
    monkeypatch.setattr(C, "PER_PRODUCT", 20)
    monkeypatch.setattr(C, "MIN_PRODUCT_SUPPORT", 20)
    C.RAW_DIR.mkdir(parents=True)
    write(C.RAW_DIR / C.EXPORTS[C.DEFAULT_EXPORT])
    data.prepare()
    pipeline.risk_labels()  # must work before any model has run
    assert (C.REVIEW_DIR / "risk_labels.csv").exists()
    pipeline.triage(["haiku"], mode="mock", limit=10)
    report.build_report(["haiku"])
    m = json.loads((C.RESULTS_DIR / "summary.json").read_text())["models"][0]
    assert m["n"] == 10 and m["errors"] == 0
