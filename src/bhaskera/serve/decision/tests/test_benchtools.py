import math

import pytest

from bhaskera.serve.decision import benchtools as bt
from bhaskera.serve.decision import wire


def test_parity_set_is_deterministic_and_valid():
    first, second = bt.parity_requests(), bt.parity_requests()
    assert first == second and len(first) == 300
    counts = set()
    for item in first:
        wire.SystemOneRequest.model_validate(item["request"])
        counts.add(len(item["request"]["questions"]))
    assert counts == {1, 2, 3, 5, 10}
    assert len({item["id"] for item in first}) == 300
    assert any(isinstance(i["request"]["state"], dict) for i in first)


def test_bench_sets_differ_by_length():
    short = bt.bench_requests("short", 3, n=5)
    document = bt.bench_requests("document", 3, n=5)
    assert all(len(i["request"]["questions"]) == 3 for i in short + document)
    assert min(len(i["request"]["state"]) for i in document) > 5000
    assert max(len(str(i["request"]["state"])) for i in short) < 400


def _rec(rid, **noul):
    answers = {q: {"type": "noul", "noul": p} for q, p in noul.items()}
    return {"id": rid, "status": 200, "response": {"answers": answers}}


def test_compare_passes_within_tolerance_and_reports_the_worst():
    reference = [_rec("a", q0=0.90, q1=0.20), _rec("b", q0=0.55)]
    candidate = [_rec("a", q0=0.905, q1=0.20), _rec("b", q0=0.548)]
    report = bt.compare(reference, candidate)
    assert report["passed"] and report["answers"] == 3
    assert report["max_delta"] == pytest.approx(0.005)
    assert report["worst"][0]["id"] == "a"


def test_compare_fails_on_drift_flip_or_missing():
    report = bt.compare([_rec("a", q0=0.52)], [_rec("a", q0=0.48)])
    assert not report["passed"] and report["decisions_flipped"] == 1
    assert not bt.compare([_rec("a", q0=0.5)], [])["passed"]
    bad = {"id": "a", "status": 422, "response": {}}
    assert bt.compare([_rec("a", q0=0.5)], [bad])["status_mismatch"] == ["a"]


def test_percentile_and_summary():
    values = [0.001 * i for i in range(1, 101)]
    assert bt.percentile(values, 50) == pytest.approx(0.050)
    assert bt.percentile(values, 90) == pytest.approx(0.090)
    assert math.isnan(bt.percentile([], 50))
    summary = bt.summarize(values, errors=2, questions=300, seconds=10.0)
    assert summary["requests"] == 100 and summary["errors"] == 2
    assert summary["req_per_s"] == pytest.approx(10.0)
    assert summary["questions_per_s"] == pytest.approx(30.0)
    assert summary["p50_ms"] == pytest.approx(50.0)


def test_table_renders_markdown():
    rows = [{"label": "x", "requests": "bench_short_q1", "concurrency": 4, "completed": 180, "req_per_s": 12.3,
             "questions_per_s": 12.3, "p50_ms": 5.0, "p90_ms": 7.0, "p99_ms": 9.0, "errors": 0,
             "max_memory_mib": 1024.0, "mean_util_pct": 55.0}]
    text = bt.table(rows)
    assert text.splitlines()[0].startswith("| label |")
    assert "| x | bench_short_q1 | 4 | 180 | 12.3 |" in text
