import json

import httpx
import pytest
from fastapi.testclient import TestClient

gateway = pytest.importorskip("bhaskera.gateway")

BODY = {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i?"}}}
ANSWER = {"model": "jevos-q8_0", "answers": {"q": {"type": "noul", "noul": 0.8}},
          "usage": {"input_tokens": 5, "output_tokens": 0}}


def _mock(monkeypatch, status=200, payload=ANSWER):
    seen, traced = {}, []

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(status, json=payload, headers={"Server-Timing": "inference;dur=3.0"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ray")
    monkeypatch.setattr(gateway, "decision_client", client)
    monkeypatch.setattr(gateway, "_trace_decision", lambda *args: traced.append(args))
    return seen, traced


def test_systemone_forwards_and_traces(monkeypatch):
    seen, traced = _mock(monkeypatch)
    response = TestClient(gateway.app).post(
        "/v1/systemone", json=BODY, headers={"Authorization": "Bearer sk-bhaskera-admin"}
    )
    assert response.status_code == 200
    assert response.json() == ANSWER
    assert response.headers["server-timing"] == "inference;dur=3.0"
    assert seen == {"path": "/v1/systemone", "body": BODY}
    user_id, body, status, payload, seconds = traced[0]
    assert (user_id, body, status, payload) == ("admin", BODY, 200, ANSWER) and seconds >= 0


def test_systemone_passes_422_through(monkeypatch):
    detail = {"detail": [{"loc": ["body"], "msg": "bad", "type": "value_error"}]}
    _mock(monkeypatch, status=422, payload=detail)
    response = TestClient(gateway.app).post(
        "/v1/systemone", json=BODY, headers={"Authorization": "Bearer sk-bhaskera-alice"}
    )
    assert response.status_code == 422 and response.json() == detail


def test_systemone_rejects_an_unknown_key(monkeypatch):
    _mock(monkeypatch)
    response = TestClient(gateway.app).post(
        "/v1/systemone", json=BODY, headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 401


def test_trace_decision_never_raises():
    gateway._trace_decision("admin", BODY, 200, ANSWER, 0.01)  # no Langfuse configured


class _FakeSpan:
    def __init__(self, calls, kwargs):
        self.calls, self.kwargs = calls, kwargs

    def __enter__(self):
        self.calls.append(("enter", self.kwargs))
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))


def test_trace_decision_emits_parent_and_child_spans(monkeypatch):
    import langfuse

    calls, propagated = [], []

    class FakeClient:
        def start_as_current_observation(self, **kwargs):
            return _FakeSpan(calls, kwargs)

    class Propagate:
        def __init__(self, **kwargs):
            propagated.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(langfuse, "get_client", lambda: FakeClient())
    monkeypatch.setattr(langfuse, "propagate_attributes", Propagate)
    gateway._trace_decision("admin", BODY, 200, ANSWER, 0.01)
    entered = [kw for kind, kw in calls if kind == "enter"]
    assert [kw["name"] for kw in entered] == ["systemone", "question:q"]
    assert entered[1]["input"] == BODY["questions"]["q"]
    assert entered[1]["output"] == ANSWER["answers"]["q"]
    assert propagated == [{"user_id": "admin", "tags": ["decision"]}]
    final = [kw for kind, kw in calls if kind == "update"][0]
    assert final["output"]["status"] == 200 and final["output"]["usage"] == ANSWER["usage"]


def test_systemone_maps_connection_error_to_502(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ray")
    monkeypatch.setattr(gateway, "decision_client", client)
    response = TestClient(gateway.app).post(
        "/v1/systemone", json=BODY, headers={"Authorization": "Bearer sk-bhaskera-admin"}
    )
    assert response.status_code == 502
    assert response.json()["detail"].startswith("decision backend unavailable")


def test_systemone_maps_non_json_upstream_to_502(monkeypatch):
    def handler(request):
        return httpx.Response(502, text="<html>bad gateway</html>", headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ray")
    monkeypatch.setattr(gateway, "decision_client", client)
    response = TestClient(gateway.app).post(
        "/v1/systemone", json=BODY, headers={"Authorization": "Bearer sk-bhaskera-admin"}
    )
    assert response.status_code == 502
    assert response.json()["detail"].startswith("decision backend unavailable")
