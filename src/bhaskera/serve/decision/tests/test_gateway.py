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


def test_trace_decision_never_raises(monkeypatch):
    gateway._trace_decision("admin", BODY, 200, ANSWER, 0.01)  # no Langfuse configured
