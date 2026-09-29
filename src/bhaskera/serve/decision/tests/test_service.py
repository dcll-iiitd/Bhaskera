import pytest

from bhaskera.serve.decision import service, wire
from bhaskera.serve.decision.schema import Request
from bhaskera.serve.decision.translate import WireError


def _body(model="jev-latest"):
    return wire.SystemOneRequest.model_validate(
        {
            "model": model,
            "state": "I was charged twice.",
            "questions": {"billing": {"type": "noul", "instructions": "Billing problem?"}},
        }
    )


def _result(native):
    return {
        "prefix_tokens": 4,
        "answers": {
            qid: {"probabilities": {"false": 0.3, "true": 0.7}, "input_tokens": 10}
            for qid in native.questions
        },
        "timing": {"inference_seconds": 0.002, "total_seconds": 0.003},
    }


def test_prepare_then_finish():
    body = _body()
    model, native, keys = service.prepare(body, "jevos-q8_0")
    assert model == "jevos-q8_0" and isinstance(native, Request)
    payload, headers = service.finish(body, _result(native), keys, model)
    assert payload["answers"]["billing"] == {"type": "noul", "noul": 0.7}
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 0}
    assert headers == {"Server-Timing": "inference;dur=2.0, total;dur=3.0"}
    wire.SystemOneResponse.model_validate(payload)


def test_prepare_rejects_an_unknown_model():
    with pytest.raises(WireError):
        service.prepare(_body("gpt-4"), "jevos-q8_0")


def test_error_detail_shapes():
    try:
        Request.model_validate({"state": "", "questions": {}})
    except ValueError as error:
        detail = service.error_detail(error)
    assert detail and detail[0]["loc"][0] == "body" and "msg" in detail[0]
    wire_detail = service.error_detail(WireError(["body", "model"], "Unknown model"))
    assert wire_detail == [{"loc": ["body", "model"], "msg": "Unknown model", "type": "value_error"}]


def test_models_and_health_payloads():
    models = service.models_payload("jevos-q8_0", "2026-09-29")
    wire.ModelMetadataList.model_validate(models)
    assert [m["name"] for m in models["models"]] == ["jevos-q8_0", "jev-latest"]
    health = service.health_payload("jevos-q8_0", {"fingerprint": "fp"})
    assert health == {"status": "ready", "model": "jevos-q8_0", "engine": {"fingerprint": "fp"}}
