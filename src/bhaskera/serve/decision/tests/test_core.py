import math

import pytest

from bhaskera.serve.decision import wire
from bhaskera.serve.decision.calibration import Calibration, fit_temperature
from bhaskera.serve.decision.decisions import decode, softmax
from bhaskera.serve.decision.prompts import compile_request
from bhaskera.serve.decision.schema import Request
from bhaskera.serve.decision.tests.fakes import CharTokenizer
from bhaskera.serve.decision.translate import (
    WireError,
    resolve_model,
    served_name,
    to_native,
    to_wire,
)


def _request(questions, state="Invoice 7 was paid twice."):
    return Request.model_validate({"state": state, "questions": questions})


def _text(tokens):
    return "".join(map(chr, tokens))


# --- decisions -----------------------------------------------------------------------------


def test_softmax_is_shift_invariant_and_tempered():
    assert softmax([0.0, 0.0]) == [0.5, 0.5]
    assert softmax([1.0, 3.0]) == pytest.approx(softmax([11.0, 13.0]))
    assert softmax([0.0, 2.0], temperature=2.0) == pytest.approx(softmax([0.0, 1.0]))


def test_decode_boolean_reports_noul_as_p_true():
    request = _request({"q": {"type": "boolean", "instructions": "Paid?"}})
    answer = decode(request.questions["q"], [0.0, math.log(3.0)])
    assert answer["noul"] == pytest.approx(0.75)
    assert answer["value"] is True


def test_decode_choice_and_score():
    request = _request(
        {
            "c": {
                "type": "choice",
                "instructions": "Which?",
                "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
            },
            "s": {"type": "score", "instructions": "How much?", "levels": ["low", "mid", "high"]},
        }
    )
    choice = decode(request.questions["c"], [2.0, 0.0])
    assert choice["choice"] == "a"
    assert 0 < choice["confidence"] <= 1
    score = decode(request.questions["s"], [0.0, 0.0, 0.0])
    assert score["score"] == pytest.approx(1.0)
    assert score["confidence"] == pytest.approx(0.0, abs=1e-9)


# --- calibration ---------------------------------------------------------------------------


def test_fit_temperature_recovers_a_sharpening():
    # Labels drawn as if the true logit were half the model's: T close to 2 fits best.
    rows = []
    for logit in (0.5, 1.0, 2.0, 4.0):
        ones = round(100 / (1 + math.exp(-logit / 2)))
        rows += [{"type": "boolean", "logits": [0.0, logit], "label_index": 1}] * ones
        rows += [{"type": "boolean", "logits": [0.0, logit], "label_index": 0}] * (100 - ones)
    calibration = fit_temperature(rows, fingerprint="f")
    assert calibration.temperatures["boolean"] == pytest.approx(2.0, rel=0.1)
    assert calibration.temperature_for("boolean", 2) == calibration.temperatures["boolean:2"]
    assert calibration.temperature_for("choice", 4) == 1.0


def test_calibration_file_round_trip(tmp_path):
    rows = [{"type": "boolean", "logits": [0.0, 1.0], "label_index": 1}] * 10
    calibration = fit_temperature(rows, fingerprint="f")
    path = tmp_path / "calibration.json"
    path.write_text(calibration.model_dump_json())
    assert Calibration.from_file(path) == calibration


# --- prompts -------------------------------------------------------------------------------


def test_binary_prompt_shape_and_slots():
    request = _request({"paid": {"type": "boolean", "instructions": "Was it paid?"}})
    prefix, jobs = compile_request(CharTokenizer(), request, ctx=8192, binary=True)
    assert _text(jobs[0].tokens) == (
        "\x02<evidence>\nInvoice 7 was paid twice.\n</evidence>\n\nQuestion: Was it paid?\nAnswer:"
    )
    assert jobs[0].slots == [ord("0"), ord("1")]
    assert jobs[0].tokens[: len(prefix)] == prefix


def test_questions_share_one_state_prefix():
    request = _request(
        {
            "a": {"type": "boolean", "instructions": "First?"},
            "b": {"type": "boolean", "instructions": "Second question?"},
        }
    )
    prefix, jobs = compile_request(CharTokenizer(), request, ctx=8192, binary=True)
    assert len(jobs) == 2
    for job in jobs:
        assert job.tokens[: len(prefix)] == prefix
    # The last token of the state is left out of the prefix and re-read by every branch.
    assert _text(prefix).endswith("</evidence")


def test_chat_prompt_uses_letters_for_options():
    request = _request(
        {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "options": [
                    {"id": "billing", "description": "Payments"},
                    {"id": "tech", "description": "Bugs"},
                ],
            }
        }
    )
    _, jobs = compile_request(CharTokenizer(), request, ctx=8192)
    text = _text(jobs[0].tokens)
    assert "A. Payments\nB. Bugs" in text
    assert text.endswith("<assistant>")
    assert jobs[0].slots == [ord("A"), ord("B")]


def test_context_limit_is_enforced_without_truncation():
    request = _request({"q": {"type": "boolean", "instructions": "Q?"}}, state="x" * 200)
    with pytest.raises(ValueError, match="exceeds the context limit"):
        compile_request(CharTokenizer(), request, ctx=100, binary=True)


# --- wire / translate ----------------------------------------------------------------------


def _wire(questions, model="jev-latest"):
    return wire.SystemOneRequest.model_validate(
        {"model": model, "state": "I was charged twice.", "questions": questions}
    )


def test_noul_round_trip_and_usage():
    body = _wire(
        {
            "billing": {"type": "noul", "instructions": "Is this a billing problem?"},
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        }
    )
    native, keys = to_native(body)
    assert native.questions["billing"].type == "boolean"
    assert keys["billing"] == ["false", "true"]
    result = {
        "prefix_tokens": 10,
        "answers": {
            "billing": {"probabilities": {"false": 0.1, "true": 0.9}, "input_tokens": 20},
            "angry": {"probabilities": {"false": 0.6, "true": 0.4}, "input_tokens": 15},
        },
    }
    out = to_wire(body, result, keys, "jevos-q8_0")
    assert out["answers"]["billing"] == {"type": "noul", "noul": 0.9}
    assert out["usage"] == {"input_tokens": 25, "output_tokens": 0}
    assert out["model"] == "jevos-q8_0"


def test_choice_options_get_positional_ids():
    body = _wire({"team": {"type": "choice", "criteria": {"billing": "refunds", "tech": None}}})
    native, keys = to_native(body)
    options = native.questions["team"].options
    assert [o.id for o in options] == ["o0", "o1"]
    assert options[0].description == "billing: refunds"
    assert keys["team"] == ["billing", "tech"]


def test_model_names():
    assert resolve_model("jev-latest", "jevos-q8_0") == "jevos-q8_0"
    assert resolve_model("jevos-q8_0", "jevos-q8_0") == "jevos-q8_0"
    with pytest.raises(WireError):
        resolve_model("gpt-4", "jevos-q8_0")
    assert served_name({"model": "jevos", "precision": "q8_0"}) == "jevos-q8_0"
    assert served_name({"source_files": {"My-Model.gguf": "x"}}) == "my-model"
