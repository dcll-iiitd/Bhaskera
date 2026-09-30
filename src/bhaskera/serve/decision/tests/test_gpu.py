"""GPU tests: a CUDA llama.cpp runtime and downloaded weights are required. On the GPU host:

    source $BJ/env.sh && .venv/bin/pytest -m gpu src/bhaskera/serve/decision/tests -v
"""
import os

import pytest

from bhaskera.serve.decision.backend import LlamaBackend
from bhaskera.serve.decision.engine import Engine

pytestmark = pytest.mark.gpu

README_REQUEST = {
    "state": {
        "item": "wireless mouse",
        "delivered": "5 days ago",
        "customer_message": "The box arrived empty. This is the second time!",
    },
    "questions": {
        "refund": {
            "type": "boolean",
            "instructions": "Our policy refunds items reported missing within 30 days of "
            "delivery. Should this customer get a refund?",
        },
        "upset": {"type": "boolean", "instructions": "Is the customer upset?"},
        "wrong_item": {
            "type": "boolean",
            "instructions": "Does the customer say they received the wrong item?",
        },
    },
}


def load(env: str, **kwargs) -> LlamaBackend:
    gguf, runtime = os.environ.get(env), os.environ.get("BHASKERA_LLAMA_DIR")
    if not (gguf and runtime):
        pytest.skip(f"set {env} and BHASKERA_LLAMA_DIR")
    return LlamaBackend.load(gguf, device="cuda", runtime_dir=runtime, **kwargs)


@pytest.fixture(scope="module")
def jevos():
    backend = load("BHASKERA_JEVOS_GGUF")
    yield Engine(backend, binary=True)
    backend.session.close()


def test_jevos_loads_on_the_gpu(jevos):
    meta = jevos.backend.metadata
    assert meta["device"] == "gpu"
    assert meta["architecture"] == "llama"
    assert meta["binary_prompt_version"] == "binary"


def test_branch_strategy_follows_the_probe(jevos):
    meta = jevos.backend.metadata
    print("branch:", meta["branch_strategy"], "probe delta:", meta["probe_branch_max_delta"])
    if meta["branch_strategy"] == "seq-copy":
        assert meta["probe_branch_max_delta"] <= 0.05
    else:
        assert meta["probe_branch_max_delta"] > 0.05


def test_jevos_answers_the_readme_example(jevos):
    noul = {k: v["noul"] for k, v in jevos.decide(README_REQUEST)["answers"].items()}
    print(noul)
    assert noul["refund"] > 0.5 and noul["upset"] > 0.5 and noul["wrong_item"] < 0.5


def test_shared_prefix_matches_direct_scoring(jevos):
    shared = jevos.decide({**README_REQUEST, "mode": "shared"})["answers"]
    direct = jevos.decide({**README_REQUEST, "mode": "direct"})["answers"]
    for qid in shared:
        assert abs(shared[qid]["noul"] - direct[qid]["noul"]) <= 0.01


TICKET = {
    "state": {
        "ticket": "Hi, we were billed twice for March. Refund the duplicate today or we cancel.",
        "plan": "Business, monthly, 12 seats",
    },
    "questions": {
        "refund": {"type": "boolean", "instructions": "Does the customer ask for a refund?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "options": [
                {"id": "billing", "description": "Payments, invoicing, refunds"},
                {"id": "technical", "description": "Bugs, outages"},
                {"id": "sales", "description": "Pricing, new contracts"},
            ],
        },
        "anger": {
            "type": "score",
            "instructions": "How angry is the customer?",
            "levels": ["Calm", "Frustrated but civil", "Very angry or threatening"],
        },
    },
}


@pytest.fixture(scope="module")
def gemma():
    backend = load("BHASKERA_GEMMA_GGUF")
    yield Engine(backend)
    backend.session.close()


def test_gemma_uses_the_chat_template_path(gemma):
    meta = gemma.backend.metadata
    print("gemma:", meta["architecture"], meta["branch_strategy"], meta["probe_letter_mass"])
    assert meta["device"] == "gpu"
    assert not meta["binary_prompt_version"]
    assert meta["probe_letter_mass"] >= 0.5


def test_gemma_answers_choice_score_and_boolean(gemma):
    answers = gemma.decide(TICKET)["answers"]
    print({k: {x: v[x] for x in ("probabilities",)} for k, v in answers.items()})
    assert answers["refund"]["value"] is True
    assert answers["team"]["choice"] == "billing"
    assert answers["anger"]["score"] >= 0.5
