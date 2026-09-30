"""Test doubles for the decision engine: no llama.cpp, no weights."""

from __future__ import annotations


class CharTokenizer:
    """One token per character (its code point). Every string is exactly reversible, so the
    prefix and answer-slot checks in `compile_request` behave as with a real tokenizer."""

    special_tokens = {"bos_token": "\x02"}
    answer_prefix = ""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


class FakeBackend:
    """Scores from a table of logits keyed by question id; records every call."""

    def __init__(self, logits_by_question: dict[str, list[float]]):
        self.tokenizer = CharTokenizer()
        self.metadata = {
            "fingerprint": "fp",
            "binary_prompt_version": "binary",
            "model": "jevos",
            "precision": "q8_0",
        }
        self.logits = logits_by_question
        self.calls: list = []
        self.many_calls: list[int] = []
        self.max_requests = 1

    def score(self, prefix, jobs, mode="shared"):
        self.calls.append((prefix, [job.id for job in jobs], mode))
        return {job.id: self.logits[job.id] for job in jobs}, {"inference_seconds": 0.001}

    def score_many(self, requests):
        self.many_calls.append(len(requests))
        results = [{job.id: self.logits[job.id] for job in jobs} for _, jobs in requests]
        return results, {"inference_seconds": 0.001, "batch_requests": len(requests)}
