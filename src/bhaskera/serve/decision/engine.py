# Ported from feder-cr/jev@5be02a3 (MIT); see NOTICE in this directory.
"""Thread-safe long-lived model service with deterministic postprocessing."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .decisions import candidates, decode
from .prompts import BINARY_PROMPT_VERSION, compile_request
from .schema import Request


class Engine:
    def __init__(self, backend, ctx=8192, calibration=None, binary=False):
        if ctx < 1:
            raise ValueError("ctx must be positive")
        self.backend = backend
        self.ctx = ctx
        self.calibration = calibration
        # binary=True: a model trained on yes/no questions only (0/1 answers); nothing else is accepted.
        # The prompt version comes from the GGUF (`jev.prompt_version`), else the default.
        self.binary = binary
        self.binary_version = backend.metadata.get("binary_prompt_version") or BINARY_PROMPT_VERSION
        if calibration and calibration.fingerprint != backend.metadata["fingerprint"]:
            raise ValueError(
                "Calibration was fitted for a different model/runtime/prompt configuration"
            )
        self._lock = Lock()
        # All model work runs on one thread that lives as long as the process: some compute
        # backends abort when a thread that ran computations exits.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-inference")

    def decide(self, request: Request | dict) -> dict:
        # Re-validate a serialized snapshot, also protecting mutable Pydantic objects.
        request = Request.model_validate(
            request.model_dump() if isinstance(request, Request) else request
        )
        started = time.perf_counter()
        # A binary model answers yes/no questions only, on the binary prompt (0/1 tokens):
        # a request with any other question type is refused before the lock is taken.
        self._check_binary(request)
        with self._lock:
            acquired = time.perf_counter()
            prefix, jobs = self._compile(request)
            encoded = time.perf_counter()
            logits, timing = self._worker.submit(
                self.backend.score, prefix, jobs, request.mode
            ).result()
            answers = self._answers(request, jobs, logits)
        return self._response(
            request, prefix, answers, request.mode, timing, started, acquired, encoded
        )

    def decide_many(self, requests: list) -> list:
        """Several requests scored together in shared llama_decode calls (seq-copy only).

        Returns one entry per request, in order: the dict `decide` returns, or the ValueError
        that request alone raised; a bad request never fails its neighbours. Every request
        shares its state prefix, whatever its `mode`. Requests are scored in chunks of at most
        the backend's `max_requests`; a scoring error fails only its chunk.
        """
        started = time.perf_counter()
        outcomes: list = [None] * len(requests)
        compiled = []
        with self._lock:
            acquired = time.perf_counter()
            for index, raw in enumerate(requests):
                try:
                    request = Request.model_validate(
                        raw.model_dump() if isinstance(raw, Request) else raw
                    )
                    self._check_binary(request)
                    prefix, jobs = self._compile(request)
                except ValueError as error:
                    outcomes[index] = error
                    continue
                compiled.append((index, request, prefix, jobs))
            encoded = time.perf_counter()
            size = max(1, getattr(self.backend, "max_requests", 1))
            for offset in range(0, len(compiled), size):
                chunk = compiled[offset : offset + size]
                try:
                    scored, timing = self._worker.submit(
                        self.backend.score_many, [(prefix, jobs) for _, _, prefix, jobs in chunk]
                    ).result()
                except ValueError as error:
                    for index, *_ in chunk:
                        outcomes[index] = error
                    continue
                for (index, request, prefix, jobs), logits in zip(chunk, scored, strict=True):
                    outcomes[index] = self._response(
                        request,
                        prefix,
                        self._answers(request, jobs, logits),
                        "shared",
                        {**timing, "batch_requests": len(chunk)},
                        started,
                        acquired,
                        encoded,
                    )
        return outcomes

    def _check_binary(self, request: Request) -> None:
        if not self.binary:
            return
        others = [qid for qid, q in request.questions.items() if q.type != "boolean"]
        if others:
            raise ValueError(
                f"Binary model: only yes/no (noul) questions are answered; not {', '.join(others)}"
            )

    def _compile(self, request: Request):
        if self.binary:
            return compile_request(
                self.backend.tokenizer, request, self.ctx, binary=self.binary_version
            )
        return compile_request(self.backend.tokenizer, request, self.ctx)

    def _response(self, request, prefix, answers, mode, timing, started, acquired, encoded) -> dict:
        model = self.backend.metadata
        if self.binary:
            model = model | {"prompt_version": self.binary_version}
        return {
            "model": model,
            "mode": mode,
            # Tokens of the state that every question starts with: counted once in `usage`.
            "prefix_tokens": len(prefix),
            "answers": answers,
            "calibration": self.calibration.model_dump() if self.calibration else None,
            "timing": {
                **timing,
                "queue_seconds": acquired - started,
                "compile_seconds": encoded - acquired,
                "total_seconds": time.perf_counter() - started,
            },
        }

    def _answers(self, request: Request, jobs, logits) -> dict:
        answers = {}
        for job in jobs:
            question = request.questions[job.id]
            kind = "boolean" if self.binary else question.type
            temperature = (
                self.calibration.temperature_for(kind, len(candidates(question)))
                if self.calibration
                else 1.0
            )
            answers[job.id] = decode(question, logits[job.id], temperature)
            answers[job.id]["prompt_sha256"] = job.prompt_sha256
            answers[job.id]["input_tokens"] = len(job.tokens)
        return answers
