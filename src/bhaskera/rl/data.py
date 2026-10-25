"""
bhaskera.rl.data
================
Prompt datasets for RL. The file format is JSONL, one object per line:

    {"prompt": "What is 2+3?", "answer": "5"}

``answer`` is whatever the chosen reward function compares against; for GSM8K it
may be the full solution ending in ``#### 5``.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Example:
    prompt: str
    answer: str


def load_examples(path: str, limit: Optional[int] = None) -> List[Example]:
    examples: List[Example] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                examples.append(Example(prompt=str(row["prompt"]), answer=str(row["answer"])))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{path}:{lineno}: expected JSON with 'prompt' and 'answer' ({e})")
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


def format_prompt(tokenizer, text: str, system_prompt: str = "") -> str:
    """Wrap a question for the model: chat template if the tokenizer has one, else plain text."""
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n{text}\n" if system_prompt else f"{text}\n"


class PromptSampler:
    """Endless, seeded stream of examples: reshuffled every pass, reproducible from ``position``.

    ``position`` counts examples handed out so far. Starting at a non-zero position
    resumes the same stream, which is how successive federated rounds see fresh
    prompts instead of replaying the same ones.
    """

    def __init__(self, examples: List[Example], seed: int, position: int = 0):
        if not examples:
            raise ValueError("PromptSampler needs at least one example")
        self.examples = examples
        self.seed = seed
        self.position = position

    @property
    def epoch(self) -> int:
        """Pass number (from 0) that the most recently returned example belongs to."""
        return max(self.position - 1, 0) // len(self.examples)

    def _order(self, epoch: int) -> List[int]:
        order = list(range(len(self.examples)))
        random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
        return order

    def next_batch(self, n: int) -> List[Example]:
        batch = []
        for _ in range(n):
            epoch, offset = divmod(self.position, len(self.examples))
            batch.append(self.examples[self._order(epoch)[offset]])
            self.position += 1
        return batch
