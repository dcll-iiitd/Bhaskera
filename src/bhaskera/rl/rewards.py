"""
bhaskera.rl.rewards
===================
Verifiable rewards: a reward is a plain function ``(completion, answer) -> float``
that checks the model's output against a known answer. No learned reward model.

Register your own with the decorator and select it with ``rl.reward`` in the YAML:

    @register_reward("my_task")
    def my_task(completion: str, answer: str) -> float: ...

Rewards must be deterministic and must not execute model output. Anything that
runs generated code needs a sandbox and does not belong here.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Callable, Dict, Optional

RewardFn = Callable[[str, str], float]
REWARD_REGISTRY: Dict[str, RewardFn] = {}


def register_reward(name: str) -> Callable[[RewardFn], RewardFn]:
    def wrapper(fn: RewardFn) -> RewardFn:
        REWARD_REGISTRY[name] = fn
        return fn
    return wrapper


def get_reward(name: str) -> RewardFn:
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward '{name}'. Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]


# ---------------------------------------------------------------------------
# Answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST ``\\boxed{...}`` in ``text`` (brace-balanced)."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth, begin = 1, i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i]
        i += 1
    return None  # unbalanced braces: treat as no answer


_FRAC = re.compile(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")
_TEXT = re.compile(r"\\(?:text|mathrm|textbf)\{[^{}]*\}")


def to_number(s: str) -> Optional[Fraction]:
    """Parse '1,234', '$12.50', '75%', '\\frac{1}{2}', '3/4' into an exact Fraction."""
    s = s.strip()
    s = _TEXT.sub("", s)                      # drop units such as \text{ dollars}
    s = _FRAC.sub(r"\1/\2", s)
    s = s.replace("\\%", "").replace("%", "").replace("\\$", "").replace("$", "")
    s = s.replace("\\,", "").replace(",", "").replace(" ", "").rstrip(".")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def gold_answer(answer: str) -> str:
    """GSM8K stores the solution followed by '#### <number>'; plain answers pass through."""
    return answer.split("####")[-1].strip() if "####" in answer else answer.strip()


# ---------------------------------------------------------------------------
# Built-in reward
# ---------------------------------------------------------------------------

@register_reward("math_boxed")
def math_boxed(completion: str, answer: str) -> float:
    """1.0 if the final ``\\boxed{}`` answer equals the gold answer numerically, else 0.0.

    Strict on purpose: no ``\\boxed{}`` means no reward, which also teaches the format.
    """
    boxed = extract_boxed(completion)
    if boxed is None:
        return 0.0
    predicted, gold = to_number(boxed), to_number(gold_answer(answer))
    if predicted is None or gold is None:
        return 1.0 if boxed.strip() == gold_answer(answer) else 0.0
    return 1.0 if predicted == gold else 0.0
