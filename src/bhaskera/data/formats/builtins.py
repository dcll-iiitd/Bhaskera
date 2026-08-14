"""
bhaskera.data.formats.builtins
==============================
Built-in renderers for the most common SFT data layouts. All conversational
renderers return a standard `List[dict]` of messages, allowing `tokenize.py`
to correctly align natively-supported assistant masks across HF tokenizers.
"""
from __future__ import annotations

from typing import Any, List

from . import register_format


def _to_list(x: Any) -> List:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)

def _to_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    try:
        return dict(x)
    except Exception:
        return {"role": "user", "content": str(x)}

def _manual_chatml(messages: List[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)

def _manual_chatml_tokenize(tokenizer: Any, messages: List[dict]) -> tuple[list[int], list[int]]:
    """Fallback manual ChatML tokenisation with exact sequence alignment/labels."""
    input_ids = []
    labels = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        role_str = f"<|im_start|>{role}\n"
        role_ids = tokenizer.encode(role_str, add_special_tokens=False)

        content_str = f"{content}<|im_end|>\n"
        content_ids = tokenizer.encode(content_str, add_special_tokens=False)

        input_ids.extend(role_ids + content_ids)
        if role == "assistant":
            labels.extend([-100] * len(role_ids) + content_ids)
        else:
            labels.extend([-100] * (len(role_ids) + len(content_ids)))

    return input_ids, labels

def _apply_chat_template_safe(tokenizer: Any, messages: List[dict]) -> str:
    has_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    )
    if not has_template:
        return _manual_chatml(messages)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


@register_format("chatml")
def render_chatml(row: dict, tokenizer: Any, options: dict) -> list[dict]:
    messages_field = options.get("messages_field", "messages")
    messages = _to_list(row.get(messages_field, []))
    return [_to_dict(m) for m in messages]


@register_format("alpaca")
def render_alpaca(row: dict, tokenizer: Any, options: dict) -> list[dict]:
    instruction = str(row.get("instruction", "") or "")
    inp         = str(row.get("input", "") or "")
    output      = str(row.get("output", "") or "")

    user_turn = instruction if not inp else f"{instruction}\n\n{inp}"

    # Render Alpaca directly to messages so the trainer masks instruction tokens natively.
    return [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": output},
    ]


_SHAREGPT_ROLE_MAP = {
    "human":     "user",
    "user":      "user",
    "gpt":       "assistant",
    "assistant": "assistant",
    "chatgpt":   "assistant",
    "bard":      "assistant",
    "system":    "system",
    "tool":      "tool",
    "function":  "tool",
}

@register_format("sharegpt")
def render_sharegpt(row: dict, tokenizer: Any, options: dict) -> list[dict]:
    field    = options.get("conversations_field", "conversations")
    role_map = {**_SHAREGPT_ROLE_MAP, **(options.get("role_map") or {})}

    convs = _to_list(row.get(field, []))
    messages = []
    for c in convs:
        c = _to_dict(c)
        sender = str(c.get("from", "user")).lower()
        role = role_map.get(sender, "user")
        messages.append({"role": role, "content": c.get("value", "") or ""})

    return messages
