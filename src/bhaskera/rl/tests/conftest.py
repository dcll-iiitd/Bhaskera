"""Shared fixtures: a tiny real Llama + word-level tokenizer, so tests run on CPU in seconds."""
from __future__ import annotations

import pytest

from .helpers import WORDS, make_lora_model


@pytest.fixture(scope="session")
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


@pytest.fixture()
def lora_model(tokenizer):
    return make_lora_model(tokenizer)
