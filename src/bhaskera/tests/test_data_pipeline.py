# tests/test_data_pipeline.py
import pytest
import numpy as np
from unittest.mock import patch
from bhaskera.data.tokenize import TokenizerActor, _cache_version_hash

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 99
        self.eos_token_id = 100
        self.chat_template = "mock"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token_id = None
        self.is_fast = False

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            return [98] + [ord(c) for c in text] + [self.eos_token_id]
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, return_dict=False, return_assistant_tokens_mask=False, **kwargs):
        if not tokenize:
            return "".join(m["role"] + m["content"] for m in messages)

        ids = []
        mask = []
        for m in messages:
            content = m["content"]
            c_ids = [ord(c) for c in content]
            ids.extend(c_ids + [self.eos_token_id])
            if m["role"] == "assistant":
                mask.extend([1] * len(c_ids) + [1])
            else:
                mask.extend([0] * len(c_ids) + [0])

        if return_dict and return_assistant_tokens_mask:
            return {"input_ids": ids, "assistant_masks": mask}

        return ids

@patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer())
def test_sft_multi_turn_padding(mock_load):
    actor = TokenizerActor("mock-model", seq_len=16, format_name="chatml")
    batch = {"messages": [[
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]]}

    # Expected content (ascii): 'hi' -> [104, 105], 'u' -> [117], 'a' -> [97]
    # Total with EOS: [104, 105, 100, 117, 100, 97, 100] = 7 tokens. Pad to 16.
    out = actor(batch)

    assert len(out["input_ids"][0]) == 16
    assert out["labels"][0][0] == -100 # system 'h'
    assert out["labels"][0][1] == -100 # system 'i'
    assert out["labels"][0][2] == -100 # system eos
    assert out["labels"][0][3] == -100 # user 'u'
    assert out["labels"][0][4] == -100 # user eos
    assert out["labels"][0][5] == 97   # assistant 'a'
    assert out["labels"][0][6] == 100  # assistant eos
    assert out["labels"][0][7] == -100 # padded

    # Verify standard position_ids and seq_idx alignment
    assert list(out["position_ids"][0]) == [0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert list(out["seq_idx"][0]) == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

@patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer())
def test_sft_truncation(mock_load):
    actor = TokenizerActor("mock-model", seq_len=4, format_name="chatml")
    batch = {"messages": [[
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]]}
    out = actor(batch)

    assert len(out["input_ids"][0]) == 4
    assert len(out["labels"][0]) == 4
    assert list(out["labels"][0]) == [-100, -100, -100, -100] # Cuts before assistant
    assert list(out["position_ids"][0]) == [0, 1, 2, 3]
    assert list(out["seq_idx"][0]) == [1, 1, 1, 1]

@patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer())
def test_conversation_without_assistant(mock_load):
    actor = TokenizerActor("mock-model", seq_len=16, format_name="chatml")
    batch = {"messages": [[
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "u"},
    ]]}
    out = actor(batch)

    # Dummy returned as filter excludes rows where all labels == -100
    assert np.all(out["attention_mask"] == 0)
    assert np.all(out["labels"] == -100)
    assert np.all(out["position_ids"] == 0)
    assert np.all(out["seq_idx"] == 0)

@patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer())
def test_cpt_chunks_and_eos(mock_load):
    actor = TokenizerActor("mock-model", seq_len=5, text_col="text", is_cpt=True, train_on_inputs=True)
    batch = {"text": ["123", "45"]}

    # '123' -> [49, 50, 51] + EOS -> 4 tokens
    # '45' -> [52, 53] + EOS -> 3 tokens
    # Total = 7 tokens -> 1 chunk of 5, 2 remainder
    out = actor(batch)
    assert out["input_ids"].shape == (1, 5)
    assert np.array_equal(out["labels"], out["input_ids"])
    assert np.all(out["attention_mask"] == 1)

    assert list(out["input_ids"][0]) == [49, 50, 51, 100, 52]
    assert len(actor._remainder_ids) == 2
    assert actor._remainder_ids == [53, 100]

    # CPT acts as one continuous timeline
    assert list(out["position_ids"][0]) == [0, 1, 2, 3, 4]
    assert list(out["seq_idx"][0]) == [1, 1, 1, 1, 1]

@patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer())
def test_sft_multipack_ffd_logic(mock_load):
    """Test that First-Fit Decreasing packing respects document boundaries and position IDs."""
    actor = TokenizerActor("mock-model", seq_len=8, format_name="chatml", pack_sequences=True, train_on_inputs=False)

    # Document 1: 4 tokens -> 'u'+eos, 'a'+eos
    # Document 2: 2 tokens -> 'v'+eos (no assistant, all -100 label but still takes space)
    batch = {"messages": [
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        [
            {"role": "user", "content": "v"}
        ]
    ]}

    out = actor(batch)

    # They should both pack into the single 8-token sequence bucket, leaving 2 pad tokens
    assert out["input_ids"].shape == (1, 8)

    # Expected Document 1: 'u'=117, eos=100, 'a'=97, eos=100
    # Expected Document 2: 'v'=118, eos=100
    # Pad = 99
    assert list(out["input_ids"][0]) == [117, 100, 97, 100, 118, 100, 99, 99]

    # Doc 1 Assistant is index 2, 3. Rest are -100.
    assert list(out["labels"][0]) == [-100, -100, 97, 100, -100, -100, -100, -100]

    # IMPORTANT: positions must reset for Document 2!
    assert list(out["position_ids"][0]) == [0, 1, 2, 3, 0, 1, 0, 0]

    # IMPORTANT: Sequence index must increment for Document 2!
    assert list(out["seq_idx"][0]) == [1, 1, 1, 1, 2, 2, 0, 0]

def test_cache_invalidation():
    # Ensure all hash parameters are provided explicitly to avoid signature mismatches
    hash_sft_default = _cache_version_hash("falcon", 2048, "local", train_on_inputs=False, pack_sequences=False)
    hash_sft_packed  = _cache_version_hash("falcon", 2048, "local", train_on_inputs=False, pack_sequences=True)
    hash_cpt         = _cache_version_hash("falcon", 2048, "local", is_cpt=True)

    assert hash_sft_default != hash_sft_packed
    assert hash_sft_packed != hash_cpt
