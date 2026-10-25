"""Tiny real Llama + word-level tokenizer vocabulary shared by the RL tests."""
from __future__ import annotations

import torch

WORDS = ["<pad>", "<eos>", "<unk>", "Q", "A"] + [str(i) for i in range(10)]


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_base_model(tokenizer, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        # Wider init than the 0.02 default: with a tiny hidden size the default caps logit
        # range so tightly that even supervised training cannot move token probabilities.
        initializer_range=0.3,
    )
    return LlamaForCausalLM(cfg)


def make_lora_model(tokenizer, seed: int = 0):
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=TARGETS)
    return get_peft_model(make_base_model(tokenizer, seed), lora)
