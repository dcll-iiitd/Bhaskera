"""
bhaskera.data
=============
Ray Data–native dataset pipeline.

Adding a new dataset: create a function with signature
    build(cfg) -> ray.data.Dataset
and register it with @register('name'). That's it.
"""
from __future__ import annotations
from typing import Callable
import logging

import ray.data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a dataset builder."""
    def _wrap(fn):
        REGISTRY[name] = fn
        return fn
    return _wrap


def build_ray_dataset(cfg) -> ray.data.Dataset:
    source = str(_cfg_get(cfg, "data", "source", default="registry")).lower()

    if source == "registry":
        return _build_registry_dataset(cfg)
    if source == "huggingface":
        return _build_huggingface_dataset(cfg)
    if source == "local":
        return _build_local_dataset(cfg)

    raise ValueError(
        f"Unsupported data.source '{source}'. "
        "Expected one of: registry, huggingface, local."
    )


def _cfg_get(cfg, *keys, default=None):
    sentinel = object()
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, sentinel)
        else:
            cur = getattr(cur, key, sentinel)
        if cur is sentinel:
            return default
    return cur


def _build_registry_dataset(cfg) -> ray.data.Dataset:
    name = _cfg_get(cfg, "data", "name", default="ultrachat")
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(REGISTRY)}. "
            "Register yours with @register('name')."
        )
    logger.info(f"Building registry dataset: {name}")
    return REGISTRY[name](cfg)


def _build_huggingface_dataset(cfg) -> ray.data.Dataset:
    from datasets import load_dataset

    dataset_name = str(_cfg_get(cfg, "data", "hf_dataset", default="")).strip()
    split = str(_cfg_get(cfg, "data", "hf_split", default="train")).strip()

    if not dataset_name:
        raise ValueError(
            "data.hf_dataset is required when data.source='huggingface'."
        )

    logger.info(f"Loading HuggingFace dataset '{dataset_name}' split='{split}'")
    hf_ds = load_dataset(dataset_name, split=split)
    ds = ray.data.from_huggingface(hf_ds)
    return _prepare_text_and_tokenize(ds, cfg, source_desc=f"hf:{dataset_name}")


def _build_local_dataset(cfg) -> ray.data.Dataset:
    local_path = str(_cfg_get(cfg, "data", "local_path", default="")).strip()
    local_format = str(_cfg_get(cfg, "data", "local_format", default="jsonl")).lower().strip()

    if not local_path:
        raise ValueError("data.local_path is required when data.source='local'.")

    readers = {
        "json": ray.data.read_json,
        "jsonl": ray.data.read_json,
        "csv": ray.data.read_csv,
        "parquet": ray.data.read_parquet,
        "text": ray.data.read_text,
        "txt": ray.data.read_text,
    }
    if local_format not in readers:
        raise ValueError(
            f"Unsupported data.local_format '{local_format}'. "
            "Expected one of: jsonl, json, csv, parquet, text."
        )

    logger.info(f"Loading local dataset from '{local_path}' (format={local_format})")
    ds = readers[local_format](local_path)
    return _prepare_text_and_tokenize(ds, cfg, source_desc=f"local:{local_path}")


def _prepare_text_and_tokenize(ds: ray.data.Dataset, cfg, source_desc: str) -> ray.data.Dataset:
    text_col = str(_cfg_get(cfg, "data", "text_column", default="text") or "text")
    prompt_col = str(_cfg_get(cfg, "data", "prompt_column", default="") or "").strip()
    completion_col = str(_cfg_get(cfg, "data", "completion_column", default="") or "").strip()

    if (prompt_col and not completion_col) or (completion_col and not prompt_col):
        raise ValueError(
            "data.prompt_column and data.completion_column must be provided together."
        )

    if prompt_col and completion_col:
        _validate_columns(ds, [prompt_col, completion_col], source_desc=source_desc)
        template = str(
            _cfg_get(
                cfg,
                "data",
                "prompt_completion_template",
                default="{prompt}\n{completion}",
            )
        )
        rendered_col = "__bhaskera_text"
        ds = ds.map_batches(
            _render_prompt_completion_batch,
            batch_format="numpy",
            fn_kwargs={
                "prompt_column": prompt_col,
                "completion_column": completion_col,
                "template": template,
                "output_column": rendered_col,
            },
        )
        return _tokenize_dataset(ds, cfg, text_col=rendered_col)

    _validate_columns(ds, [text_col], source_desc=source_desc)
    return _tokenize_dataset(ds, cfg, text_col=text_col)


def _render_prompt_completion_batch(
    batch: dict,
    *,
    prompt_column: str,
    completion_column: str,
    template: str,
    output_column: str,
) -> dict:
    import numpy as np

    prompts = batch[prompt_column]
    completions = batch[completion_column]

    if hasattr(prompts, "tolist"):
        prompts = prompts.tolist()
    if hasattr(completions, "tolist"):
        completions = completions.tolist()

    texts = np.array(
        [
            template.format(prompt=str(prompt), completion=str(completion))
            for prompt, completion in zip(prompts, completions)
        ],
        dtype=object,
    )
    return {output_column: texts}


def _validate_columns(ds: ray.data.Dataset, required: list[str], source_desc: str) -> None:
    sample = ds.take(1)
    if not sample:
        raise ValueError(f"Dataset '{source_desc}' is empty.")

    row = sample[0]
    missing = [col for col in required if col not in row]
    if missing:
        available = sorted(row.keys())
        raise ValueError(
            f"Dataset '{source_desc}' is missing required column(s): {missing}. "
            f"Available columns: {available}"
        )


# ---------------------------------------------------------------------------
# Built-in datasets
# ---------------------------------------------------------------------------

@register("ultrachat")
def _ultrachat(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ray.data.from_huggingface(hf_ds)
    return _tokenize_dataset(ds, cfg, text_col="prompt")


@register("openassistant")
def _openassistant(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("OpenAssistant/oasst1", split="train")
    ds = ray.data.from_huggingface(hf_ds)
    return _tokenize_dataset(ds, cfg, text_col="text")


@register("redpajama")
def _redpajama(cfg) -> ray.data.Dataset:
    from datasets import load_dataset
    hf_ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    ds = ray.data.from_huggingface(hf_ds)
    return _tokenize_dataset(ds, cfg, text_col="text")


# ---------------------------------------------------------------------------
# Tokenisation — class-based actor for efficiency
# ---------------------------------------------------------------------------
# BEFORE: tokenizer was instantiated INSIDE the map function, meaning it was
#         loaded from disk for EVERY batch on EVERY worker.
# AFTER:  TokenizerActor loads the tokenizer ONCE in __init__, reuses across
#         all batches assigned to that actor. Much faster for large tokenizers
#         (like Param2's Indic language tokenizer).
# ---------------------------------------------------------------------------

class TokenizerActor:
    """
    Stateful batch tokenizer for Ray Data.
    Loads the tokenizer once; processes many batches.
    """

    def __init__(self, model_name: str, seq_len: int, trust_remote_code: bool = False):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self._text_col: str = "text"  # set by caller via _TokenizerActorFactory

    def __call__(self, batch: dict) -> dict:
        texts = batch[self._text_col]
        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        out = self.tokenizer(
            texts,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return {
            "input_ids":      out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels":         out["input_ids"].copy(),
        }


class _TokenizerActorFactory:
    """
    Factory that creates TokenizerActor instances with the right text_col.
    This is needed because Ray Data's class-based map_batches expects a callable
    class, and we need to pass the text_col dynamically.
    """

    def __init__(self, model_name: str, seq_len: int, text_col: str,
                 trust_remote_code: bool = False):
        self.model_name = model_name
        self.seq_len = seq_len
        self.text_col = text_col
        self.trust_remote_code = trust_remote_code

    def __call__(self, batch: dict) -> dict:
        # Lazy init — loads tokenizer on first call, then reuses
        if not hasattr(self, "_actor"):
            self._actor = TokenizerActor(
                self.model_name, self.seq_len, self.trust_remote_code
            )
            self._actor._text_col = self.text_col
        return self._actor(batch)


def _tokenize_dataset(
    ds: ray.data.Dataset,
    cfg,
    text_col: str,
) -> ray.data.Dataset:
    model_name = cfg.model.name
    seq_len = cfg.data.seq_len
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)
    num_workers = getattr(cfg.data, "num_workers", 4)

    factory = _TokenizerActorFactory(
        model_name=model_name,
        seq_len=seq_len,
        text_col=text_col,
        trust_remote_code=trust_remote_code,
    )
    ds = ds.repartition(num_workers * 2)
    return ds.map_batches(
        factory,
        batch_format="numpy",
        batch_size=256,
        num_cpus=1,
        concurrency=num_workers,
    )
