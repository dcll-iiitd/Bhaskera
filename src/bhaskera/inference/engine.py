"""
bhaskera.inference.engine
==========================
InferenceEngine — the main entry point for LLM generation in Bhaskera.

Integrates:
  - Model loading (HuggingFace `AutoModelForCausalLM`)
  - ModelProfile detection via `bhaskera.introspect`
  - KV cache strategy selection (static / TurboQuant / none)
  - Speculative decoding (optional, zero quality loss)
  - Sampling (greedy / top-k / top-p / nucleus)
  - torch.compile for decode acceleration (optional)

Usage::

    from bhaskera.config import load_config
    from bhaskera.inference import InferenceEngine

    cfg = load_config("config.yaml")
    engine = InferenceEngine(cfg)
    outputs = engine.generate(["Hello, world!", "Translate to French:"])
    print(outputs)
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "auto":     None,     # resolved at runtime
}


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Bhaskera inference engine.

    Loads a causal LM from HuggingFace, wires up the configured KV cache
    strategy and optional speculative decoder, then exposes a simple
    `generate()` method that handles batching, sampling, and caching.

    Args:
        cfg:         Bhaskera `Config` object (loaded from YAML or defaults).
        model_name:  Optional override for `cfg.model.name`.
    """

    def __init__(self, cfg, model_name: Optional[str] = None):
        self.cfg        = cfg
        self.infer_cfg  = cfg.inference
        self.model_name = model_name or cfg.model.name
        self.device     = _resolve_device(self.infer_cfg.device)

        # Loaded lazily — call .load() explicitly or first call to .generate()
        self._model     = None
        self._tokenizer = None
        self._kv_cache  = None
        self._spec_dec  = None
        self._profile   = None
        self._loaded    = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> "InferenceEngine":
        """Load model, tokenizer, and configure KV cache + speculative decoder.

        Can be called explicitly or is triggered lazily by `generate()`.
        """
        if self._loaded:
            return self

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from bhaskera.introspect import introspect_model

        logger.info(f"[InferenceEngine] Loading model: {self.model_name}")
        t0 = time.time()

        # ── Resolve compute dtype ─────────────────────────────────────
        raw_dtype = getattr(self.cfg.model, "dtype", "bfloat16")
        compute_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

        # ── Tokenizer ────────────────────────────────────────────────
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.cfg.model.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ── Model ────────────────────────────────────────────────────
        model_kwargs = dict(
            torch_dtype=compute_dtype or "auto",
            device_map=str(self.device),
            trust_remote_code=self.cfg.model.trust_remote_code,
        )
        if self.cfg.model.attn_impl:
            model_kwargs["attn_implementation"] = self.cfg.model.attn_impl

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        self._model.eval()

        if compute_dtype is None:
            compute_dtype = next(self._model.parameters()).dtype

        # ── ModelProfile ─────────────────────────────────────────────
        self._profile = introspect_model(self._model)

        # ── KV Cache ─────────────────────────────────────────────────
        self._kv_cache = self._build_kv_cache(compute_dtype)

        # ── torch.compile (decode acceleration) ──────────────────────
        if self.infer_cfg.torch_compile:
            logger.info("[InferenceEngine] Applying torch.compile…")
            try:
                self._model = torch.compile(
                    self._model, mode="reduce-overhead", fullgraph=False
                )
                logger.info("[InferenceEngine] torch.compile applied ✓")
            except Exception as e:
                logger.warning(f"torch.compile failed (skipping): {e}")

        # ── Speculative decoding ──────────────────────────────────────
        if self.infer_cfg.speculative.enabled:
            from .speculative import build_speculative_decoder
            self._spec_dec = build_speculative_decoder(
                target_model=self._model,
                cfg=self.infer_cfg.speculative,
                infer_cfg=self.infer_cfg,
                device=self.device,
            )

        elapsed = time.time() - t0
        logger.info(f"[InferenceEngine] Ready in {elapsed:.1f}s on {self.device}")
        self._loaded = True
        return self

    def _build_kv_cache(self, dtype: torch.dtype):
        """Construct the KV cache object from inference config."""
        if self.infer_cfg.kv_cache == "none":
            return None

        from .kv_cache import build_kv_cache

        model_cfg = self._model.config
        num_layers = getattr(model_cfg, "num_hidden_layers", 0)
        if num_layers == 0:
            logger.warning("Could not detect num_hidden_layers — disabling KV cache.")
            return None

        # Head geometry — try multiple attribute names across architectures
        num_kv_heads = (
            getattr(model_cfg, "num_key_value_heads", None)
            or getattr(model_cfg, "num_attention_heads", None)
            or getattr(model_cfg, "n_head", 0)
        )
        hidden_size = (
            getattr(model_cfg, "hidden_size", None)
            or getattr(model_cfg, "n_embd", 0)
        )
        num_attn_heads = (
            getattr(model_cfg, "num_attention_heads", None)
            or getattr(model_cfg, "n_head", 1)
        )
        head_dim = hidden_size // num_attn_heads if num_attn_heads > 0 else 64

        # Max sequence length
        max_seq_len = (
            self.infer_cfg.max_new_tokens
            + getattr(model_cfg, "max_position_embeddings", 2048)
        )

        tq_cfg = (
            self.infer_cfg.turboquant
            if self.infer_cfg.kv_cache == "turboquant"
            else None
        )

        logger.info(
            f"[InferenceEngine] KV cache: strategy={self.infer_cfg.kv_cache} "
            f"layers={num_layers} heads={num_kv_heads} head_dim={head_dim} "
            f"max_seq={max_seq_len}"
        )

        return build_kv_cache(
            strategy=self.infer_cfg.kv_cache,
            num_layers=num_layers,
            batch_size=self.infer_cfg.batch_size,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=self.device,
            tq_cfg=tq_cfg,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        return_full_text: bool = False,
    ) -> List[str]:
        """Generate text completions for one or more prompts.

        Args:
            prompts:         A single string or list of strings.
            max_new_tokens:  Override cfg.inference.max_new_tokens.
            temperature:     Override cfg.inference.temperature.
            top_p:           Override cfg.inference.top_p.
            top_k:           Override cfg.inference.top_k.
            do_sample:       Override cfg.inference.do_sample.
            return_full_text: If True, prepend the prompt to the output.

        Returns:
            List of generated string completions (one per prompt).
        """
        if not self._loaded:
            self.load()

        if isinstance(prompts, str):
            prompts = [prompts]

        # Resolve generation parameters (call-level override > config default)
        gen_cfg = self.infer_cfg
        max_new  = max_new_tokens if max_new_tokens is not None else gen_cfg.max_new_tokens
        temp     = temperature    if temperature    is not None else gen_cfg.temperature
        p        = top_p          if top_p          is not None else gen_cfg.top_p
        k        = top_k          if top_k          is not None else gen_cfg.top_k
        sample   = do_sample      if do_sample      is not None else gen_cfg.do_sample

        # Tokenise
        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        if self._spec_dec is not None:
            # Speculative decoding path
            generated_ids = self._generate_speculative(
                input_ids, attention_mask, max_new, temp, p, k
            )
        else:
            # Standard autoregressive path (with or without KV cache)
            generated_ids = self._generate_autoregressive(
                input_ids, attention_mask, max_new, temp, p, k, sample
            )

        # Decode outputs
        prompt_lens = [enc["input_ids"].shape[1]] * len(prompts)
        outputs = []
        for i, ids in enumerate(generated_ids):
            if return_full_text:
                text = self._tokenizer.decode(ids, skip_special_tokens=True)
            else:
                new_ids = ids[prompt_lens[i]:]
                text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            outputs.append(text)

        return outputs

    # ------------------------------------------------------------------
    # Autoregressive generation loop
    # ------------------------------------------------------------------

    def _generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> List[torch.Tensor]:
        """Pure autoregressive generation with HF `past_key_values` or custom KV cache."""
        from .sampling import sample_from_logits

        batch_size  = input_ids.shape[0]
        eos_id      = self._tokenizer.eos_token_id
        past_kv     = None
        generated   = input_ids.clone()
        finished    = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        cur_input   = input_ids
        cur_mask    = attention_mask

        # Reset custom KV cache if present
        if self._kv_cache is not None:
            self._kv_cache.reset()
            past_kv = self._kv_cache

        for step in range(max_new_tokens):
            # ── Forward pass ──────────────────────────────────────────
            with torch.autocast(
                self.device.type if self.device.type != "cpu" else "cpu",
                enabled=(self.device.type != "cpu"),
            ):
                out = self._model(
                    input_ids=cur_input,
                    attention_mask=cur_mask,
                    past_key_values=past_kv,
                    use_cache=True,  # always use build-in caching logic
                )

            logits  = out.logits[:, -1, :]  # (batch, vocab)
            past_kv = out.past_key_values   # updated HF KV or custom cache

            # Advance custom cache sequence length pointer
            if self._kv_cache is not None:
                self._kv_cache.advance(cur_input.shape[1])

            # ── Sample next token ──────────────────────────────────────
            next_token = sample_from_logits(
                logits, temperature=temperature, top_p=top_p,
                top_k=top_k, do_sample=do_sample,
            )  # (batch,)

            # Mark sequences that hit EOS
            if eos_id is not None:
                finished = finished | (next_token == eos_id)

            generated  = torch.cat([generated,  next_token.unsqueeze(1)], dim=1)
            cur_input  = next_token.unsqueeze(1)
            # Extend attention mask by 1
            cur_mask   = torch.cat(
                [cur_mask, torch.ones(batch_size, 1, device=self.device, dtype=cur_mask.dtype)],
                dim=1,
            )

            if finished.all():
                logger.debug(f"All sequences finished at step {step + 1}")
                break

        return [generated[i] for i in range(batch_size)]

    # ------------------------------------------------------------------
    # Speculative generation loop
    # ------------------------------------------------------------------

    def _generate_speculative(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[torch.Tensor]:
        """Speculative decoding outer loop."""
        batch_size = input_ids.shape[0]
        eos_id     = self._tokenizer.eos_token_id
        generated  = input_ids.clone()
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        target_past_kv = None
        draft_past_kv  = None
        cur_input      = input_ids
        total_new      = 0

        spec = self._spec_dec
        spec.temperature = temperature
        spec.top_p       = top_p
        spec.top_k       = top_k

        while total_new < max_new_tokens:
            new_tokens, target_past_kv, draft_past_kv = spec.generate_step(
                input_ids=cur_input,
                target_past_kv=target_past_kv,
                draft_past_kv=draft_past_kv,
            )
            # new_tokens: (batch, n_accepted)
            n_accepted = new_tokens.shape[1]

            if eos_id is not None:
                eos_mask = (new_tokens == eos_id).any(dim=1)
                finished = finished | eos_mask

            generated  = torch.cat([generated, new_tokens], dim=1)
            cur_input  = new_tokens[:, -1:]  # feed back only the last accepted token
            total_new += n_accepted

            if finished.all():
                break

        return [generated[i] for i in range(batch_size)]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def kv_cache_stats(self) -> Optional[dict]:
        """Return KV cache memory stats (TurboQuant only)."""
        if self._kv_cache is None:
            return None
        if hasattr(self._kv_cache, "compression_stats"):
            return self._kv_cache.compression_stats()
        return {"bytes": self._kv_cache.memory_bytes()}

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"InferenceEngine("
            f"model={self.model_name!r}, "
            f"kv_cache={self.infer_cfg.kv_cache!r}, "
            f"device={self.device}, "
            f"status={status})"
        )
