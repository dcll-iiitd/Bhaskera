"""
bhaskera.inference.backends.hf
================================
HuggingFace ``model.generate()`` backend.

Used when:
  - vLLM is not installed
  - ``BHASKERA_BACKEND=hf`` is set
  - Device is CPU or MPS (vLLM requires CUDA)

Compared to the Ray/vLLM backend this is slower (no PagedAttention,
no CUDA graphs, no continuous batching) but requires no extra dependencies
and works on any hardware.

Speculative decoding is supported via the existing
``bhaskera.inference.speculative`` module when
``cfg.inference.speculative.enabled = true``.

TurboQuant (HF path)
--------------------
The ``TurboQuantKVCache`` from ``kv_cache.py`` is wired here for the HF
path — it plugs directly into HF's ``past_key_values`` interface.  The
Ray/vLLM path defers TurboQuant to a future milestone.
"""
from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "auto":     None,
}

# Thread pool so tokeniser CPU work never blocks the GPU
_TOKENIZER_POOL = ThreadPoolExecutor(max_workers=2)

# Model IDs recognised as Param2 / thinking models
_PARAM2_MODEL_IDS = {
    "bharatgenai/param2-17b-a2.4b-thinking",
    "bharatgenai/param2-17b-a2.4b",
    "bharatgenai/param-2-17b-moe-a2.4b",
}


def _is_param2(model_name: str) -> bool:
    n = model_name.lower()
    return any(n == mid for mid in _PARAM2_MODEL_IDS) or "param2" in n or "param-2" in n


class HFBackend:
    """
    HuggingFace generate() backend with optional TurboQuant KV cache.

    Parameters
    ----------
    model_name : str
    cfg        : Bhaskera Config
    device     : torch.device
    """

    def __init__(self, model_name: str, cfg, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"[HFBackend] Loading {model_name!r} on {device}")
        t0 = time.perf_counter()

        self._cfg        = cfg
        self._device     = device
        self._model_name = model_name
        infer_cfg        = cfg.inference
        model_cfg        = cfg.model

        raw_dtype    = getattr(model_cfg, "dtype", "bfloat16")
        self._dtype  = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

        # ── Tokenizer ────────────────────────────────────────────────
        self._tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=getattr(model_cfg, "trust_remote_code", False),
        )
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        # ── Model ────────────────────────────────────────────────────
        load_kwargs: dict = dict(
            torch_dtype=self._dtype or "auto",
            trust_remote_code=getattr(model_cfg, "trust_remote_code", False),
            low_cpu_mem_usage=True,
        )
        if getattr(model_cfg, "attn_impl", None):
            load_kwargs["attn_implementation"] = model_cfg.attn_impl

        visible = torch.cuda.device_count()
        if device.type == "cuda" and visible > 1:
            single_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            load_kwargs["device_map"] = "cuda:0" if single_vram >= 38.0 else "auto"
        else:
            load_kwargs["device_map"] = str(device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self._model.eval()

        if self._dtype is None:
            self._dtype = next(self._model.parameters()).dtype

        # ── Param2 / thinking model detection ────────────────────────
        self._is_thinking = _is_param2(model_name)

        # ── KV cache ─────────────────────────────────────────────────
        self._kv_cache = self._build_kv_cache()

        # ── Speculative decoding ──────────────────────────────────────
        self._spec_dec = None
        if getattr(infer_cfg.speculative, "enabled", False):
            from bhaskera.inference.speculative import build_speculative_decoder
            self._spec_dec = build_speculative_decoder(
                target_model=self._model,
                cfg=infer_cfg.speculative,
                infer_cfg=infer_cfg,
                device=device,
            )

        elapsed = time.perf_counter() - t0
        logger.info(f"[HFBackend] Ready in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # KV cache wiring
    # ------------------------------------------------------------------

    def _build_kv_cache(self):
        from bhaskera.inference.kv_cache import build_kv_cache

        infer_cfg = self._cfg.inference
        if infer_cfg.kv_cache == "none":
            return None

        mc = self._model.config
        num_layers   = getattr(mc, "num_hidden_layers", 0)
        num_kv_heads = (
            getattr(mc, "num_key_value_heads", None)
            or getattr(mc, "multi_query_group_num", None)
            or 1
        )
        hidden_size = getattr(mc, "hidden_size", None) or getattr(mc, "n_embd", 0)
        num_attn    = getattr(mc, "num_attention_heads", None) or getattr(mc, "n_head", 1)
        head_dim    = hidden_size // num_attn if num_attn else 64
        max_pos     = getattr(mc, "max_position_embeddings", 2048)
        max_seq_len = infer_cfg.max_new_tokens + max_pos

        if num_layers == 0:
            logger.warning("[HFBackend] Could not detect num_hidden_layers — KV cache disabled")
            return None

        tq_cfg = infer_cfg.turboquant if infer_cfg.kv_cache == "turboquant" else None
        return build_kv_cache(
            strategy=infer_cfg.kv_cache,
            num_layers=num_layers,
            batch_size=infer_cfg.batch_size,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=self._dtype,
            device=self._device,
            tq_cfg=tq_cfg,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        return_full_text: bool = False,
    ) -> List[str]:
        infer_cfg = self._cfg.inference

        # Tokenize — thinking models need chat template
        if self._is_thinking:
            input_ids, attention_mask, prompt_len = self._apply_chat_template(prompts)
        else:
            enc_future = _TOKENIZER_POOL.submit(
                self._tok, prompts,
                return_tensors="pt", padding=True, truncation=True,
            )
            enc = enc_future.result()
            input_ids      = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)
            prompt_len     = input_ids.shape[1]

        # Reset KV cache between requests
        if self._kv_cache is not None:
            self._kv_cache.reset()

        gen_kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tok.eos_token_id,
            eos_token_id=self._tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"]       = top_p
            if top_k > 0:
                gen_kwargs["top_k"]   = top_k

        if self._kv_cache is not None:
            gen_kwargs["past_key_values"] = self._kv_cache
        else:
            try:
                from transformers import DynamicCache
                gen_kwargs["past_key_values"] = DynamicCache()
            except ImportError:
                pass

        ctx = (
            torch.autocast(self._device.type, dtype=self._dtype)
            if self._device.type in ("cuda", "cpu")
            else torch.autocast("cpu", dtype=self._dtype)
        )
        with ctx:
            if self._spec_dec is not None:
                output_ids = self._generate_speculative(
                    input_ids, attention_mask, max_new_tokens,
                    temperature, top_p, top_k
                )
            else:
                output_ids = self._model.generate(**gen_kwargs)

        # Decode — thinking models: keep <think> tags (skip_special_tokens=False)
        skip_sp = not self._is_thinking
        outputs = []
        for i, ids in enumerate(output_ids):
            if return_full_text:
                text = self._tok.decode(ids, skip_special_tokens=skip_sp)
            else:
                text = self._tok.decode(ids[prompt_len:], skip_special_tokens=skip_sp)
            outputs.append(text)
        return outputs

    def generate_param2(
        self,
        prompts: List[str],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> List[str]:
        """
        Param2-Thinking generation with chat template.  Returns raw strings
        including <think> tags; caller handles parsing.
        """
        if self._kv_cache is not None:
            self._kv_cache.reset()

        raw_outputs = []
        infer_cfg = self._cfg.inference

        for prompt in prompts:
            input_ids = self._tok.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self._device)

            gen_kwargs: dict = dict(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self._tok.eos_token_id,
                pad_token_id=self._tok.eos_token_id,
            )
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
            if self._kv_cache is not None:
                self._kv_cache.reset()
                gen_kwargs["past_key_values"] = self._kv_cache

            ctx = (
                torch.autocast(self._device.type, dtype=self._dtype)
                if self._device.type in ("cuda", "cpu")
                else torch.autocast("cpu", dtype=self._dtype)
            )
            with ctx:
                output_ids = self._model.generate(**gen_kwargs)

            prompt_len = input_ids.shape[1]
            # CRITICAL: skip_special_tokens=False to preserve <think> tags
            raw_text = self._tok.decode(
                output_ids[0][prompt_len:], skip_special_tokens=False
            )
            raw_outputs.append(raw_text)

        return raw_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, prompts: List[str]):
        """Tokenize a batch of prompts via apply_chat_template with padding."""
        all_ids = []
        for p in prompts:
            ids = self._tok.apply_chat_template(
                [{"role": "user", "content": p}],
                return_tensors="pt",
                add_generation_prompt=True,
            )
            all_ids.append(ids[0])

        max_len = max(t.shape[0] for t in all_ids)
        pad_id  = self._tok.pad_token_id or self._tok.eos_token_id
        padded  = [
            torch.cat([
                torch.full((max_len - t.shape[0],), pad_id, dtype=torch.long),
                t,
            ])
            for t in all_ids
        ]
        input_ids      = torch.stack(padded).to(self._device)
        attention_mask = (input_ids != pad_id).long()
        prompt_len     = input_ids.shape[1]
        return input_ids, attention_mask, prompt_len

    def _generate_speculative(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        from bhaskera.inference.sampling import sample_from_logits

        batch_size = input_ids.shape[0]
        eos_id     = self._tok.eos_token_id
        generated  = input_ids.clone()
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=self._device)
        target_pkv = None
        draft_pkv  = None
        cur_input  = input_ids
        total_new  = 0
        spec       = self._spec_dec
        spec.temperature = temperature
        spec.top_p       = top_p
        spec.top_k       = top_k

        while total_new < max_new_tokens:
            new_tokens, target_pkv, draft_pkv = spec.generate_step(
                input_ids=cur_input,
                target_past_kv=target_pkv,
                draft_past_kv=draft_pkv,
            )
            n_accepted = new_tokens.shape[1]
            if eos_id is not None:
                finished = finished | (new_tokens == eos_id).any(dim=1)
            generated  = torch.cat([generated, new_tokens], dim=1)
            cur_input  = new_tokens[:, -1:]
            total_new += n_accepted
            if finished.all():
                break
        return generated

    def kv_cache_stats(self) -> Optional[dict]:
        if self._kv_cache is None:
            return None
        if hasattr(self._kv_cache, "compression_stats"):
            return self._kv_cache.compression_stats()
        return {"bytes": self._kv_cache.memory_bytes()}