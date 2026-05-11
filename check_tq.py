# Quick check script
from bhaskera.config import load_config
from bhaskera.inference import InferenceEngine

cfg = load_config("configs/inference_param2.yaml")
engine = InferenceEngine(cfg)
engine.load()

# Check 1: what backend?
print(f"Backend: {engine._backend_name}")

# Check 2: is TurboQuant configured?
print(f"KV cache strategy: {cfg.inference.kv_cache}")
print(f"TurboQuant enabled: {cfg.inference.turboquant.enabled}")
print(f"Key bits: {cfg.inference.turboquant.key_bits}")
print(f"Value bits: {cfg.inference.turboquant.value_bits}")

# Check 3: does the backend have a kv_cache?
if hasattr(engine._backend, '_kv_cache'):
    kv = engine._backend._kv_cache
    print(f"KV cache object: {type(kv).__name__}")
    if kv is None:
        print("WARNING: _kv_cache is None — TurboQuant NOT active")
else:
    print("WARNING: backend has no _kv_cache attr")

# Check 4: generate and inspect stats
outputs = engine.generate(["Hello world"], max_new_tokens=20)
stats = engine.kv_cache_stats()
print(f"KV cache stats: {stats}")
