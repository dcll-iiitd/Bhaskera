#!/usr/bin/env bash
# Decision-serving benchmark matrix on one GPU (spec §8). On the GPU host, from the bhaskera
# checkout, with $BJ/env.sh sourced:
#   BASE_CONFIG=$BJ/serve_jevos.a6000.yaml JEV_DIR=$BJ/jev GGUF_DIR=$BJ/cache/gguf/jevos \
#     bash scripts/decision/run_bench.sh upstream|replicas|staterestore|quant|gateway|batching
set -euo pipefail
MODE=${1:?usage: run_bench.sh upstream|replicas|staterestore|quant|gateway|batching}
BASE=${BASE_CONFIG:?path to this host's serve_jevos config}
JEV_DIR=${JEV_DIR:?path to the uv-synced upstream jev checkout}
GGUF_DIR=${GGUF_DIR:?directory holding jevos-v2-q8_0.gguf and jevos-v2-q4_k_m.gguf}
OUT=${OUT:-benchmarks/decision/results}
LOGS=${LOGS:-$BJ/logs/bench}
DURATION=${DURATION:-15}
WARMUP=${WARMUP:-3}
LEVELS=${LEVELS:-1,4,16,64}
BEST=${BEST_REPLICAS:-4}
PY=.venv/bin/python
mkdir -p "$OUT" "$LOGS" benchmarks/decision/requests
$PY scripts/decision/make_requests.py bench --out benchmarks/decision/requests >/dev/null
SETS=(benchmarks/decision/requests/bench_*.jsonl)

wait_http() { # $1 url that must answer 2xx
  for _ in $(seq 900); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done
  echo "never ready: $1" >&2; return 1
}
bench_all() { # $1 label, $2 base url, rest: extra bench.py args
  local label=$1 url=$2; shift 2
  for set in "${SETS[@]}"; do
    $PY scripts/decision/bench.py --url "$url" --requests "$set" --levels "$LEVELS" \
      --duration "$DURATION" --warmup "$WARMUP" --label "$label" --out "$OUT/results.jsonl" "$@"
  done
}
stop() { kill -TERM "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; sleep 8; }

upstream() { # $1 device, $2 quant
  (cd "$JEV_DIR" && exec uv run jev serve --gguf "$GGUF_DIR/jevos-v2-$2.gguf" --device "$1" \
    --threads 32 --port 8017) > "$LOGS/upstream-$1-$2.log" 2>&1 &
  local pid=$!
  wait_http http://127.0.0.1:8017/health
  bench_all "upstream-$1-$2" http://127.0.0.1:8017
  stop $pid
}

PID=
bhaskera() { # $1 label, $2 replicas, rest: key=value config overrides
  local label=$1 replicas=$2; shift 2
  $PY scripts/decision/variant.py "$BASE" "$LOGS/$label.yaml" "$@"
  .venv/bin/bhaskera-serve --config "$LOGS/$label.yaml" --num-replicas "$replicas" --port 8100 \
    --ray-address local --log-level WARNING > "$LOGS/$label.log" 2>&1 &
  PID=$!
  wait_http http://127.0.0.1:8100/health
}

case $MODE in
  upstream)
    upstream cpu q8_0; upstream cuda q8_0; upstream cuda q4_k_m ;;
  replicas)
    for n in 1 2 4 8; do
      bhaskera "bhaskera-r$n-q8_0" $n
      bench_all "bhaskera-r$n-q8_0" http://127.0.0.1:8100; stop $PID
    done ;;
  quant)
    bhaskera "bhaskera-r$BEST-q4_k_m" "$BEST" model.gguf=jevos-q4_k_m
    bench_all "bhaskera-r$BEST-q4_k_m" http://127.0.0.1:8100; stop $PID ;;
  gateway)
    bhaskera "bhaskera-r$BEST-gateway" "$BEST" serve.gateway.enabled=true \
      serve.gateway.proxy_port=8200 serve.gateway.cloudflared=false
    wait_http http://127.0.0.1:8200/docs
    bench_all "bhaskera-r$BEST-gateway" http://127.0.0.1:8200 --api-key sk-bhaskera-admin; stop $PID ;;
  staterestore)
    bhaskera "bhaskera-r$BEST-staterestore" "$BEST" serve.decision.branch=state-restore
    bench_all "bhaskera-r$BEST-staterestore" http://127.0.0.1:8100; stop $PID ;;
  batching)
    bhaskera "bhaskera-r$BEST-batched" "$BEST" serve.decision.batching.enabled=true
    bench_all "bhaskera-r$BEST-batched" http://127.0.0.1:8100; stop $PID ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
