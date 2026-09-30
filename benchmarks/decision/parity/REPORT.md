# Decision parity report: Bhaskera vs upstream jev

- Date: 2026-09-30
- Host: A6000 (GPU 0), driver 580, llama.cpp b11081 (161755f) CUDA 13.4 build from Task 8
- Upstream jev: feder-cr/jev@5be02a3db3b0b32f01c176d8cd122f8c2a8e4db1; weights jevos-v2 (q8_0, q4_k_m)
- Bhaskera: branch serve-jevos, branch `probe` default (bhaskera-serve, /v1/systemone)
- Request set: 300 deterministic requests (1100 answers), tolerance 0.01 on |dP(yes)|

| reference | candidate | quant | answers | max_delta | mean_delta | decisions_flipped | passed |
|---|---|---|---|---|---|---|---|
| upstream CPU | Bhaskera CUDA | q8_0 | 1100 | 0.0537 | 0.00375 | 3 | false |
| upstream CUDA | Bhaskera CUDA | q8_0 | 1100 | 0.0527 | 0.00197 | 5 | false |
| upstream CPU | upstream CUDA | q8_0 | 1100 | 0.0710 | 0.00374 | 6 | false |
| upstream CPU | Bhaskera CUDA | q4_k_m | 1100 | 0.1467 | 0.00780 | 11 | false |
| upstream CUDA | Bhaskera CUDA | q4_k_m | 1100 | 0.1103 | 0.00392 | 4 | false |
| upstream CPU | upstream CUDA | q4_k_m | 1100 | 0.1115 | 0.00782 | 11 | false |
| upstream CUDA | Bhaskera CUDA (branch=state-restore) | q8_0 | 1100 | 0.0000 | 0.00000 | 0 | true |
| upstream CPU | Bhaskera CUDA (branch=state-restore) | q8_0 | 1100 | 0.0710 | 0.00374 | 6 | false |
| upstream CUDA | Bhaskera CUDA (branch=state-restore) | q4_k_m | 1100 | 0.0000 | 0.00000 | 0 | true |
| upstream CPU | Bhaskera CUDA (branch=state-restore) | q4_k_m | 1100 | 0.1115 | 0.00782 | 11 | false |

Diagnostic (state-restore): upstream's /health reports `branch_strategy: state-restore` (its default), while Bhaskera's default is `probe`/auto. With Bhaskera set to state-restore, its CUDA answers are bit-identical to upstream CUDA (max_delta 0.0, 0 flips, 1100 answers, both quants), so the port itself is exact and the drift in the gate rows is branch-strategy/batching numerics on GPU plus llama.cpp CUDA-vs-CPU numerics; the gate verdict below is unchanged.

Parity gate (upstream CPU vs Bhaskera CUDA, <= 0.01): **FAIL** for q8_0 and q4_k_m.

Diagnosis: upstream CPU vs upstream CUDA (same GGUF, upstream code on both sides) also exceeds 0.01 for both quants (q8_0 0.0710, q4_k_m 0.1115), so the drift is llama.cpp CUDA vs CPU numerics, not the Bhaskera port. Bhaskera CUDA sits closer to upstream CUDA (q8_0 0.0527, q4_k_m 0.1103, mean_delta about half) than to upstream CPU, but is not identical to it (max_delta > 0.01 with 4-5 flips), the residual is the probe-vs-state-restore branch difference (see the state-restore rows: bit-identical to upstream CUDA). Tolerance was not relaxed.
