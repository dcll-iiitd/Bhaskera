# Decision parity report: Bhaskera vs upstream jev

- Date: 2026-09-30
- Host: A6000 (GPU 0), driver 580, llama.cpp b11081 (161755f) CUDA 13.4 build from Task 8
- Upstream jev: feder-cr/jev@5be02a3db3b0b32f01c176d8cd122f8c2a8e4db1; weights jevos-v2 (q8_0, q4_k_m)
- Bhaskera: branch serve-jevos, branch `probe` default (bhaskera-serve, /v1/systemone)
- Request set: 300 deterministic requests (1100 answers), tolerance 0.01 on |dP(yes)|
- Gate (amended by the user on 2026-09-30): port fidelity, max |dP(yes)| <= 0.01 between Bhaskera and upstream jev on the same device with the same branch strategy. The original gate (upstream CPU vs Bhaskera CUDA) was dropped because llama.cpp's own CPU vs CUDA drift exceeds 0.01 even for upstream vs itself (0.071 on q8_0). CPU vs GPU drift and seq-copy vs state-restore drift are reported numbers, not gates.

| reference | candidate | quant | answers | max_delta | mean_delta | decisions_flipped | passed |
|---|---|---|---|---|---|---|---|
| upstream CPU | Bhaskera CUDA (default branch) | q8_0 | 1100 | 0.0537 | 0.00375 | 3 | false |
| upstream CUDA | Bhaskera CUDA (default branch) | q8_0 | 1100 | 0.0527 | 0.00197 | 5 | false |
| upstream CPU | upstream CUDA | q8_0 | 1100 | 0.0710 | 0.00374 | 6 | false |
| upstream CPU | Bhaskera CUDA (default branch) | q4_k_m | 1100 | 0.1467 | 0.00780 | 11 | false |
| upstream CUDA | Bhaskera CUDA (default branch) | q4_k_m | 1100 | 0.1103 | 0.00392 | 4 | false |
| upstream CPU | upstream CUDA | q4_k_m | 1100 | 0.1115 | 0.00782 | 11 | false |
| upstream CUDA | Bhaskera CUDA (branch=state-restore) | q8_0 | 1100 | 0.0000 | 0.00000 | 0 | true |
| upstream CPU | Bhaskera CUDA (branch=state-restore) | q8_0 | 1100 | 0.0710 | 0.00374 | 6 | false |
| upstream CUDA | Bhaskera CUDA (branch=state-restore) | q4_k_m | 1100 | 0.0000 | 0.00000 | 0 | true |
| upstream CPU | Bhaskera CUDA (branch=state-restore) | q4_k_m | 1100 | 0.1115 | 0.00782 | 11 | false |

Diagnostic (state-restore): upstream's /health reports `branch_strategy: state-restore` (its default), while Bhaskera's default is `probe`/auto. With Bhaskera set to state-restore, its CUDA answers are bit-identical to upstream CUDA (max_delta 0.0, 0 flips, 1100 answers, both quants), so the port itself is exact and the drift in the gate rows is branch-strategy/batching numerics on GPU plus llama.cpp CUDA-vs-CPU numerics; the gate verdict below is unchanged.

Reported drift (not a gate): the original comparison, upstream CPU vs Bhaskera CUDA with the default seq-copy branch, gave max 0.054 / 0.147 (q8_0 / q4_k_m) versus upstream's own CPU vs CUDA drift of 0.071 / 0.112; mean deltas are equal (0.0037 / 0.0078), flips 3 vs 6 (q8_0) and 11 vs 11 (q4_k_m). Seq-copy is Bhaskera's default; the benchmark adds a state-restore run.

Port-fidelity gate (Bhaskera CUDA state-restore vs upstream CUDA, <= 0.01): PASS for q8_0 and q4_k_m (bit-identical: max 0.0, 0 flips).

## Batching fidelity

Batched vs unbatched Bhaskera (CUDA, q8_0, seq-copy), 1100 answers, gate 0.01.

| run | max | mean | flips | result |
|---|---|---|---|---|
| batched, concurrency 16 | 0.0517 | 0.0029 | 4 | FAIL |
| batched, concurrency 1 (diagnostic) | 0.0 | 0.0 | 0 | bit-identical |

At concurrency 1 every batch holds one request and the output equals the unbatched run exactly, so the wiring
(`serve.batch`, `decide_many`, `max_requests`) is correct. The c=16 drift comes from batch composition (requests
decoded together in one CUDA batch), not from the deployment code. Worst c=16 rows:

| id | question | unbatched | batched | delta |
|---|---|---|---|---|
| p275 | q4 | 0.3811 | 0.3294 | 0.0517 |
| p179 | q7 | 0.6227 | 0.5752 | 0.0475 |
| p050 | q0 | 0.6991 | 0.6540 | 0.0451 |
| p166 | q1 | 0.4610 | 0.4184 | 0.0426 |
| p089 | q1 | 0.3971 | 0.4367 | 0.0396 |

Batching stays off pending a decision on the tolerance.
