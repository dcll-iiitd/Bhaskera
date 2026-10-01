# Decision serving, Phase A (one RTX A6000, GPU 0)

Date 2026-09-30. Host: AMD EPYC 7282, RTX A6000 (GPU 0 only), llama.cpp release b11081 (pinned). Closed-loop load, 15 s per level after 3 s warmup, concurrency 1/4/16/64. 396 rows (360 Phase A + 36 batched), 0 errors.

**BEST_REPLICAS = 8**, decided by `bench_long_q3` at concurrency 64: r1 18.7, r2 38.4, r4 74.9, r8 89.6 req/s (r8 highest; ties would go to fewer replicas, none tied).

**Data caveats.** req/s is completed requests divided by the nominal window (15 s), so very slow configs are biased, and a request that outlasts the window is never counted. Upstream CPU rows at concurrency 64 with completed=0 (req/s 0.0, latencies nan; every request outlasted the window): bench_document_q1, bench_document_q3, bench_document_q10, bench_long_q10. Low-sample upstream CPU rows (fewer than 100 completed): bench_long_q3 c=64 (27 completed), bench_short_q10 c=64 (25), bench_long_q1 c=64 (52), bench_short_q3 c=64 (52), and all other document/long_q10 rows (5 to 18 completed). Treat these percentiles and rates as indicative only. The `completed` column is shown in every table. The earlier 10.2x figure (bench_long_q3 c=64) came from the 27-completion row and is withdrawn.

Observations (all at concurrency 64 unless noted):
- GPU vs CPU (upstream q8_0): 6.6x on bench_short_q1 at concurrency 64 (67.5 vs 10.3 req/s; 1012 vs 154 completed). This is the only request set where both sides have at least 100 completed requests (it holds at every level: 5.4x at c=1, 6.3x at c=4, 6.8x at c=16). At concurrency 1 on bench_short_q1 p50 is 17.2 ms (CUDA) vs 92.8 ms (CPU). Longer sets are far slower on CPU (e.g. bench_long_q1 c=64: 800 vs 52 completed) but rest on fewer than 100 CPU completions, so no speedup is quoted from them.
- Replica scaling (bench_long_q3): 18.7 -> 38.4 -> 74.9 -> 89.6 req/s for 1/2/4/8, near-linear to 4 and a knee at 8 (+20%). On bench_short_q1 the knee is at 4: 168.7 req/s (r4) vs 166.3 (r8).
- Bhaskera r1 matches upstream CUDA on long prompts (18.7 vs 18.3 req/s) but is slower on bench_short_q1 (49.6 vs 67.5 req/s); single-request p50 is 23.5 ms vs 17.2 ms.
- q4_k_m vs q8_0 at r8: bench_long_q3 91.5 vs 89.6 req/s, bench_short_q1 178.8 vs 166.3 req/s (small gain, +2% to +8%).
- State-restore branch at r8 vs seq-copy: bench_long_q3 52.1 vs 89.6 req/s (slower), bench_short_q1 188.8 vs 166.3 req/s (faster).
- Gateway overhead at r8 (p50): bench_short_q1 +3.3 ms at concurrency 1 (29.1 vs 25.8), +39 ms at concurrency 64 (360 vs 321); bench_long_q3 at concurrency 64 indistinguishable (704 vs 710 ms).

**Phase B (batching, r8, 36 rows `bhaskera-r8-batched`).** Decision rule: batching becomes the default only if `bhaskera-r8-batched` has higher req_per_s than `bhaskera-r8-q8_0` at concurrency 64 on at least 6 of the 9 request sets and its p90_ms is not worse on those sets. Result: it wins on 1 of 9 (bench_short_q1 only), so **batching stays off** (`serve.decision.batching.enabled` unchanged). Peak GPU memory (max_memory_mib) with batching was 17857 MiB (about 17.8 GB, 17.4 GiB) versus 9793 MiB (about 9.7 GB) unbatched at r8. Concurrency 64 per set:

| request set | unbatched req/s | batched req/s | unbatched p90 ms | batched p90 ms | wins |
|---|---|---|---|---|---|
| bench_document_q1 | 26.4 | 25.9 | 2513 | 2553 | no |
| bench_document_q10 | 18.1 | 18.4 | 3924 | 3933 | no |
| bench_document_q3 | 24.1 | 23.1 | 2876 | 3062 | no |
| bench_long_q1 | 141.7 | 119.1 | 661 | 572 | no |
| bench_long_q10 | 46.1 | 41.0 | 1491 | 1704 | no |
| bench_long_q3 | 89.6 | 79.9 | 919 | 847 | no |
| bench_short_q1 | 166.3 | 171.6 | 587 | 409 | yes |
| bench_short_q10 | 53.1 | 49.6 | 1267 | 1372 | no |
| bench_short_q3 | 115.9 | 99.9 | 801 | 680 | no |

Count: 1 of 9 (needs 6). Batching drift vs unbatched (reported, not gated): at concurrency 16 max |dP(yes)| 0.0517, mean 0.0029, 4 of 1100 decisions flipped; at concurrency 1 bit-identical.

| bhaskera-r8-batched | bench_document_q1 | 1 | 184 | 12.3 | 12.3 | 81.2 | 83.4 | 87.3 | 0 | 17797.0 | 43.4 |
| bhaskera-r8-batched | bench_document_q1 | 4 | 382 | 25.5 | 25.5 | 156.4 | 185.0 | 259.0 | 0 | 17805.0 | 92.6 |
| bhaskera-r8-batched | bench_document_q1 | 16 | 387 | 25.8 | 25.8 | 612.0 | 864.8 | 1019.1 | 0 | 17813.0 | 99.6 |
| bhaskera-r8-batched | bench_document_q1 | 64 | 388 | 25.9 | 25.9 | 2494.4 | 2552.5 | 2611.7 | 0 | 17817.0 | 100.0 |
| bhaskera-r8-gateway | bench_document_q1 | 1 | 191 | 12.7 | 12.7 | 78.3 | 82.5 | 89.1 | 0 | 9729.0 | 47.8 |
| bhaskera-r8-gateway | bench_document_q1 | 4 | 396 | 26.4 | 26.4 | 151.6 | 173.6 | 236.2 | 0 | 9737.0 | 95.2 |
| bhaskera-r8-gateway | bench_document_q1 | 16 | 398 | 26.5 | 26.5 | 595.6 | 854.4 | 1052.7 | 0 | 9745.0 | 99.9 |
| bhaskera-r8-gateway | bench_document_q1 | 64 | 395 | 26.3 | 26.3 | 2431.0 | 2626.1 | 2761.4 | 0 | 9749.0 | 100.0 |
| bhaskera-r8-q4_k_m | bench_document_q1 | 1 | 198 | 13.2 | 13.2 | 76.0 | 79.5 | 83.7 | 0 | 8053.0 | 43.5 |
| bhaskera-r8-q4_k_m | bench_document_q1 | 4 | 380 | 25.3 | 25.3 | 157.3 | 182.9 | 259.3 | 0 | 8065.0 | 92.2 |
| bhaskera-r8-q4_k_m | bench_document_q1 | 16 | 382 | 25.5 | 25.5 | 610.8 | 891.6 | 1143.7 | 0 | 8065.0 | 100.0 |
| bhaskera-r8-q4_k_m | bench_document_q1 | 64 | 383 | 25.5 | 25.5 | 2498.2 | 2573.6 | 2637.1 | 0 | 8073.0 | 97.3 |
| bhaskera-r8-q8_0 | bench_document_q1 | 1 | 202 | 13.5 | 13.5 | 74.0 | 77.2 | 81.4 | 0 | 9733.0 | 45.5 |
| bhaskera-r8-q8_0 | bench_document_q1 | 4 | 397 | 26.5 | 26.5 | 153.4 | 173.8 | 249.7 | 0 | 9757.0 | 95.1 |
| bhaskera-r8-q8_0 | bench_document_q1 | 16 | 400 | 26.7 | 26.7 | 597.8 | 857.0 | 1020.0 | 0 | 9757.0 | 99.9 |
| bhaskera-r8-q8_0 | bench_document_q1 | 64 | 396 | 26.4 | 26.4 | 2429.6 | 2512.5 | 2705.8 | 0 | 9761.0 | 100.0 |
| bhaskera-r8-staterestore | bench_document_q1 | 1 | 203 | 13.5 | 13.5 | 73.7 | 78.0 | 80.6 | 0 | 9761.0 | 47.2 |
| bhaskera-r8-staterestore | bench_document_q1 | 4 | 399 | 26.6 | 26.6 | 151.9 | 173.3 | 264.2 | 0 | 9761.0 | 94.2 |
| bhaskera-r8-staterestore | bench_document_q1 | 16 | 399 | 26.6 | 26.6 | 583.1 | 857.1 | 1062.4 | 0 | 9761.0 | 100.0 |
| bhaskera-r8-staterestore | bench_document_q1 | 64 | 399 | 26.6 | 26.6 | 2408.4 | 2467.6 | 2544.4 | 0 | 9761.0 | 97.7 |
| upstream-cpu-q8_0 | bench_document_q1 | 1 | 9 | 0.6 | 0.6 | 1556.2 | 1867.9 | 1867.9 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q1 | 4 | 9 | 0.6 | 0.6 | 7502.9 | 7575.9 | 7575.9 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q1 | 16 | 8 | 0.5 | 0.5 | 26035.1 | 27519.1 | 27519.1 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q1 | 64 | 0 | 0.0 | 0.0 | nan | nan | nan | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_document_q1 | 1 | 234 | 15.6 | 15.6 | 63.9 | 65.2 | 82.7 | 0 | 1014.0 | 52.2 |
| upstream-cuda-q4_k_m | bench_document_q1 | 4 | 248 | 16.5 | 16.5 | 242.4 | 246.1 | 263.2 | 0 | 1014.0 | 56.6 |
| upstream-cuda-q4_k_m | bench_document_q1 | 16 | 246 | 16.4 | 16.4 | 972.0 | 994.2 | 996.5 | 0 | 1014.0 | 57.2 |
| upstream-cuda-q4_k_m | bench_document_q1 | 64 | 256 | 17.1 | 17.1 | 3685.4 | 4517.4 | 4576.4 | 0 | 1014.0 | 57.9 |
| upstream-cuda-q8_0 | bench_document_q1 | 1 | 252 | 16.8 | 16.8 | 62.1 | 63.5 | 72.1 | 0 | 1224.0 | 57.0 |
| upstream-cuda-q8_0 | bench_document_q1 | 4 | 264 | 17.6 | 17.6 | 227.4 | 238.3 | 262.7 | 0 | 1224.0 | 57.0 |
| upstream-cuda-q8_0 | bench_document_q1 | 16 | 262 | 17.5 | 17.5 | 912.2 | 933.5 | 944.1 | 0 | 1224.0 | 60.5 |
| upstream-cuda-q8_0 | bench_document_q1 | 64 | 276 | 18.4 | 18.4 | 3583.8 | 4366.2 | 4385.5 | 0 | 1224.0 | 56.5 |

## bench_document_q10

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_document_q10 | 1 | 93 | 6.2 | 62.0 | 161.8 | 164.9 | 183.2 | 0 | 1220.0 | 31.7 |
| bhaskera-r1-q8_0 | bench_document_q10 | 4 | 81 | 5.4 | 54.0 | 747.7 | 753.6 | 768.6 | 0 | 1220.0 | 25.5 |
| bhaskera-r1-q8_0 | bench_document_q10 | 16 | 75 | 5.0 | 50.0 | 3221.4 | 3338.3 | 3562.0 | 0 | 1220.0 | 17.5 |
| bhaskera-r1-q8_0 | bench_document_q10 | 64 | 70 | 4.7 | 46.7 | 13760.1 | 13909.5 | 14089.0 | 0 | 1220.0 | 22.6 |
| bhaskera-r2-q8_0 | bench_document_q10 | 1 | 89 | 5.9 | 59.3 | 164.0 | 185.6 | 190.8 | 0 | 2435.0 | 22.8 |
| bhaskera-r2-q8_0 | bench_document_q10 | 4 | 151 | 10.1 | 100.7 | 376.1 | 457.0 | 465.3 | 0 | 2435.0 | 43.4 |
| bhaskera-r2-q8_0 | bench_document_q10 | 16 | 140 | 9.3 | 93.3 | 1722.4 | 2005.7 | 2302.0 | 0 | 2435.0 | 42.6 |
| bhaskera-r2-q8_0 | bench_document_q10 | 64 | 132 | 8.8 | 88.0 | 8027.9 | 8437.4 | 8772.0 | 0 | 2435.0 | 29.2 |
| bhaskera-r4-q8_0 | bench_document_q10 | 1 | 87 | 5.8 | 58.0 | 167.4 | 189.1 | 201.4 | 0 | 4866.0 | 20.3 |
| bhaskera-r4-q8_0 | bench_document_q10 | 4 | 204 | 13.6 | 136.0 | 263.1 | 478.8 | 540.7 | 0 | 4866.0 | 70.4 |
| bhaskera-r4-q8_0 | bench_document_q10 | 16 | 223 | 14.9 | 148.7 | 1127.0 | 1225.8 | 1255.9 | 0 | 4866.0 | 74.6 |
| bhaskera-r4-q8_0 | bench_document_q10 | 64 | 208 | 13.9 | 138.7 | 4731.5 | 5082.3 | 5355.3 | 0 | 4866.0 | 68.4 |
| bhaskera-r8-batched | bench_document_q10 | 1 | 83 | 5.5 | 55.3 | 177.2 | 199.9 | 221.3 | 0 | 17793.0 | 24.5 |
| bhaskera-r8-batched | bench_document_q10 | 4 | 211 | 14.1 | 140.7 | 290.0 | 323.0 | 509.4 | 0 | 17793.0 | 67.0 |
| bhaskera-r8-batched | bench_document_q10 | 16 | 282 | 18.8 | 188.0 | 873.2 | 1228.9 | 1442.3 | 0 | 17793.0 | 95.8 |
| bhaskera-r8-batched | bench_document_q10 | 64 | 276 | 18.4 | 184.0 | 3497.0 | 3933.0 | 4009.1 | 0 | 17797.0 | 99.6 |
| bhaskera-r8-gateway | bench_document_q10 | 1 | 86 | 5.7 | 57.3 | 171.6 | 190.1 | 199.1 | 0 | 9729.0 | 30.4 |
| bhaskera-r8-gateway | bench_document_q10 | 4 | 214 | 14.3 | 142.7 | 279.8 | 328.3 | 544.4 | 0 | 9729.0 | 67.6 |
| bhaskera-r8-gateway | bench_document_q10 | 16 | 262 | 17.5 | 174.7 | 932.4 | 1371.2 | 1537.2 | 0 | 9729.0 | 92.6 |
| bhaskera-r8-gateway | bench_document_q10 | 64 | 274 | 18.3 | 182.7 | 3521.1 | 3952.8 | 4047.6 | 0 | 9729.0 | 94.8 |
| bhaskera-r8-q4_k_m | bench_document_q10 | 1 | 86 | 5.7 | 57.3 | 168.1 | 189.4 | 198.1 | 0 | 8049.0 | 32.4 |
| bhaskera-r8-q4_k_m | bench_document_q10 | 4 | 205 | 13.7 | 136.7 | 290.6 | 339.5 | 577.6 | 0 | 8049.0 | 68.1 |
| bhaskera-r8-q4_k_m | bench_document_q10 | 16 | 259 | 17.3 | 172.7 | 944.2 | 1407.0 | 1521.7 | 0 | 8049.0 | 75.8 |
| bhaskera-r8-q4_k_m | bench_document_q10 | 64 | 267 | 17.8 | 178.0 | 3573.5 | 4050.4 | 4251.6 | 0 | 8049.0 | 89.1 |
| bhaskera-r8-q8_0 | bench_document_q10 | 1 | 89 | 5.9 | 59.3 | 165.1 | 186.5 | 191.3 | 0 | 9729.0 | 30.8 |
| bhaskera-r8-q8_0 | bench_document_q10 | 4 | 219 | 14.6 | 146.0 | 279.8 | 323.4 | 491.5 | 0 | 9729.0 | 71.6 |
| bhaskera-r8-q8_0 | bench_document_q10 | 16 | 267 | 17.8 | 178.0 | 938.0 | 1396.0 | 1866.1 | 0 | 9729.0 | 89.2 |
| bhaskera-r8-q8_0 | bench_document_q10 | 64 | 271 | 18.1 | 180.7 | 3496.0 | 3923.9 | 3988.0 | 0 | 9729.0 | 99.1 |
| bhaskera-r8-staterestore | bench_document_q10 | 1 | 59 | 3.9 | 39.3 | 252.3 | 270.6 | 278.8 | 0 | 9761.0 | 51.2 |
| bhaskera-r8-staterestore | bench_document_q10 | 4 | 151 | 10.1 | 100.7 | 388.8 | 427.2 | 739.8 | 0 | 9761.0 | 69.9 |
| bhaskera-r8-staterestore | bench_document_q10 | 16 | 191 | 12.7 | 127.3 | 1264.3 | 1852.5 | 2055.0 | 0 | 9761.0 | 97.1 |
| bhaskera-r8-staterestore | bench_document_q10 | 64 | 192 | 12.8 | 128.0 | 4916.5 | 5417.8 | 5956.2 | 0 | 9761.0 | 98.4 |
| upstream-cpu-q8_0 | bench_document_q10 | 1 | 6 | 0.4 | 4.0 | 2872.6 | 2917.1 | 2917.1 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q10 | 4 | 5 | 0.3 | 3.3 | 11270.8 | 11347.4 | 11347.4 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q10 | 16 | 5 | 0.3 | 3.3 | 43139.3 | 43583.8 | 43583.8 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q10 | 64 | 0 | 0.0 | 0.0 | nan | nan | nan | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_document_q10 | 1 | 63 | 4.2 | 42.0 | 240.5 | 256.2 | 270.0 | 0 | 1014.0 | 37.2 |
| upstream-cuda-q4_k_m | bench_document_q10 | 4 | 61 | 4.1 | 40.7 | 985.7 | 998.5 | 1009.2 | 0 | 1014.0 | 32.7 |
| upstream-cuda-q4_k_m | bench_document_q10 | 16 | 61 | 4.1 | 40.7 | 3899.5 | 3938.4 | 3960.3 | 0 | 1014.0 | 35.3 |
| upstream-cuda-q4_k_m | bench_document_q10 | 64 | 56 | 3.7 | 37.3 | 13339.3 | 15063.8 | 15463.4 | 0 | 1014.0 | 41.2 |
| upstream-cuda-q8_0 | bench_document_q10 | 1 | 60 | 4.0 | 40.0 | 251.5 | 255.7 | 273.9 | 0 | 1224.0 | 19.2 |
| upstream-cuda-q8_0 | bench_document_q10 | 4 | 60 | 4.0 | 40.0 | 1012.2 | 1028.4 | 1043.9 | 0 | 1224.0 | 35.1 |
| upstream-cuda-q8_0 | bench_document_q10 | 16 | 60 | 4.0 | 40.0 | 4028.1 | 4045.5 | 4057.2 | 0 | 1224.0 | 19.8 |
| upstream-cuda-q8_0 | bench_document_q10 | 64 | 53 | 3.5 | 35.3 | 13210.0 | 15428.6 | 15912.0 | 0 | 1224.0 | 46.7 |

## bench_document_q3

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_document_q3 | 1 | 130 | 8.7 | 26.0 | 116.0 | 118.7 | 120.8 | 0 | 1224.0 | 34.9 |
| bhaskera-r1-q8_0 | bench_document_q3 | 4 | 149 | 9.9 | 29.8 | 401.7 | 423.1 | 430.4 | 0 | 1224.0 | 40.5 |
| bhaskera-r1-q8_0 | bench_document_q3 | 16 | 137 | 9.1 | 27.4 | 1819.5 | 1992.0 | 2075.6 | 0 | 1224.0 | 35.8 |
| bhaskera-r1-q8_0 | bench_document_q3 | 64 | 127 | 8.5 | 25.4 | 7828.4 | 8346.8 | 8486.8 | 0 | 1224.0 | 33.0 |
| bhaskera-r2-q8_0 | bench_document_q3 | 1 | 131 | 8.7 | 26.2 | 115.7 | 119.3 | 123.7 | 0 | 2443.0 | 34.8 |
| bhaskera-r2-q8_0 | bench_document_q3 | 4 | 268 | 17.9 | 53.6 | 213.4 | 267.5 | 288.5 | 0 | 2443.0 | 64.1 |
| bhaskera-r2-q8_0 | bench_document_q3 | 16 | 250 | 16.7 | 50.0 | 975.1 | 1089.0 | 1241.4 | 0 | 2443.0 | 59.0 |
| bhaskera-r2-q8_0 | bench_document_q3 | 64 | 239 | 15.9 | 47.8 | 4905.7 | 5422.9 | 5791.6 | 0 | 2443.0 | 56.5 |
| bhaskera-r4-q8_0 | bench_document_q3 | 1 | 128 | 8.5 | 25.6 | 118.1 | 122.1 | 125.4 | 0 | 4882.0 | 35.6 |
| bhaskera-r4-q8_0 | bench_document_q3 | 4 | 295 | 19.7 | 59.0 | 188.7 | 299.9 | 351.2 | 0 | 4882.0 | 81.3 |
| bhaskera-r4-q8_0 | bench_document_q3 | 16 | 347 | 23.1 | 69.4 | 676.5 | 794.9 | 849.4 | 0 | 4882.0 | 89.3 |
| bhaskera-r4-q8_0 | bench_document_q3 | 64 | 345 | 23.0 | 69.0 | 2871.1 | 3091.3 | 3258.4 | 0 | 4882.0 | 94.6 |
| bhaskera-r8-batched | bench_document_q3 | 1 | 121 | 8.1 | 24.2 | 124.7 | 128.7 | 135.3 | 0 | 17821.0 | 37.4 |
| bhaskera-r8-batched | bench_document_q3 | 4 | 297 | 19.8 | 59.4 | 209.1 | 235.3 | 367.0 | 0 | 17825.0 | 78.6 |
| bhaskera-r8-batched | bench_document_q3 | 16 | 346 | 23.1 | 69.2 | 695.9 | 980.5 | 1104.2 | 0 | 17825.0 | 99.9 |
| bhaskera-r8-batched | bench_document_q3 | 64 | 347 | 23.1 | 69.4 | 2748.5 | 3062.1 | 3104.3 | 0 | 17825.0 | 97.9 |
| bhaskera-r8-gateway | bench_document_q3 | 1 | 123 | 8.2 | 24.6 | 121.5 | 127.3 | 132.9 | 0 | 9761.0 | 32.5 |
| bhaskera-r8-gateway | bench_document_q3 | 4 | 309 | 20.6 | 61.8 | 194.9 | 234.2 | 354.3 | 0 | 9761.0 | 76.0 |
| bhaskera-r8-gateway | bench_document_q3 | 16 | 360 | 24.0 | 72.0 | 675.2 | 1026.4 | 1212.2 | 0 | 9761.0 | 97.6 |
| bhaskera-r8-gateway | bench_document_q3 | 64 | 357 | 23.8 | 71.4 | 2671.1 | 2850.3 | 3001.3 | 0 | 9761.0 | 100.0 |
| bhaskera-r8-q4_k_m | bench_document_q3 | 1 | 127 | 8.5 | 25.4 | 119.1 | 123.1 | 125.9 | 0 | 8081.0 | 35.9 |
| bhaskera-r8-q4_k_m | bench_document_q3 | 4 | 298 | 19.9 | 59.6 | 203.2 | 237.8 | 383.6 | 0 | 8081.0 | 85.2 |
| bhaskera-r8-q4_k_m | bench_document_q3 | 16 | 356 | 23.7 | 71.2 | 696.4 | 992.0 | 1131.5 | 0 | 8081.0 | 99.7 |
| bhaskera-r8-q4_k_m | bench_document_q3 | 64 | 353 | 23.5 | 70.6 | 2733.3 | 2920.4 | 3057.5 | 0 | 8081.0 | 99.7 |
| bhaskera-r8-q8_0 | bench_document_q3 | 1 | 130 | 8.7 | 26.0 | 117.1 | 120.1 | 124.5 | 0 | 9761.0 | 30.8 |
| bhaskera-r8-q8_0 | bench_document_q3 | 4 | 311 | 20.7 | 62.2 | 194.4 | 230.6 | 357.0 | 0 | 9761.0 | 83.5 |
| bhaskera-r8-q8_0 | bench_document_q3 | 16 | 356 | 23.7 | 71.2 | 699.9 | 974.9 | 1139.5 | 0 | 9761.0 | 100.0 |
| bhaskera-r8-q8_0 | bench_document_q3 | 64 | 361 | 24.1 | 72.2 | 2656.9 | 2876.2 | 2957.6 | 0 | 9761.0 | 94.8 |
| bhaskera-r8-staterestore | bench_document_q3 | 1 | 108 | 7.2 | 21.6 | 139.5 | 143.4 | 163.0 | 0 | 9761.0 | 43.2 |
| bhaskera-r8-staterestore | bench_document_q3 | 4 | 252 | 16.8 | 50.4 | 241.1 | 260.3 | 421.4 | 0 | 9761.0 | 78.5 |
| bhaskera-r8-staterestore | bench_document_q3 | 16 | 312 | 20.8 | 62.4 | 791.0 | 1131.6 | 1272.4 | 0 | 9761.0 | 98.9 |
| bhaskera-r8-staterestore | bench_document_q3 | 64 | 309 | 20.6 | 61.8 | 3110.3 | 3301.3 | 3386.4 | 0 | 9761.0 | 99.9 |
| upstream-cpu-q8_0 | bench_document_q3 | 1 | 7 | 0.5 | 1.4 | 2120.8 | 2147.3 | 2147.3 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q3 | 4 | 7 | 0.5 | 1.4 | 8475.6 | 8542.8 | 8542.8 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q3 | 16 | 7 | 0.5 | 1.4 | 34008.9 | 34102.4 | 34102.4 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_document_q3 | 64 | 0 | 0.0 | 0.0 | nan | nan | nan | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_document_q3 | 1 | 119 | 7.9 | 23.8 | 125.5 | 127.3 | 143.9 | 0 | 1014.0 | 39.6 |
| upstream-cuda-q4_k_m | bench_document_q3 | 4 | 122 | 8.1 | 24.4 | 492.5 | 502.6 | 513.9 | 0 | 1014.0 | 33.7 |
| upstream-cuda-q4_k_m | bench_document_q3 | 16 | 125 | 8.3 | 25.0 | 1905.0 | 1928.7 | 1938.2 | 0 | 1014.0 | 40.6 |
| upstream-cuda-q4_k_m | bench_document_q3 | 64 | 150 | 10.0 | 30.0 | 6439.5 | 8889.4 | 8940.0 | 0 | 1014.0 | 37.3 |
| upstream-cuda-q8_0 | bench_document_q3 | 1 | 127 | 8.5 | 25.4 | 125.5 | 128.2 | 149.7 | 0 | 1224.0 | 43.5 |
| upstream-cuda-q8_0 | bench_document_q3 | 4 | 128 | 8.5 | 25.6 | 472.6 | 479.7 | 493.0 | 0 | 1224.0 | 39.6 |
| upstream-cuda-q8_0 | bench_document_q3 | 16 | 122 | 8.1 | 24.4 | 1968.2 | 2014.0 | 2030.3 | 0 | 1224.0 | 40.3 |
| upstream-cuda-q8_0 | bench_document_q3 | 64 | 140 | 9.3 | 28.0 | 7424.9 | 8114.5 | 9285.3 | 0 | 1224.0 | 39.9 |

## bench_long_q1

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_long_q1 | 1 | 519 | 34.6 | 34.6 | 28.7 | 30.4 | 32.2 | 0 | 1228.0 | 21.6 |
| bhaskera-r1-q8_0 | bench_long_q1 | 4 | 746 | 49.7 | 49.7 | 80.2 | 83.7 | 86.2 | 0 | 1228.0 | 31.0 |
| bhaskera-r1-q8_0 | bench_long_q1 | 16 | 651 | 43.4 | 43.4 | 404.3 | 461.3 | 520.9 | 0 | 1228.0 | 29.1 |
| bhaskera-r1-q8_0 | bench_long_q1 | 64 | 629 | 41.9 | 41.9 | 1541.3 | 2650.2 | 2870.1 | 0 | 1228.0 | 27.4 |
| bhaskera-r2-q8_0 | bench_long_q1 | 1 | 523 | 34.9 | 34.9 | 28.5 | 29.8 | 31.9 | 0 | 2451.0 | 22.5 |
| bhaskera-r2-q8_0 | bench_long_q1 | 4 | 1495 | 99.7 | 99.7 | 39.9 | 42.2 | 45.5 | 0 | 2451.0 | 62.9 |
| bhaskera-r2-q8_0 | bench_long_q1 | 16 | 1388 | 92.5 | 92.5 | 179.8 | 233.6 | 279.0 | 0 | 2451.0 | 58.1 |
| bhaskera-r2-q8_0 | bench_long_q1 | 64 | 1156 | 77.1 | 77.1 | 824.2 | 1415.6 | 1771.4 | 0 | 2451.0 | 50.2 |
| bhaskera-r4-q8_0 | bench_long_q1 | 1 | 511 | 34.1 | 34.1 | 29.3 | 30.8 | 33.6 | 0 | 4898.0 | 24.0 |
| bhaskera-r4-q8_0 | bench_long_q1 | 4 | 1708 | 113.9 | 113.9 | 33.3 | 45.1 | 55.4 | 0 | 4898.0 | 74.0 |
| bhaskera-r4-q8_0 | bench_long_q1 | 16 | 2142 | 142.8 | 142.8 | 112.2 | 121.1 | 129.9 | 0 | 4898.0 | 97.8 |
| bhaskera-r4-q8_0 | bench_long_q1 | 64 | 2089 | 139.3 | 139.3 | 412.3 | 746.8 | 1018.9 | 0 | 4898.0 | 96.4 |
| bhaskera-r8-batched | bench_long_q1 | 1 | 395 | 26.3 | 26.3 | 38.2 | 39.8 | 43.7 | 0 | 17857.0 | 21.7 |
| bhaskera-r8-batched | bench_long_q1 | 4 | 1424 | 94.9 | 94.9 | 41.2 | 46.9 | 60.8 | 0 | 17857.0 | 72.8 |
| bhaskera-r8-batched | bench_long_q1 | 16 | 1841 | 122.7 | 122.7 | 128.3 | 178.7 | 213.2 | 0 | 17857.0 | 98.8 |
| bhaskera-r8-batched | bench_long_q1 | 64 | 1787 | 119.1 | 119.1 | 542.4 | 571.6 | 597.6 | 0 | 17857.0 | 100.0 |
| bhaskera-r8-gateway | bench_long_q1 | 1 | 445 | 29.7 | 29.7 | 33.7 | 36.0 | 38.5 | 0 | 9793.0 | 21.2 |
| bhaskera-r8-gateway | bench_long_q1 | 4 | 1642 | 109.5 | 109.5 | 35.8 | 40.0 | 52.3 | 0 | 9793.0 | 71.1 |
| bhaskera-r8-gateway | bench_long_q1 | 16 | 2124 | 141.6 | 141.6 | 111.1 | 155.1 | 187.5 | 0 | 9793.0 | 99.0 |
| bhaskera-r8-gateway | bench_long_q1 | 64 | 2123 | 141.5 | 141.5 | 407.7 | 673.7 | 939.3 | 0 | 9793.0 | 100.0 |
| bhaskera-r8-q4_k_m | bench_long_q1 | 1 | 489 | 32.6 | 32.6 | 30.6 | 32.4 | 36.1 | 0 | 8113.0 | 24.2 |
| bhaskera-r8-q4_k_m | bench_long_q1 | 4 | 1751 | 116.7 | 116.7 | 33.3 | 38.4 | 52.4 | 0 | 8113.0 | 73.4 |
| bhaskera-r8-q4_k_m | bench_long_q1 | 16 | 2157 | 143.8 | 143.8 | 109.9 | 152.3 | 182.6 | 0 | 8113.0 | 98.3 |
| bhaskera-r8-q4_k_m | bench_long_q1 | 64 | 2146 | 143.1 | 143.1 | 402.9 | 665.6 | 975.8 | 0 | 8113.0 | 100.0 |
| bhaskera-r8-q8_0 | bench_long_q1 | 1 | 489 | 32.6 | 32.6 | 30.7 | 32.4 | 35.6 | 0 | 9793.0 | 24.0 |
| bhaskera-r8-q8_0 | bench_long_q1 | 4 | 1771 | 118.1 | 118.1 | 33.0 | 37.6 | 52.4 | 0 | 9793.0 | 76.2 |
| bhaskera-r8-q8_0 | bench_long_q1 | 16 | 2131 | 142.1 | 142.1 | 111.1 | 153.6 | 183.9 | 0 | 9793.0 | 99.2 |
| bhaskera-r8-q8_0 | bench_long_q1 | 64 | 2125 | 141.7 | 141.7 | 422.0 | 661.3 | 902.7 | 0 | 9793.0 | 99.9 |
| bhaskera-r8-staterestore | bench_long_q1 | 1 | 491 | 32.7 | 32.7 | 30.7 | 32.4 | 33.7 | 0 | 9793.0 | 23.4 |
| bhaskera-r8-staterestore | bench_long_q1 | 4 | 1746 | 116.4 | 116.4 | 33.4 | 38.5 | 52.1 | 0 | 9793.0 | 72.8 |
| bhaskera-r8-staterestore | bench_long_q1 | 16 | 2137 | 142.5 | 142.5 | 110.5 | 153.9 | 188.8 | 0 | 9793.0 | 98.1 |
| bhaskera-r8-staterestore | bench_long_q1 | 64 | 2122 | 141.5 | 141.5 | 412.1 | 680.5 | 982.4 | 0 | 9793.0 | 100.0 |
| upstream-cpu-q8_0 | bench_long_q1 | 1 | 60 | 4.0 | 4.0 | 253.7 | 265.2 | 269.9 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q1 | 4 | 60 | 4.0 | 4.0 | 1003.1 | 1027.8 | 1035.8 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q1 | 16 | 60 | 4.0 | 4.0 | 4014.7 | 4055.8 | 4095.2 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q1 | 64 | 52 | 3.5 | 3.5 | 13117.4 | 15574.6 | 15815.9 | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_long_q1 | 1 | 703 | 46.9 | 46.9 | 21.4 | 22.3 | 24.5 | 0 | 1018.0 | 30.8 |
| upstream-cuda-q4_k_m | bench_long_q1 | 4 | 817 | 54.5 | 54.5 | 73.9 | 77.7 | 79.7 | 0 | 1018.0 | 35.4 |
| upstream-cuda-q4_k_m | bench_long_q1 | 16 | 798 | 53.2 | 53.2 | 301.2 | 305.6 | 308.9 | 0 | 1018.0 | 34.1 |
| upstream-cuda-q4_k_m | bench_long_q1 | 64 | 800 | 53.3 | 53.3 | 1255.4 | 1382.9 | 1416.5 | 0 | 1018.0 | 32.3 |
| upstream-cuda-q8_0 | bench_long_q1 | 1 | 685 | 45.7 | 45.7 | 21.7 | 22.9 | 24.8 | 0 | 1228.0 | 30.9 |
| upstream-cuda-q8_0 | bench_long_q1 | 4 | 828 | 55.2 | 55.2 | 74.1 | 78.3 | 79.8 | 0 | 1228.0 | 36.6 |
| upstream-cuda-q8_0 | bench_long_q1 | 16 | 802 | 53.5 | 53.5 | 300.6 | 306.8 | 328.7 | 0 | 1228.0 | 34.6 |
| upstream-cuda-q8_0 | bench_long_q1 | 64 | 800 | 53.3 | 53.3 | 1182.1 | 1372.2 | 1430.4 | 0 | 1228.0 | 34.7 |

## bench_long_q10

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_long_q10 | 1 | 136 | 9.1 | 90.7 | 106.8 | 120.2 | 123.5 | 0 | 1228.0 | 16.4 |
| bhaskera-r1-q8_0 | bench_long_q10 | 4 | 137 | 9.1 | 91.3 | 435.5 | 445.6 | 451.1 | 0 | 1228.0 | 17.6 |
| bhaskera-r1-q8_0 | bench_long_q10 | 16 | 129 | 8.6 | 86.0 | 1871.6 | 1994.0 | 2120.0 | 0 | 1228.0 | 17.1 |
| bhaskera-r1-q8_0 | bench_long_q10 | 64 | 110 | 7.3 | 73.3 | 8687.1 | 8861.5 | 8984.4 | 0 | 1228.0 | 14.8 |
| bhaskera-r2-q8_0 | bench_long_q10 | 1 | 134 | 8.9 | 89.3 | 110.3 | 121.5 | 128.8 | 0 | 2451.0 | 18.0 |
| bhaskera-r2-q8_0 | bench_long_q10 | 4 | 272 | 18.1 | 181.3 | 218.7 | 241.8 | 250.2 | 0 | 2451.0 | 34.0 |
| bhaskera-r2-q8_0 | bench_long_q10 | 16 | 251 | 16.7 | 167.3 | 960.6 | 1080.7 | 1219.6 | 0 | 2451.0 | 32.0 |
| bhaskera-r2-q8_0 | bench_long_q10 | 64 | 223 | 14.9 | 148.7 | 4707.4 | 5212.4 | 5424.1 | 0 | 2451.0 | 30.3 |
| bhaskera-r4-q8_0 | bench_long_q10 | 1 | 128 | 8.5 | 85.3 | 119.4 | 123.1 | 127.3 | 0 | 4894.0 | 18.3 |
| bhaskera-r4-q8_0 | bench_long_q10 | 4 | 388 | 25.9 | 258.7 | 137.9 | 231.6 | 275.3 | 0 | 4898.0 | 50.7 |
| bhaskera-r4-q8_0 | bench_long_q10 | 16 | 499 | 33.3 | 332.7 | 479.8 | 523.8 | 555.4 | 0 | 4898.0 | 65.8 |
| bhaskera-r4-q8_0 | bench_long_q10 | 64 | 483 | 32.2 | 322.0 | 2023.4 | 2124.0 | 2195.1 | 0 | 4898.0 | 66.4 |
| bhaskera-r8-batched | bench_long_q10 | 1 | 131 | 8.7 | 87.3 | 114.8 | 124.3 | 132.0 | 0 | 17849.0 | 21.0 |
| bhaskera-r8-batched | bench_long_q10 | 4 | 431 | 28.7 | 287.3 | 137.2 | 160.3 | 253.1 | 0 | 17853.0 | 60.7 |
| bhaskera-r8-batched | bench_long_q10 | 16 | 619 | 41.3 | 412.7 | 391.0 | 552.6 | 680.2 | 0 | 17857.0 | 92.4 |
| bhaskera-r8-batched | bench_long_q10 | 64 | 615 | 41.0 | 410.0 | 1602.9 | 1704.1 | 1848.5 | 0 | 17857.0 | 94.8 |
| bhaskera-r8-gateway | bench_long_q10 | 1 | 122 | 8.1 | 81.3 | 123.9 | 128.3 | 134.0 | 0 | 9769.0 | 21.6 |
| bhaskera-r8-gateway | bench_long_q10 | 4 | 431 | 28.7 | 287.3 | 137.3 | 163.5 | 245.5 | 0 | 9789.0 | 53.1 |
| bhaskera-r8-gateway | bench_long_q10 | 16 | 687 | 45.8 | 458.0 | 346.1 | 480.4 | 588.0 | 0 | 9793.0 | 94.3 |
| bhaskera-r8-gateway | bench_long_q10 | 64 | 690 | 46.0 | 460.0 | 1394.4 | 1489.7 | 1572.3 | 0 | 9793.0 | 92.0 |
| bhaskera-r8-q4_k_m | bench_long_q10 | 1 | 125 | 8.3 | 83.3 | 121.0 | 125.4 | 130.5 | 0 | 8097.0 | 18.3 |
| bhaskera-r8-q4_k_m | bench_long_q10 | 4 | 436 | 29.1 | 290.7 | 134.2 | 161.7 | 241.7 | 0 | 8109.0 | 48.7 |
| bhaskera-r8-q4_k_m | bench_long_q10 | 16 | 702 | 46.8 | 468.0 | 337.3 | 470.7 | 587.3 | 0 | 8113.0 | 94.6 |
| bhaskera-r8-q4_k_m | bench_long_q10 | 64 | 713 | 47.5 | 475.3 | 1348.7 | 1460.2 | 1547.6 | 0 | 8113.0 | 94.7 |
| bhaskera-r8-q8_0 | bench_long_q10 | 1 | 125 | 8.3 | 83.3 | 120.5 | 126.6 | 133.2 | 0 | 9777.0 | 18.4 |
| bhaskera-r8-q8_0 | bench_long_q10 | 4 | 441 | 29.4 | 294.0 | 132.5 | 162.3 | 246.1 | 0 | 9789.0 | 64.7 |
| bhaskera-r8-q8_0 | bench_long_q10 | 16 | 684 | 45.6 | 456.0 | 344.8 | 491.5 | 613.3 | 0 | 9793.0 | 94.1 |
| bhaskera-r8-q8_0 | bench_long_q10 | 64 | 691 | 46.1 | 460.7 | 1390.4 | 1490.6 | 1575.2 | 0 | 9793.0 | 96.7 |
| bhaskera-r8-staterestore | bench_long_q10 | 1 | 101 | 6.7 | 67.3 | 150.6 | 153.7 | 156.1 | 0 | 9777.0 | 32.8 |
| bhaskera-r8-staterestore | bench_long_q10 | 4 | 267 | 17.8 | 178.0 | 216.0 | 293.2 | 416.9 | 0 | 9789.0 | 74.7 |
| bhaskera-r8-staterestore | bench_long_q10 | 16 | 297 | 19.8 | 198.0 | 758.9 | 1285.8 | 2016.8 | 0 | 9793.0 | 95.1 |
| bhaskera-r8-staterestore | bench_long_q10 | 64 | 296 | 19.7 | 197.3 | 3140.1 | 3908.9 | 4634.4 | 0 | 9793.0 | 99.8 |
| upstream-cpu-q8_0 | bench_long_q10 | 1 | 18 | 1.2 | 12.0 | 854.1 | 876.7 | 893.5 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q10 | 4 | 18 | 1.2 | 12.0 | 3401.5 | 3427.5 | 3433.8 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q10 | 16 | 17 | 1.1 | 11.3 | 13756.8 | 13805.7 | 13809.3 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q10 | 64 | 0 | 0.0 | 0.0 | nan | nan | nan | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_long_q10 | 1 | 123 | 8.2 | 82.0 | 119.8 | 130.1 | 132.5 | 0 | 1018.0 | 28.7 |
| upstream-cuda-q4_k_m | bench_long_q10 | 4 | 116 | 7.7 | 77.3 | 525.2 | 530.6 | 535.9 | 0 | 1018.0 | 30.7 |
| upstream-cuda-q4_k_m | bench_long_q10 | 16 | 120 | 8.0 | 80.0 | 2007.4 | 2038.6 | 2058.9 | 0 | 1018.0 | 27.7 |
| upstream-cuda-q4_k_m | bench_long_q10 | 64 | 127 | 8.5 | 84.7 | 6877.4 | 9537.1 | 9607.3 | 0 | 1018.0 | 30.5 |
| upstream-cuda-q8_0 | bench_long_q10 | 1 | 116 | 7.7 | 77.3 | 130.8 | 133.9 | 138.8 | 0 | 1228.0 | 31.6 |
| upstream-cuda-q8_0 | bench_long_q10 | 4 | 111 | 7.4 | 74.0 | 547.2 | 556.4 | 563.6 | 0 | 1228.0 | 35.1 |
| upstream-cuda-q8_0 | bench_long_q10 | 16 | 116 | 7.7 | 77.3 | 2086.9 | 2139.8 | 2182.9 | 0 | 1228.0 | 33.2 |
| upstream-cuda-q8_0 | bench_long_q10 | 64 | 130 | 8.7 | 86.7 | 7378.1 | 9825.6 | 9957.8 | 0 | 1228.0 | 34.8 |

## bench_long_q3

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_long_q3 | 1 | 281 | 18.7 | 56.2 | 53.3 | 55.4 | 58.9 | 0 | 1228.0 | 16.7 |
| bhaskera-r1-q8_0 | bench_long_q3 | 4 | 339 | 22.6 | 67.8 | 176.5 | 180.0 | 182.9 | 0 | 1228.0 | 23.2 |
| bhaskera-r1-q8_0 | bench_long_q3 | 16 | 307 | 20.5 | 61.4 | 780.0 | 826.4 | 882.8 | 0 | 1228.0 | 21.9 |
| bhaskera-r1-q8_0 | bench_long_q3 | 64 | 281 | 18.7 | 56.2 | 3640.7 | 5186.5 | 5458.7 | 0 | 1228.0 | 19.5 |
| bhaskera-r2-q8_0 | bench_long_q3 | 1 | 278 | 18.5 | 55.6 | 53.7 | 56.2 | 58.0 | 0 | 2451.0 | 18.8 |
| bhaskera-r2-q8_0 | bench_long_q3 | 4 | 656 | 43.7 | 131.2 | 89.7 | 99.9 | 107.7 | 0 | 2451.0 | 43.5 |
| bhaskera-r2-q8_0 | bench_long_q3 | 16 | 639 | 42.6 | 127.8 | 391.6 | 459.6 | 519.4 | 0 | 2451.0 | 43.3 |
| bhaskera-r2-q8_0 | bench_long_q3 | 64 | 576 | 38.4 | 115.2 | 1780.3 | 2555.6 | 2864.8 | 0 | 2451.0 | 39.1 |
| bhaskera-r4-q8_0 | bench_long_q3 | 1 | 276 | 18.4 | 55.2 | 54.2 | 57.0 | 61.3 | 0 | 4898.0 | 24.7 |
| bhaskera-r4-q8_0 | bench_long_q3 | 4 | 856 | 57.1 | 171.2 | 65.3 | 96.0 | 120.1 | 0 | 4898.0 | 60.9 |
| bhaskera-r4-q8_0 | bench_long_q3 | 16 | 1170 | 78.0 | 234.0 | 204.5 | 220.8 | 238.9 | 0 | 4898.0 | 82.0 |
| bhaskera-r4-q8_0 | bench_long_q3 | 64 | 1124 | 74.9 | 224.8 | 863.3 | 1345.5 | 1670.2 | 0 | 4898.0 | 78.9 |
| bhaskera-r8-batched | bench_long_q3 | 1 | 233 | 15.5 | 46.6 | 64.1 | 66.8 | 71.2 | 0 | 17857.0 | 21.5 |
| bhaskera-r8-batched | bench_long_q3 | 4 | 801 | 53.4 | 160.2 | 72.4 | 87.6 | 111.9 | 0 | 17857.0 | 62.9 |
| bhaskera-r8-batched | bench_long_q3 | 16 | 1206 | 80.4 | 241.2 | 198.5 | 273.9 | 324.4 | 0 | 17857.0 | 97.3 |
| bhaskera-r8-batched | bench_long_q3 | 64 | 1198 | 79.9 | 239.6 | 808.1 | 846.7 | 882.8 | 0 | 17857.0 | 100.0 |
| bhaskera-r8-gateway | bench_long_q3 | 1 | 254 | 16.9 | 50.8 | 58.8 | 61.8 | 66.3 | 0 | 9793.0 | 22.3 |
| bhaskera-r8-gateway | bench_long_q3 | 4 | 889 | 59.3 | 177.8 | 65.4 | 77.9 | 109.1 | 0 | 9793.0 | 58.2 |
| bhaskera-r8-gateway | bench_long_q3 | 16 | 1341 | 89.4 | 268.2 | 178.3 | 245.7 | 293.1 | 0 | 9793.0 | 97.9 |
| bhaskera-r8-gateway | bench_long_q3 | 64 | 1342 | 89.5 | 268.4 | 704.4 | 929.6 | 1213.4 | 0 | 9793.0 | 99.9 |
| bhaskera-r8-q4_k_m | bench_long_q3 | 1 | 267 | 17.8 | 53.4 | 56.3 | 59.0 | 63.8 | 0 | 8113.0 | 17.5 |
| bhaskera-r8-q4_k_m | bench_long_q3 | 4 | 924 | 61.6 | 184.8 | 62.7 | 76.3 | 105.7 | 0 | 8113.0 | 64.0 |
| bhaskera-r8-q4_k_m | bench_long_q3 | 16 | 1373 | 91.5 | 274.6 | 173.0 | 245.0 | 296.3 | 0 | 8113.0 | 96.7 |
| bhaskera-r8-q4_k_m | bench_long_q3 | 64 | 1372 | 91.5 | 274.4 | 701.5 | 843.2 | 1265.3 | 0 | 8113.0 | 99.9 |
| bhaskera-r8-q8_0 | bench_long_q3 | 1 | 271 | 18.1 | 54.2 | 55.2 | 57.7 | 63.6 | 0 | 9793.0 | 20.8 |
| bhaskera-r8-q8_0 | bench_long_q3 | 4 | 932 | 62.1 | 186.4 | 62.3 | 76.1 | 104.5 | 0 | 9793.0 | 66.6 |
| bhaskera-r8-q8_0 | bench_long_q3 | 16 | 1345 | 89.7 | 269.0 | 176.9 | 249.4 | 295.2 | 0 | 9793.0 | 99.0 |
| bhaskera-r8-q8_0 | bench_long_q3 | 64 | 1344 | 89.6 | 268.8 | 710.0 | 918.9 | 1253.8 | 0 | 9793.0 | 99.9 |
| bhaskera-r8-staterestore | bench_long_q3 | 1 | 212 | 14.1 | 42.4 | 70.8 | 73.7 | 75.9 | 0 | 9793.0 | 28.0 |
| bhaskera-r8-staterestore | bench_long_q3 | 4 | 676 | 45.1 | 135.2 | 85.8 | 108.1 | 156.7 | 0 | 9793.0 | 79.8 |
| bhaskera-r8-staterestore | bench_long_q3 | 16 | 785 | 52.3 | 157.0 | 281.1 | 482.0 | 663.7 | 0 | 9793.0 | 96.0 |
| bhaskera-r8-staterestore | bench_long_q3 | 64 | 781 | 52.1 | 156.2 | 1218.8 | 1398.1 | 1555.8 | 0 | 9793.0 | 99.5 |
| upstream-cpu-q8_0 | bench_long_q3 | 1 | 35 | 2.3 | 7.0 | 428.4 | 459.5 | 472.4 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q3 | 4 | 36 | 2.4 | 7.2 | 1702.8 | 1750.6 | 1779.7 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q3 | 16 | 37 | 2.5 | 7.4 | 6495.8 | 6652.4 | 6680.5 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_long_q3 | 64 | 27 | 1.8 | 5.4 | 22220.9 | 26515.4 | 26931.7 | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_long_q3 | 1 | 251 | 16.7 | 50.2 | 59.7 | 61.8 | 63.3 | 0 | 1018.0 | 27.7 |
| upstream-cuda-q4_k_m | bench_long_q3 | 4 | 272 | 18.1 | 54.4 | 223.0 | 226.6 | 230.3 | 0 | 1018.0 | 28.7 |
| upstream-cuda-q4_k_m | bench_long_q3 | 16 | 279 | 18.6 | 55.8 | 866.6 | 904.8 | 910.8 | 0 | 1018.0 | 31.1 |
| upstream-cuda-q4_k_m | bench_long_q3 | 64 | 275 | 18.3 | 55.0 | 3177.1 | 4117.1 | 4171.7 | 0 | 1018.0 | 28.5 |
| upstream-cuda-q8_0 | bench_long_q3 | 1 | 246 | 16.4 | 49.2 | 60.5 | 62.6 | 64.2 | 0 | 1228.0 | 27.9 |
| upstream-cuda-q8_0 | bench_long_q3 | 4 | 283 | 18.9 | 56.6 | 214.3 | 219.1 | 222.6 | 0 | 1228.0 | 33.5 |
| upstream-cuda-q8_0 | bench_long_q3 | 16 | 285 | 19.0 | 57.0 | 847.5 | 886.3 | 926.5 | 0 | 1228.0 | 32.5 |
| upstream-cuda-q8_0 | bench_long_q3 | 64 | 275 | 18.3 | 55.0 | 3357.9 | 4179.4 | 4264.3 | 0 | 1228.0 | 31.0 |

## bench_short_q1

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_short_q1 | 1 | 637 | 42.5 | 42.5 | 23.5 | 24.4 | 26.1 | 0 | 1228.0 | 16.6 |
| bhaskera-r1-q8_0 | bench_short_q1 | 4 | 965 | 64.3 | 64.3 | 62.1 | 64.2 | 67.2 | 0 | 1228.0 | 25.7 |
| bhaskera-r1-q8_0 | bench_short_q1 | 16 | 794 | 52.9 | 52.9 | 336.9 | 392.5 | 456.9 | 0 | 1228.0 | 22.3 |
| bhaskera-r1-q8_0 | bench_short_q1 | 64 | 744 | 49.6 | 49.6 | 1203.3 | 2299.3 | 2498.4 | 0 | 1228.0 | 20.1 |
| bhaskera-r2-q8_0 | bench_short_q1 | 1 | 614 | 40.9 | 40.9 | 24.2 | 25.7 | 27.4 | 0 | 2451.0 | 15.5 |
| bhaskera-r2-q8_0 | bench_short_q1 | 4 | 1899 | 126.6 | 126.6 | 31.4 | 33.2 | 36.1 | 0 | 2451.0 | 49.5 |
| bhaskera-r2-q8_0 | bench_short_q1 | 16 | 1735 | 115.7 | 115.7 | 134.2 | 195.1 | 255.1 | 0 | 2451.0 | 49.0 |
| bhaskera-r2-q8_0 | bench_short_q1 | 64 | 1329 | 88.6 | 88.6 | 693.0 | 1266.3 | 1573.1 | 0 | 2451.0 | 36.5 |
| bhaskera-r4-q8_0 | bench_short_q1 | 1 | 604 | 40.3 | 40.3 | 24.7 | 26.3 | 28.4 | 0 | 4898.0 | 20.4 |
| bhaskera-r4-q8_0 | bench_short_q1 | 4 | 2042 | 136.1 | 136.1 | 28.1 | 36.2 | 45.4 | 0 | 4898.0 | 55.0 |
| bhaskera-r4-q8_0 | bench_short_q1 | 16 | 2996 | 199.7 | 199.7 | 80.4 | 94.7 | 107.1 | 0 | 4898.0 | 87.6 |
| bhaskera-r4-q8_0 | bench_short_q1 | 64 | 2531 | 168.7 | 168.7 | 355.2 | 603.1 | 847.9 | 0 | 4898.0 | 76.1 |
| bhaskera-r8-batched | bench_short_q1 | 1 | 437 | 29.1 | 29.1 | 34.3 | 36.0 | 38.8 | 0 | 17857.0 | 20.6 |
| bhaskera-r8-batched | bench_short_q1 | 4 | 1588 | 105.9 | 105.9 | 37.0 | 43.1 | 52.4 | 0 | 17857.0 | 56.4 |
| bhaskera-r8-batched | bench_short_q1 | 16 | 2656 | 177.1 | 177.1 | 88.2 | 120.0 | 143.4 | 0 | 17857.0 | 97.2 |
| bhaskera-r8-batched | bench_short_q1 | 64 | 2574 | 171.6 | 171.6 | 378.9 | 409.1 | 428.7 | 0 | 17857.0 | 100.0 |
| bhaskera-r8-gateway | bench_short_q1 | 1 | 515 | 34.3 | 34.3 | 29.1 | 31.2 | 34.6 | 0 | 9793.0 | 19.9 |
| bhaskera-r8-gateway | bench_short_q1 | 4 | 1836 | 122.4 | 122.4 | 31.9 | 36.5 | 46.9 | 0 | 9793.0 | 49.1 |
| bhaskera-r8-gateway | bench_short_q1 | 16 | 3103 | 206.9 | 206.9 | 74.2 | 97.6 | 126.1 | 0 | 9793.0 | 95.1 |
| bhaskera-r8-gateway | bench_short_q1 | 64 | 2284 | 152.3 | 152.3 | 360.5 | 618.1 | 1611.9 | 0 | 9793.0 | 68.2 |
| bhaskera-r8-q4_k_m | bench_short_q1 | 1 | 572 | 38.1 | 38.1 | 26.2 | 27.9 | 30.2 | 0 | 8113.0 | 13.6 |
| bhaskera-r8-q4_k_m | bench_short_q1 | 4 | 2055 | 137.0 | 137.0 | 28.6 | 32.6 | 43.6 | 0 | 8113.0 | 53.0 |
| bhaskera-r8-q4_k_m | bench_short_q1 | 16 | 3227 | 215.1 | 215.1 | 72.7 | 91.5 | 111.4 | 0 | 8113.0 | 90.0 |
| bhaskera-r8-q4_k_m | bench_short_q1 | 64 | 2682 | 178.8 | 178.8 | 316.0 | 519.4 | 765.8 | 0 | 8113.0 | 65.8 |
| bhaskera-r8-q8_0 | bench_short_q1 | 1 | 580 | 38.7 | 38.7 | 25.8 | 27.4 | 29.1 | 0 | 9793.0 | 19.1 |
| bhaskera-r8-q8_0 | bench_short_q1 | 4 | 2049 | 136.6 | 136.6 | 28.5 | 33.0 | 44.5 | 0 | 9793.0 | 57.6 |
| bhaskera-r8-q8_0 | bench_short_q1 | 16 | 3190 | 212.7 | 212.7 | 72.9 | 95.3 | 116.0 | 0 | 9793.0 | 95.9 |
| bhaskera-r8-q8_0 | bench_short_q1 | 64 | 2495 | 166.3 | 166.3 | 321.3 | 587.2 | 1405.4 | 0 | 9793.0 | 84.0 |
| bhaskera-r8-staterestore | bench_short_q1 | 1 | 571 | 38.1 | 38.1 | 26.1 | 28.1 | 31.4 | 0 | 9793.0 | 18.1 |
| bhaskera-r8-staterestore | bench_short_q1 | 4 | 2058 | 137.2 | 137.2 | 28.5 | 32.6 | 43.3 | 0 | 9793.0 | 55.9 |
| bhaskera-r8-staterestore | bench_short_q1 | 16 | 3205 | 213.7 | 213.7 | 72.6 | 96.2 | 114.5 | 0 | 9793.0 | 96.0 |
| bhaskera-r8-staterestore | bench_short_q1 | 64 | 2832 | 188.8 | 188.8 | 304.6 | 483.6 | 712.6 | 0 | 9793.0 | 80.7 |
| upstream-cpu-q8_0 | bench_short_q1 | 1 | 160 | 10.7 | 10.7 | 92.8 | 100.3 | 107.1 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q1 | 4 | 159 | 10.6 | 10.6 | 377.4 | 396.9 | 404.2 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q1 | 16 | 152 | 10.1 | 10.1 | 1591.3 | 1622.6 | 1630.8 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q1 | 64 | 154 | 10.3 | 10.3 | 5569.9 | 7100.7 | 7314.3 | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_short_q1 | 1 | 905 | 60.3 | 60.3 | 16.9 | 17.5 | 18.9 | 0 | 1018.0 | 23.0 |
| upstream-cuda-q4_k_m | bench_short_q1 | 4 | 1037 | 69.1 | 69.1 | 57.8 | 59.4 | 61.1 | 0 | 1018.0 | 26.9 |
| upstream-cuda-q4_k_m | bench_short_q1 | 16 | 1027 | 68.5 | 68.5 | 234.6 | 238.8 | 242.3 | 0 | 1018.0 | 25.9 |
| upstream-cuda-q4_k_m | bench_short_q1 | 64 | 1036 | 69.1 | 69.1 | 903.6 | 1009.3 | 1086.2 | 0 | 1018.0 | 26.6 |
| upstream-cuda-q8_0 | bench_short_q1 | 1 | 868 | 57.9 | 57.9 | 17.2 | 17.8 | 19.3 | 0 | 1228.0 | 24.4 |
| upstream-cuda-q8_0 | bench_short_q1 | 4 | 1003 | 66.9 | 66.9 | 60.1 | 61.9 | 64.1 | 0 | 1228.0 | 28.1 |
| upstream-cuda-q8_0 | bench_short_q1 | 16 | 1028 | 68.5 | 68.5 | 233.6 | 240.6 | 244.1 | 0 | 1228.0 | 27.8 |
| upstream-cuda-q8_0 | bench_short_q1 | 64 | 1012 | 67.5 | 67.5 | 927.3 | 1071.4 | 1186.9 | 0 | 1228.0 | 27.5 |

## bench_short_q10

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_short_q10 | 1 | 148 | 9.9 | 98.7 | 98.9 | 112.4 | 117.0 | 0 | 1228.0 | 16.3 |
| bhaskera-r1-q8_0 | bench_short_q10 | 4 | 150 | 10.0 | 100.0 | 402.1 | 407.3 | 410.5 | 0 | 1228.0 | 16.6 |
| bhaskera-r1-q8_0 | bench_short_q10 | 16 | 143 | 9.5 | 95.3 | 1690.1 | 1803.7 | 1997.8 | 0 | 1228.0 | 15.7 |
| bhaskera-r1-q8_0 | bench_short_q10 | 64 | 121 | 8.1 | 80.7 | 7947.1 | 8112.2 | 8244.8 | 0 | 1228.0 | 11.9 |
| bhaskera-r2-q8_0 | bench_short_q10 | 1 | 151 | 10.1 | 100.7 | 97.6 | 108.7 | 115.1 | 0 | 2451.0 | 18.3 |
| bhaskera-r2-q8_0 | bench_short_q10 | 4 | 299 | 19.9 | 199.3 | 201.1 | 209.8 | 218.3 | 0 | 2451.0 | 33.0 |
| bhaskera-r2-q8_0 | bench_short_q10 | 16 | 277 | 18.5 | 184.7 | 870.7 | 982.0 | 1083.5 | 0 | 2451.0 | 31.4 |
| bhaskera-r2-q8_0 | bench_short_q10 | 64 | 244 | 16.3 | 162.7 | 4578.4 | 5023.7 | 5257.1 | 0 | 2451.0 | 26.3 |
| bhaskera-r4-q8_0 | bench_short_q10 | 1 | 148 | 9.9 | 98.7 | 99.5 | 111.7 | 117.4 | 0 | 4898.0 | 17.7 |
| bhaskera-r4-q8_0 | bench_short_q10 | 4 | 434 | 28.9 | 289.3 | 127.0 | 204.8 | 249.8 | 0 | 4898.0 | 47.3 |
| bhaskera-r4-q8_0 | bench_short_q10 | 16 | 561 | 37.4 | 374.0 | 425.9 | 460.0 | 500.1 | 0 | 4898.0 | 61.3 |
| bhaskera-r4-q8_0 | bench_short_q10 | 64 | 535 | 35.7 | 356.7 | 1833.3 | 1953.9 | 2083.4 | 0 | 4898.0 | 63.0 |
| bhaskera-r8-batched | bench_short_q10 | 1 | 133 | 8.9 | 88.7 | 110.6 | 123.3 | 126.9 | 0 | 17857.0 | 22.1 |
| bhaskera-r8-batched | bench_short_q10 | 4 | 478 | 31.9 | 318.7 | 122.2 | 146.5 | 198.0 | 0 | 17857.0 | 55.6 |
| bhaskera-r8-batched | bench_short_q10 | 16 | 734 | 48.9 | 489.3 | 323.9 | 445.9 | 545.5 | 0 | 17857.0 | 98.0 |
| bhaskera-r8-batched | bench_short_q10 | 64 | 744 | 49.6 | 496.0 | 1299.7 | 1371.7 | 1432.9 | 0 | 17857.0 | 94.7 |
| bhaskera-r8-gateway | bench_short_q10 | 1 | 140 | 9.3 | 93.3 | 103.8 | 117.3 | 122.8 | 0 | 9793.0 | 18.8 |
| bhaskera-r8-gateway | bench_short_q10 | 4 | 494 | 32.9 | 329.3 | 116.3 | 142.8 | 230.3 | 0 | 9793.0 | 54.2 |
| bhaskera-r8-gateway | bench_short_q10 | 16 | 791 | 52.7 | 527.3 | 301.0 | 424.5 | 501.7 | 0 | 9793.0 | 96.6 |
| bhaskera-r8-gateway | bench_short_q10 | 64 | 795 | 53.0 | 530.0 | 1209.7 | 1274.0 | 1337.5 | 0 | 9793.0 | 96.9 |
| bhaskera-r8-q4_k_m | bench_short_q10 | 1 | 144 | 9.6 | 96.0 | 102.3 | 114.3 | 119.1 | 0 | 8113.0 | 18.5 |
| bhaskera-r8-q4_k_m | bench_short_q10 | 4 | 501 | 33.4 | 334.0 | 115.4 | 139.8 | 226.5 | 0 | 8113.0 | 50.5 |
| bhaskera-r8-q4_k_m | bench_short_q10 | 16 | 854 | 56.9 | 569.3 | 275.1 | 401.3 | 491.8 | 0 | 8113.0 | 94.3 |
| bhaskera-r8-q4_k_m | bench_short_q10 | 64 | 863 | 57.5 | 575.3 | 1116.0 | 1185.3 | 1236.9 | 0 | 8113.0 | 94.3 |
| bhaskera-r8-q8_0 | bench_short_q10 | 1 | 146 | 9.7 | 97.3 | 100.4 | 112.1 | 116.3 | 0 | 9793.0 | 19.2 |
| bhaskera-r8-q8_0 | bench_short_q10 | 4 | 508 | 33.9 | 338.7 | 114.1 | 141.1 | 205.9 | 0 | 9793.0 | 55.2 |
| bhaskera-r8-q8_0 | bench_short_q10 | 16 | 788 | 52.5 | 525.3 | 299.1 | 425.2 | 514.9 | 0 | 9793.0 | 92.1 |
| bhaskera-r8-q8_0 | bench_short_q10 | 64 | 797 | 53.1 | 531.3 | 1207.7 | 1266.8 | 1327.7 | 0 | 9793.0 | 99.2 |
| bhaskera-r8-staterestore | bench_short_q10 | 1 | 114 | 7.6 | 76.0 | 131.1 | 140.1 | 146.1 | 0 | 9793.0 | 31.8 |
| bhaskera-r8-staterestore | bench_short_q10 | 4 | 203 | 13.5 | 135.3 | 304.2 | 350.2 | 492.9 | 0 | 9793.0 | 90.4 |
| bhaskera-r8-staterestore | bench_short_q10 | 16 | 221 | 14.7 | 147.3 | 1058.7 | 1571.8 | 1728.8 | 0 | 9793.0 | 97.5 |
| bhaskera-r8-staterestore | bench_short_q10 | 64 | 225 | 15.0 | 150.0 | 4333.2 | 4506.9 | 4634.9 | 0 | 9793.0 | 98.6 |
| upstream-cpu-q8_0 | bench_short_q10 | 1 | 20 | 1.3 | 13.3 | 742.4 | 880.1 | 880.6 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q10 | 4 | 17 | 1.1 | 11.3 | 3454.8 | 3609.0 | 3672.0 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q10 | 16 | 22 | 1.5 | 14.7 | 11141.4 | 11196.3 | 11220.6 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q10 | 64 | 25 | 1.7 | 16.7 | 36390.7 | 43310.4 | 44682.0 | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_short_q10 | 1 | 130 | 8.7 | 86.7 | 116.1 | 118.2 | 124.9 | 0 | 1018.0 | 30.6 |
| upstream-cuda-q4_k_m | bench_short_q10 | 4 | 128 | 8.5 | 85.3 | 473.0 | 485.8 | 494.0 | 0 | 1018.0 | 27.3 |
| upstream-cuda-q4_k_m | bench_short_q10 | 16 | 128 | 8.5 | 85.3 | 1868.9 | 1913.7 | 1926.5 | 0 | 1018.0 | 29.8 |
| upstream-cuda-q4_k_m | bench_short_q10 | 64 | 129 | 8.6 | 86.0 | 6211.2 | 8396.7 | 8419.4 | 0 | 1018.0 | 32.2 |
| upstream-cuda-q8_0 | bench_short_q10 | 1 | 139 | 9.3 | 92.7 | 117.7 | 123.9 | 127.6 | 0 | 1228.0 | 32.9 |
| upstream-cuda-q8_0 | bench_short_q10 | 4 | 131 | 8.7 | 87.3 | 457.4 | 463.9 | 467.7 | 0 | 1228.0 | 34.6 |
| upstream-cuda-q8_0 | bench_short_q10 | 16 | 123 | 8.2 | 82.0 | 1940.3 | 1999.4 | 2006.6 | 0 | 1228.0 | 32.1 |
| upstream-cuda-q8_0 | bench_short_q10 | 64 | 152 | 10.1 | 101.3 | 6643.7 | 8848.2 | 8899.2 | 0 | 1228.0 | 33.2 |

## bench_short_q3

| label | requests | concurrency | completed | req_per_s | questions_per_s | p50_ms | p90_ms | p99_ms | errors | max_memory_mib | mean_util_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bhaskera-r1-q8_0 | bench_short_q3 | 1 | 320 | 21.3 | 64.0 | 46.5 | 49.1 | 51.7 | 0 | 1228.0 | 15.0 |
| bhaskera-r1-q8_0 | bench_short_q3 | 4 | 397 | 26.5 | 79.4 | 151.0 | 155.3 | 158.9 | 0 | 1228.0 | 20.6 |
| bhaskera-r1-q8_0 | bench_short_q3 | 16 | 346 | 23.1 | 69.2 | 694.6 | 748.0 | 826.2 | 0 | 1228.0 | 18.4 |
| bhaskera-r1-q8_0 | bench_short_q3 | 64 | 314 | 20.9 | 62.8 | 3087.1 | 4567.1 | 4701.9 | 0 | 1228.0 | 15.9 |
| bhaskera-r2-q8_0 | bench_short_q3 | 1 | 318 | 21.2 | 63.6 | 46.9 | 49.3 | 52.8 | 0 | 2451.0 | 15.7 |
| bhaskera-r2-q8_0 | bench_short_q3 | 4 | 780 | 52.0 | 156.0 | 76.7 | 79.9 | 86.0 | 0 | 2451.0 | 38.8 |
| bhaskera-r2-q8_0 | bench_short_q3 | 16 | 741 | 49.4 | 148.2 | 325.5 | 364.5 | 405.2 | 0 | 2451.0 | 37.9 |
| bhaskera-r2-q8_0 | bench_short_q3 | 64 | 658 | 43.9 | 131.6 | 1562.0 | 2299.0 | 2592.1 | 0 | 2451.0 | 35.4 |
| bhaskera-r4-q8_0 | bench_short_q3 | 1 | 312 | 20.8 | 62.4 | 47.9 | 50.8 | 54.3 | 0 | 4898.0 | 16.3 |
| bhaskera-r4-q8_0 | bench_short_q3 | 4 | 1028 | 68.5 | 205.6 | 53.6 | 80.0 | 93.6 | 0 | 4898.0 | 52.5 |
| bhaskera-r4-q8_0 | bench_short_q3 | 16 | 1409 | 93.9 | 281.8 | 170.6 | 184.7 | 197.1 | 0 | 4898.0 | 74.8 |
| bhaskera-r4-q8_0 | bench_short_q3 | 64 | 1345 | 89.7 | 269.0 | 715.6 | 1147.9 | 1475.0 | 0 | 4898.0 | 74.0 |
| bhaskera-r8-batched | bench_short_q3 | 1 | 264 | 17.6 | 52.8 | 56.7 | 59.3 | 63.9 | 0 | 17857.0 | 14.1 |
| bhaskera-r8-batched | bench_short_q3 | 4 | 915 | 61.0 | 183.0 | 63.1 | 77.8 | 101.6 | 0 | 17857.0 | 56.0 |
| bhaskera-r8-batched | bench_short_q3 | 16 | 1526 | 101.7 | 305.2 | 157.3 | 215.0 | 255.9 | 0 | 17857.0 | 96.8 |
| bhaskera-r8-batched | bench_short_q3 | 64 | 1499 | 99.9 | 299.8 | 650.3 | 679.8 | 704.3 | 0 | 17857.0 | 100.0 |
| bhaskera-r8-gateway | bench_short_q3 | 1 | 287 | 19.1 | 57.4 | 52.2 | 55.5 | 60.1 | 0 | 9793.0 | 16.9 |
| bhaskera-r8-gateway | bench_short_q3 | 4 | 1037 | 69.1 | 207.4 | 56.6 | 65.9 | 89.1 | 0 | 9793.0 | 50.0 |
| bhaskera-r8-gateway | bench_short_q3 | 16 | 1745 | 116.3 | 349.0 | 136.2 | 189.1 | 226.6 | 0 | 9793.0 | 97.0 |
| bhaskera-r8-gateway | bench_short_q3 | 64 | 1742 | 116.1 | 348.4 | 516.4 | 780.4 | 1068.0 | 0 | 9793.0 | 99.9 |
| bhaskera-r8-q4_k_m | bench_short_q3 | 1 | 307 | 20.5 | 61.4 | 49.1 | 51.8 | 54.1 | 0 | 8113.0 | 16.9 |
| bhaskera-r8-q4_k_m | bench_short_q3 | 4 | 1088 | 72.5 | 217.6 | 53.2 | 62.8 | 87.2 | 0 | 8113.0 | 55.0 |
| bhaskera-r8-q4_k_m | bench_short_q3 | 16 | 1913 | 127.5 | 382.6 | 123.6 | 173.9 | 210.3 | 0 | 8113.0 | 97.8 |
| bhaskera-r8-q4_k_m | bench_short_q3 | 64 | 1912 | 127.5 | 382.4 | 465.5 | 735.8 | 1060.5 | 0 | 8113.0 | 93.8 |
| bhaskera-r8-q8_0 | bench_short_q3 | 1 | 309 | 20.6 | 61.8 | 48.4 | 51.4 | 53.6 | 0 | 9793.0 | 18.2 |
| bhaskera-r8-q8_0 | bench_short_q3 | 4 | 1102 | 73.5 | 220.4 | 53.0 | 61.8 | 85.9 | 0 | 9793.0 | 56.7 |
| bhaskera-r8-q8_0 | bench_short_q3 | 16 | 1742 | 116.1 | 348.4 | 136.4 | 190.7 | 228.8 | 0 | 9793.0 | 97.7 |
| bhaskera-r8-q8_0 | bench_short_q3 | 64 | 1738 | 115.9 | 347.6 | 511.9 | 801.4 | 1114.2 | 0 | 9793.0 | 100.0 |
| bhaskera-r8-staterestore | bench_short_q3 | 1 | 243 | 16.2 | 48.6 | 61.5 | 65.0 | 67.9 | 0 | 9793.0 | 24.0 |
| bhaskera-r8-staterestore | bench_short_q3 | 4 | 604 | 40.3 | 120.8 | 105.0 | 117.3 | 154.2 | 0 | 9793.0 | 75.3 |
| bhaskera-r8-staterestore | bench_short_q3 | 16 | 706 | 47.1 | 141.2 | 336.3 | 489.4 | 526.8 | 0 | 9793.0 | 97.9 |
| bhaskera-r8-staterestore | bench_short_q3 | 64 | 709 | 47.3 | 141.8 | 1370.0 | 1530.5 | 1616.2 | 0 | 9793.0 | 96.2 |
| upstream-cpu-q8_0 | bench_short_q3 | 1 | 57 | 3.8 | 11.4 | 260.9 | 272.2 | 284.7 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q3 | 4 | 57 | 3.8 | 11.4 | 1038.7 | 1074.9 | 1122.9 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q3 | 16 | 57 | 3.8 | 11.4 | 4215.5 | 4234.7 | 4258.5 | 0 | 272.0 | 0.0 |
| upstream-cpu-q8_0 | bench_short_q3 | 64 | 52 | 3.5 | 10.4 | 13708.4 | 16269.5 | 16562.5 | 0 | 272.0 | 0.0 |
| upstream-cuda-q4_k_m | bench_short_q3 | 1 | 323 | 21.5 | 64.6 | 46.7 | 52.0 | 54.0 | 0 | 1018.0 | 27.3 |
| upstream-cuda-q4_k_m | bench_short_q3 | 4 | 337 | 22.5 | 67.4 | 180.8 | 192.9 | 197.1 | 0 | 1018.0 | 29.2 |
| upstream-cuda-q4_k_m | bench_short_q3 | 16 | 315 | 21.0 | 63.0 | 762.7 | 778.2 | 785.2 | 0 | 1018.0 | 25.7 |
| upstream-cuda-q4_k_m | bench_short_q3 | 64 | 308 | 20.5 | 61.6 | 2831.4 | 3562.8 | 3610.5 | 0 | 1018.0 | 26.6 |
| upstream-cuda-q8_0 | bench_short_q3 | 1 | 295 | 19.7 | 59.0 | 51.7 | 53.9 | 56.4 | 0 | 1228.0 | 27.2 |
| upstream-cuda-q8_0 | bench_short_q3 | 4 | 317 | 21.1 | 63.4 | 193.0 | 196.3 | 206.8 | 0 | 1228.0 | 29.6 |
| upstream-cuda-q8_0 | bench_short_q3 | 16 | 310 | 20.7 | 62.0 | 777.2 | 786.5 | 793.4 | 0 | 1228.0 | 29.0 |
| upstream-cuda-q8_0 | bench_short_q3 | 64 | 325 | 21.7 | 65.0 | 2917.1 | 3289.5 | 3563.4 | 0 | 1228.0 | 30.3 |

