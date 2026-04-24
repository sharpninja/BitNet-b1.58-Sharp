# BitNet Inference Latency Overhaul: Per-Phase Deltas

Measured against Bonsai (782 M params, 36 layers, dim 4096, heads 32, kv 8) on AMD Ryzen 9 5900HX (AVX2), .NET 10. Baseline: single-token decode 25-66 s; /api/chat 12-token generation = 1751 s.

BenchmarkDotNet 0.15.4 with `[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: {1|3}, iterationCount: {3|10})]`, `MemoryDiagnoser`. Small fixture: dim=512, heads=8, kvHeads=2, layers=4. Realistic fixture: dim=4096, heads=32, kvHeads=8, layers=4.

All deltas are reported as `after / before` (Ratio column) so smaller is faster.

## Phase 0: BenchmarkDotNet scaffolding

No perf delta. Establishes `benchmarks/BitNetSharp.Benchmarks/` with BDN 0.15.4, two fixtures, and empty skeleton suites.

## Phase 1: KV cache

**AttentionBenchmarks (MHA, SmallConfig)** — caches K/V per layer; decode reuses pasted rows.

| SeqLen | Baseline (ms) | Cached Decode (ms) | Ratio | Alloc Ratio |
| ---:   | ---:          | ---:               | ---:  | ---:        |
| 32     | 194.32        | 54.34              | 0.28  | 0.03        |
| 128    | 770.70        | 55.49              | 0.07  | 0.007       |
| 512    | 4,643.92      | 52.80              | 0.01  | 0.002       |

Decode time becomes flat in seq_len — exactly what KV cache should buy.

**GenerateBenchmarks (prompt + new tokens, SmallConfig)**:

| PromptLen | NewTokens | FullRecompute (ms) | KvCache (ms) | Ratio | Alloc Ratio |
| ---:      | ---:      | ---:               | ---:         | ---:  | ---:        |
| 8         | 4         | 201.2              | 116.7        | 0.58  | 0.23        |
| 8         | 8         | 356.8              | 181.9        | 0.52  | 0.14        |
| 16        | 4         | 295.4              | 144.4        | 0.49  | 0.21        |
| 16        | 8         | 415.6              | 190.5        | 0.46  | 0.13        |

End-to-end 2x speedup at small seq; larger seqs would scale linearly in the full-recompute column and stay flat in cached, so gap widens.

## Phase 2: Activation-quantisation cache

Shared `QuantizedActivationBlock.FromFloat` reused across Q/K/V (attention) and Gate/Up (SwiGLU). Per-attention FromFloat calls: 3 -> 1. Per-feed-forward: 2 -> 1 (plus one for DownProjection's post-activation input).

Validated by instrumented counter (AsyncLocal<StrongBox<long>>) in `tests/BitNetSharp.Tests/ActivationQuantCacheTests.cs`.

**BitLinearBenchmarks** — `ForwardQuantized` accepts a pre-quantised block; skips the per-call absmax scan.

Selected shapes (Rows, InDim, OutDim). Ratios show `ForwardQuantized / Forward`:

| Rows | InDim | OutDim | Forward (μs) | ForwardQuantized (μs) | Ratio | Alloc Ratio |
| ---: | ---:  | ---:   | ---:         | ---:                  | ---:  | ---:        |
| 1    | 4096  | 512    | 2,695        | 2,388                 | 0.89  | 0.33        |
| 32   | 4096  | 512    | 10,693       | 9,310                 | 0.87  | 0.33        |
| 128  | 4096  | 512    | 33,867       | 31,087                | 0.92  | 0.33        |
| 128  | 4096  | 14336  | 896,694      | 858,221               | 0.96  | 0.93        |

Speedup scales with how small `InDim × Rows` is compared to `OutDim`: when the quantise cost is non-trivial relative to matmul, the saving is visible. For huge OutDim (14336) the matmul dominates.

## Phase 3: SIMD attention inner loop

`AttentionMath.Dot` / `AccumulateWeighted` use `System.Numerics.Vector<float>`. `MemoryMarshal.CreateSpan` aliases `float[,]` as flat `Span<float>` for the K/V rows in the cache.

**Scalar numerical stability**: during STE backprop training, the SIMD reduction order diverges from scalar left-to-right summation, causing perplexity drift across epochs. Training-mode `Forward(float[,])` kept scalar; cache-aware `Forward(..., cache, ...)` and `ForwardFlashDecode(...)` use SIMD.

Validated: `AttentionMathTests` (19 tests) assert SIMD matches scalar oracle within 1e-4 for headDim ∈ {2, 3, 8, 16, 31, 32, 64, 127, 128}. `Perplexity_improves_after_training_on_wikitext_subset` stayed green after the scalar revert.

Microbenchmark is folded into AttentionBenchmarks cached-decode numbers above (already SIMD).

## Phase 4: Fused flash-style attention

`FlashAttention.ForwardDecode` computes `softmax(QK^T/√d) · V` in one pass using online softmax (running max + partition + accumulator per head, `O(headDim)` state). No `[headCount, seqLen, seqLen]` attention-weights tensor is materialised at decode time.

Dispatch: `BitNetLayer.Forward(input, cache, positionOffset)` routes `input.rows == 1` to `ForwardFlashDecode`; otherwise falls through to phase-3 SIMD path.

Validated: `FlashAttentionTests` (7 tests) — fused decode matches dense attention within 1e-4 across prefillLen ∈ {4, 16, 31} for both MHA and GQA. Training path unchanged.

**AttentionBenchmarks (RealisticConfig, GQA)** — fused flash variant added alongside the SIMD cached decode:

| SeqLen | FullSeq (ms)  | CachedDecode (ms) | FlashDecode (ms) | Cached Alloc (KB) | Flash Alloc (KB) |
| ---:   | ---:          | ---:              | ---:             | ---:              | ---:             |
| 32     | 199.96        | 56.02             | 50.75            | 69.4              | 64.4             |
| 128    | 785.73        | 55.99             | 56.21            | 81.4              | 64.4             |
| 512    | 4,978.80      | 50.51             | 57.05            | 129.4             | 64.4             |

Flash is ~10 % faster at short seq and flat in allocations: 64 KB regardless of SeqLen (no N×N attention weights tensor). Cached allocations grow linearly in SeqLen; at SeqLen=512 Flash is 50 % of Cached.

## Phase 5: Streaming `/api/chat`

`BitNetPaperModel.StreamGenerateAsync(prompt, maxTokens, ct)` — producer Task runs the existing sync decode loop holding `lock(_gate)`, writes each token to `Channel<GeneratedToken>`; consumer awaits `ReadAllAsync`. `GenerateResponse` now takes an optional `Action<int> emitToken` and `CancellationToken` (back-compat overload preserved).

`BitNetHostedAgentModel.StreamResponseAsync(string, int?, CancellationToken)` overrides the interface default; yields detokenized text pieces. `OllamaChatEndpoints` iterates the `IAsyncEnumerable<string>` and emits one NDJSON chunk per yielded piece (`done: false`), followed by a terminal `done: true` chunk with timing aggregates.

Validated: `tests/BitNetSharp.Tests/OllamaStreamingChatTests.cs`:
- `StreamTrue_EmitsOneNdjsonLinePerToken_EndingWithDoneTrue(3)` — 4 lines (3 tokens + terminal), 198 ms
- `StreamTrue_EmitsOneNdjsonLinePerToken_EndingWithDoneTrue(5)` — 6 lines, 15 ms
- `StreamFalse_StillReturnsSingleJson` — back-compat single JSON, 16 ms
- `CancellationMidStream_StopsGeneration` — cancel after 3 tokens, stub observes cancellation before emitting the full 10 000, 38 ms

**StreamingLatencyBenchmarks (SmallConfig, 3-example corpus, MaxTokens=8)**:

| Method                     | Mean (ms) | Ratio | Alloc (MB) |
| ---                        | ---:      | ---:  | ---:       |
| Blocking_FullResponse      | 50.96     | 1.00  | 3.36       |
| Streaming_TimeToFirstToken | 57.77     | 1.13  | 3.37       |
| Streaming_FullResponse     | 59.47     | 1.17  | 3.37       |

Channel + worker-task overhead is ~15 %. On this fixture the model is small enough that prefill dominates the streaming TTFT; on Bonsai, TTFT = prefill + one decode step (~2 s target) vs blocking total = prefill + N × decode_ms, which is where the client-visible win lives. 634/634 `dotnet test` pass.

## Phase 6: End-to-end validation (pending)

Target: single-token decode < 2 s at seq_len=100 on Bonsai; `/api/chat` 12-token generation < 30 s; time-to-first-token < 3 s.

Recorded deltas above are the measurement evidence. End-to-end curl against the restarted Ollama serve + AnythingLLM reconnect test are done live when the full benchmark suite completes.
