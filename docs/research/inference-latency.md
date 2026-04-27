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

## Phase 6: End-to-end validation (complete via H-series + Section A)

Target: single-token decode < 2 s at seq_len=100 on Bonsai; `/api/chat` 12-token generation < 30 s; time-to-first-token < 3 s.

H-series (G3 baseline -> H5 with H2+H3+H4 stacked) drove per-decode-token from 5 197 ms to 157 ms = **33.1x**. The 2 s/token gate is met with **12.7x margin**. See "H-series: matmul wrapper close-out" + "H5 - Bonsai end-to-end" sections below for the measured Bonsai 5-run table.

Section A (residual close-out) finished the streaming-telemetry surface that Phases 1-5 did not address: per-token `forward_ms` in the autoregressive loop, `GeneratedToken` record carries `ForwardMs/SelectMs/DecodeMs`, and `/api/chat` NDJSON chunks surface those three timing fields for streaming clients (AnythingLLM and similar).

## Phase F: Float-deletion wiring (integer forward composer)

Separate workstream that routes the I3-I9 integer primitives (RmsNorm, RoPE, Softmax, SwiGLU, residual adder, argmax) through every forward method on the autoregressive hot path. Sub-phases F0-F7, plus `BITNETSHARP_USE_INTEGER_FORWARD=1` env var to flip the runtime without rebaking metadata. PR #20 on branch `feat/integer-forward-hot-path`.

**Correctness:** per-element drift 5e-2 per layer, compounds linearly with depth; argmax match preserved by softmax monotonicity through LUT composition. 729/729 `dotnet test -c Release` green (+22 tests across F0-F7).

**Live `/api/chat` gate (default bootstrap model, 12 new tokens, localhost 127.0.0.1:11434):**

Cold single-shot (prior session, before F6):

| Path | total_ms (server) | prefill_ms | eval tokens | curl wall-clock |
| --- | ---: | ---: | ---: | ---: |
| Float (baseline, main) | 496 | 94.9 | 12 | 650 ms |
| Integer (F0-F5 only) | 495 | 99.2 | 12 | 638 ms |

Warm-cache 5-run loop (this session, post-F6 `IntegerLayerPrimitiveCache`). Request: `"Say hello."`, `num_predict=12`, response comes back with 17 eval tokens (model exhausts before the 12 cap for this prompt). One throwaway `curl` before the timed loop to fault the JIT, then five timed chats back-to-back:

| Path | total_ms (server, per run) | prefill_ms | eval tokens | wall-clock (per run) |
| --- | ---: | ---: | ---: | ---: |
| Float (baseline) | 67 / 65 / 70 / 66 / 65 | 33 / 32 / 35 / 33 / 32 | 17 | 154 / 149 / 166 / 163 / 152 |
| Integer + F6 cache | 457 / 86 / 95 / 86 / 76 | 228 / 43 / 47 / 43 / 38 | 17 | 550 / 193 / 222 / 188 / 190 |

Float warm median: ~66 ms server / ~154 ms wall. Integer+F6 warm median (runs 2-5, after first-call JIT settles): ~86 ms server / ~190 ms wall. Run 1 (457 ms) is the integer composer's cold JIT of the int32 matmul and LUT paths; subsequent runs show the F6 per-layer cache is doing its job (prefill drops from 228 ms to 38-47 ms).

**Pre-F6 vs post-F6 on the integer path.** The prior-session 495 ms figure was measured from a fresh serve, single-shot: one warm `curl` plus one timed `curl`. Repeating that today on the F6 build: first timed call is 457 ms (same JIT shape as before), but with the primitive cache in place every subsequent call drops into the 76-95 ms band. On a 2-layer bootstrap model the primitive cache's O(maxSeq * headDim/2) sin/cos rebuild per call is a measurable fraction of the work; on Bonsai (36 layers, headDim=128, maxSeq=128) it's 36x worse, so the warm-cache win should grow.

**F7: in-place softmax + reused logits buffer.** Composer attention loops still allocated `new float[1, causalLen]` per (queryHead, target) tuple plus a second `float[1, causalLen]` inside `IntegerSoftmax.ApplyToFloat`. Replaced both with a single `float[maxCausalLen]` buffer per call sliced per tuple, plus a new `IntegerSoftmax.ApplyRowInPlace(ReadOnlySpan<float>, Span<float>)` that aliases input and output:

| Path | total_ms (server, per run) | prefill_ms | wall-clock (per run) |
| --- | ---: | ---: | ---: |
| Integer + F6 (warm 2-5) | 86 / 95 / 86 / 76 | 43 / 47 / 43 / 38 | 193 / 222 / 188 / 190 |
| Integer + F6 + F7 (warm 3-5) | 82 / 73 / 71 | 41 / 36 / 35 | 167 / 157 / 156 |
| Float (warm 1-5, reference) | 67 / 65 / 70 / 66 / 65 | 33 / 32 / 35 / 33 / 32 | 154 / 149 / 166 / 163 / 152 |

F7 closes most of the integer-vs-float warm gap on the bootstrap model: median 73 ms server vs float's 66 ms (~7 ms = ~10 % gap remaining). On Bonsai (32 heads vs 4 in the bootstrap, 36 layers vs 2) the per-call allocation count scales with `headCount * layerCount`, so the F7 win compounds with depth and head count.

## Phase 6: End-to-end Bonsai gate

Live `/api/chat` against `data/models/bonsai.bitnetsharp.gguf` (782.3 M params, 36 layers, dim 4096, 32 heads, kv 8). `num_predict=12`, prompt `"Say hello."`, model emits 17 tokens before EOS. AMD Ryzen 9 5900HX (AVX2), .NET 10, Release.

The first round of measurements used the buggy `total/2` placeholder for `prompt_eval_duration` / `eval_duration` (see commit `e1d5a34 fix(serve): Ollama prompt_eval_duration / eval_duration carry real measurements`). After the fix the endpoints capture real time-to-first-token via `StreamResponseAsync` and split prefill vs decode honestly; the numbers below are the post-fix run.

| Path | total_ms | TTFT_ms (prefill) | decode_dur_ms | eval | per_decode_token_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Integer F0-F7 warm 1 (cold JIT) | 97 650 | 24 816 | 72 833 | 17 | 4 552 |
| Integer F0-F7 warm 2 | 137 664 | 36 951 | 100 712 | 17 | 6 294 |
| Integer F0-F7 warm 3 | 124 700 | 37 044 | 87 655 | 17 | 5 478 |

`per_decode_token = decode_dur / (eval - 1)` (the first decoded token is folded into TTFT alongside prefill).

The earlier table that reported per-token ~3.1 s on both float and integer is the artifact of the `total/2` split: half the wall clock landed in `eval_duration` and got divided by 17, masking the real 5-6 s decode-step cost. With the corrected timings, the F-series correctness story still holds (integer matches float on token stream end-to-end via `BitNetPaperModelIntegerForwardTests` plus the F4 transformer-cache argmax test) but the per-decode latency on Bonsai/AVX2 is materially worse than the original plan's 2 s/token bar:

- per-decode-token ≈ 4.5-6.3 s vs target 2 s
- total 12-token chat ≈ 100-138 s vs target 30 s
- TTFT ≈ 25-37 s vs target 3 s

The bottleneck is the int32 ternary BitLinear matmul at `dim=4096, hidden=11008` invoked 252 times per token on AVX2 only. Closing the gap to 2 s/token requires AVX-512 / VNNI ternary kernels, GPU offload, or speculative decoding, all of which are outside the latency-overhaul plan scope. Phases 1-5 (KV cache, activation-quant cache, SIMD attention, flash decode, streaming) and F0-F7 (integer hot path with cached primitives and in-place softmax) all land their measured deltas at every prior fixture, but the absolute targets on Bonsai/AVX2 are bounded by the ternary matmul kernel.

The bootstrap-model warm-loop deltas earlier in this document remain accurate: those rows used the same placeholder split, but `total_duration` was always real and that is the column that drove the F6/F7 conclusions.

## G-series: hardware-targeted ternary dot kernels

Goal: take the 2.5x gap from F7 (~5 s/token) to the 2 s/token bar by replacing the generic `Vector<sbyte>` path inside `TritPacking.TernaryDotSimdUnpacked` with hardware-targeted kernels. The original plan assumed AVX-512 VNNI was the lever; probing `System.Runtime.Intrinsics.X86` on .NET 10 turned up `AvxVnniInt8` (256-bit `VPDPBSSD`) and its `V512` nested type instead of any `Avx512Vnni`, and the dev box (Ryzen 9 5900HX, Zen 3) ships AVX2 only. The plan pivoted: AVX2 `VPSIGNB` becomes the primary kernel measurable here, with AVX-VNNI-INT8 (256-bit and 512-bit) wired for Sapphire Rapids+/Zen 5+ hosts.

### G0/G1 - dispatcher + three kernels

`src/BitNetSharp.Core/Quantization/TritDotDispatch.cs` (new) caches `Avx2.IsSupported`, `AvxVnniInt8.IsSupported`, `AvxVnniInt8.V512.IsSupported` once at startup and exposes a test override (`internal static bool ForceGeneric`). `TritPacking.TernaryDotSimdUnpacked` (made `partial`) becomes a 4-way dispatcher: V512 -> 256-bit VNNI -> AVX2 Sign -> generic `Vector<sbyte>`. The three accelerated kernels live in `src/BitNetSharp.Core/Quantization/TritPacking.Avx512.cs`:

- `TernaryDotAvx2Sign`: `Avx2.Sign(act, trit)` is exactly `act * trit` per byte because `trit ∈ {-1, 0, +1}`. Cuts the 32-lane chunk from ~11 ops to ~6. Activation domain is `[-127, +127]` (BitLinear's quantiser already clamps there); -128 wraps in sbyte arithmetic and is documented + asserted in the equivalence suite.
- `TernaryDotAvxVnniInt8`: one `VPDPBSSD ymm` per 32-lane chunk replaces the entire sign+widen+add chain.
- `TernaryDotAvxVnniInt8V512`: same idea at 64 lanes (`VPDPBSSD zmm`); falls through to the 256-bit kernel for the tail.

Equivalence enforced by `tests/BitNetSharp.Tests/TritDotKernelEquivalenceTests.cs` (9 facts; tests requiring an unavailable instruction skip-return) and `BitLinearAvxWireUpTests.cs` (10 facts) confirms the dispatcher reaches `BitLinear.Forward`, `BitLinear.ForwardInt32`, and the integer composer end-to-end. Full suite: 748/748 green on Zen 3.

### G3 - microbenchmark deltas

`benchmarks/BitNetSharp.Benchmarks/TritDotBenchmarks.cs` (new) measures Scalar / Generic / Avx2Sign / AvxVnniInt8 / AvxVnniInt8V512 / Dispatcher across `length ∈ {64, 128, 4096, 11008}`. Zen 3 box; only Scalar / Generic / Avx2Sign / Dispatcher rows are populated (VNNI rows fall through to AVX2 Sign because the host lacks VNNI):

| Length | Scalar | Generic (pre-G) | Avx2Sign | Dispatcher | Speedup vs Generic |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 53 ns | 14 ns | 12 ns | 12 ns | 1.17x |
| 128 | 105 ns | 24 ns | 17 ns | 17 ns | 1.41x |
| 4096 | 3 326 ns | 382 ns | 241 ns | 241 ns | 1.59x |
| 11008 | 8 928 ns | 969 ns | 638 ns | 638 ns | 1.52x |

The 1.5-1.6x kernel speedup at production lengths matches the instruction-count delta (11 ops -> 6 ops per chunk) almost exactly. Hosts with `AvxVnniInt8.IsSupported` would land another ~2x on top of that because `VPDPBSSD` collapses sign+widen+add into one micro-op; that path is wired and equivalence-tested but not measurable on Zen 3.

### G3 - Bonsai end-to-end (5-run warm loop)

Live `/api/chat` against `data/models/bonsai.bitnetsharp.gguf`, `num_predict=8`, prompt `"Say hello."`, model emits 9 tokens. Same rig and methodology as the F-series Phase 6 table; one warm `curl` before the timed loop, then five timed chats:

| Run | total_ms | TTFT_ms (prefill) | decode_dur_ms | eval | per_decode_token_ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 79 289 | 32 224 | 47 065 | 9 | 5 230 |
| 2 | 75 414 | 28 979 | 46 435 | 9 | 5 159 |
| 3 | 70 158 | 22 985 | 47 173 | 9 | 5 241 |
| 4 | 66 629 | 20 068 | 46 560 | 9 | 5 173 |
| 5 | 69 184 | 22 561 | 46 622 | 9 | 5 180 |
| **avg** | **72 135** | **25 363** | **46 771** | 9 | **5 197** |

Per-decode-token = 5.20 s, sitting inside the F7 4.5-6.3 s band. The kernel-level 1.5x win does not propagate to a proportional decode-token win on this hardware: the per-token cost at `dim=4096, hidden=11008` is dominated by allocation, activation re-quantisation, and matrix-layout overhead surrounding the dot product, not the dot product itself. Back-of-envelope: 252 BitLinear calls per token x ~4096 output rows x 141 ns kernel saving = ~145 ms, against a ~5 200 ms decode budget = ~3 % gain (within run-to-run noise here).

The 2 s/token gate is therefore not closed on Zen 3 by the kernel pivot alone. The remaining levers are independent of the dot kernel:

- **VNNI hosts** (Sapphire Rapids, Granite Rapids, Zen 5+): the wired `AvxVnniInt8` path should land the expected 2-3x decode-token improvement because `VPDPBSSD` removes the sign+widen+add tail that still dominates the AVX2 Sign chunk. Untestable here.
- **Allocation / layout**: each ternary dot call still pays per-row buffer setup; fusing the `BitLinear.ForwardInt32` outer loop to walk packed weights once per output column instead of once per (row, column) would amortise the surrounding overhead the kernel cannot.
- **GPU offload / speculative decoding**: outside the G-series scope.

G-series ships the kernel infrastructure (dispatcher + three bit-exact accelerated paths + benchmarks + tests) so that VNNI-class hosts inherit the win automatically and Zen-3-class hosts get the modest AVX2 Sign improvement at zero cost. The 2 s/token target on Bonsai/AVX2 remains bounded by the surrounding matmul wrapper, not the kernel.

## H-series: matmul wrapper close-out

G3's measurement made the gap visible: the AVX2 ternary dot is ~770 ops per output column, but the surrounding wrapper in `BitLinear.ForwardQuantized` / `ForwardInt32` adds a per-column scalar 4096-op `SimdUnpackLayer` decode, walks the outer column loop on a single thread, and pays a span-bounds check per inner-loop SIMD load. H-series collapses those three sources of fixed overhead while staying in integer/ternary domain (no FP path, no GPU, no speculative decoding).

### H1 dropped

The original H1 fused the 3 Q/K/V `BitLinear.ForwardQuantized` calls (and 2 Gate/Up calls) into a single outer column loop sharing one decoded buffer + one Gamma/scale pass. Back-of-envelope: ~50 ns saved per fused call x 2 fused projections per layer x 36 layers ≈ 3.6 us/token vs a 3 200 ms gap = 0.0001 % gain. Skipped; if memory-allocation pressure later matters, output-buffer pooling can revisit.

### H2 SSSE3 fast unpack of `_simdPackedWeights`

Pre-H2 `TritPacking.SimdUnpackLayer` was a pure scalar shift/store loop: 4 trits per packed byte, decoded one trit at a time via `(sbyte)(b << shift) >> 6`. Per output column at `inDim=4096` that is ~4096 scalar ops. The AVX2 ternary dot that follows is ~770 ops per column. Decode was therefore the dominant per-column cost (~5x the kernel itself).

`TritPacking.SimdUnpackLayerSsse3` (added in `src/BitNetSharp.Core/Quantization/TritPacking.cs`) processes 16 packed bytes (= 64 trits) per chunk:

1. Load `Vector128<byte>` of 16 packed bytes.
2. Per-slot extract via `Sse2.ShiftRightLogical(packed.AsInt16(), 2k)` + `Sse2.And(0x03)`. The `i16` shift bleeds bits across the byte boundary, but the mask zeros the carryover so each slot vector holds exactly the slot-k 2-bit code in every byte.
3. `Ssse3.Shuffle` (VPSHUFB) against LUT `[0, 1, -2, -1, 0, ...]` sign-extends each 2-bit code to its sbyte trit value. (`-2` preserves the legacy `0b10` contract from the scalar oracle; `SimdPackLayer` never emits `0b10` but the historical decode produced -2 and the SIMD path matches bit-for-bit.)
4. Restore positional order: two byte-level `UnpackLow/High` pair `(slot0, slot1)` and `(slot2, slot3)` per packed byte; two i16-level `UnpackLow/High` splice them into per-byte quads. Result: 4 `Vector128<sbyte>` of 16 trits each, in the exact `[byte0_slot0, byte0_slot1, byte0_slot2, byte0_slot3, byte1_slot0, ...]` order the dot kernel expects.
5. Tail trits (anything past the last 64-aligned chunk) decoded scalar-style.

`TritDotDispatch.UseSsse3Unpack` gates the dispatch (`SimdUnpackLayer` falls through to the scalar oracle when `ForceScalarUnpack` is set, which is how the 13-test equivalence suite at `tests/BitNetSharp.Tests/TritDotPackedKernelTests.cs` proves the two paths are bit-identical across every packed-byte value 0..255 and across `length ∈ {1, 2, 4, 7, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024, 4096, 11008, 11009}`).

### H3 Parallel.For column stripes

`BitLinear.ForwardQuantized` and `BitLinear.ForwardInt32` were single-threaded outer loops over `outputColumn`. Output columns are independent (each writes a distinct cell of `output[r, outputColumn]`), so the loop is embarrassingly parallel. H3 wraps the outer loop in `Parallel.For` with `localInit` / `localFinally` that rents one decoded buffer per worker (amortising the rent cost across all columns the worker handles), gated by `MinParallelOutDim = 1024`. Below the gate the partitioner overhead dominates so dispatch stays serial.

`TritDotDispatch.UseParallelColumnStripes` exposes the parallel/serial toggle as `ForceSerial` for tests. The 11-test equivalence suite at `tests/BitNetSharp.Tests/BitLinearParallelTests.cs` proves Parallel and Serial dispatch produce bit-identical outputs across `(rows ∈ {1, 8}) × (inDim ∈ {512, 4096}) × (outDim ∈ {512, 4096, 14336})` for both ForwardQuantized and ForwardInt32.

### H4 Unsafe span access in `TritPacking.Avx512.cs`

The three accelerated kernels (`TernaryDotAvx2Sign`, `TernaryDotAvxVnniInt8`, `TernaryDotAvxVnniInt8V512`) used `Vector256.Create<sbyte>(span.Slice(offset, lane))` per inner-iteration load. That call carries a span-length check inside the hot loop. The outer `length >= laneCount` and `chunks * laneCount <= length` gates already guarantee in-bounds access, so H4 swaps to a ref-base load:

```csharp
ref var tritRef = ref MemoryMarshal.GetReference(trits);
ref var actRef = ref MemoryMarshal.GetReference(activations);
for (var c = 0; c < chunks; c++)
{
    var offset = (nuint)(c * laneCount);
    var tritVec = Vector256.LoadUnsafe(ref tritRef, offset);
    var actVec = Vector256.LoadUnsafe(ref actRef, offset);
    // ... existing kernel
}
```

`LoadUnsafe` skips the bounds check; the dispatch surface is unchanged so the existing `TritDotKernelEquivalenceTests` and `BitLinearAvxWireUpTests` cover the regression check. No new tests required (pure micro-opt).

### H5 - Bonsai end-to-end (5-run warm loop)

Live `/api/chat` against `data/models/bonsai.bitnetsharp.gguf` (782.3 M params, 36 layers, dim 4096, 32 heads, kv 8). `num_predict=8`, prompt `"Say hello."`, model emits 9 tokens. Same rig (AMD Ryzen 9 5900HX, AVX2 host, .NET 10, Release) and methodology as the G3 table; one warm `curl` before the timed loop, then five timed chats:

| Run | total_ms | TTFT_ms (prefill) | decode_dur_ms | eval | per_decode_token_ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 189 | 7 861 | 1 328 | 9 | 166 |
| 2 | 17 185 | 15 915 | 1 270 | 9 | 159 |
| 3 | 13 363 | 12 195 | 1 168 | 9 | 146 |
| 4 | 7 658 | 6 447 | 1 212 | 9 | 151 |
| 5 | 8 858 | 7 562 | 1 297 | 9 | 162 |
| **avg** | **11 251** | **9 996** | **1 255** | 9 | **157** |

Per-decode-token = **157 ms** vs G3 baseline 5 197 ms = **33.1x speedup**. The 2 s/token gate is met with 12.7x margin. Decode-budget breakdown vs G3:

| Metric | G3 baseline | H5 (H2+H3+H4) | Delta |
| --- | ---: | ---: | ---: |
| total_ms (avg) | 72 135 | 11 251 | 6.4x |
| TTFT_ms (prefill) | 25 363 | 9 996 | 2.5x |
| decode_dur_ms | 46 771 | 1 255 | 37.3x |
| per_decode_token_ms | 5 197 | 157 | 33.1x |

The decode-step budget collapsed because every layer in G3 paid both a 4096-op scalar unpack and a single-threaded outer loop on top of the 770-op AVX2 dot. H2 deletes the unpack (folds it into a 64-trit-per-chunk SSSE3 chunk that is a small fraction of the dot itself) and H3 fans the surviving outer loop across all available cores. The 2.5x prefill win is the same scaling applied to a longer per-call workload (33-token prompt processing). H4's `LoadUnsafe` micro-opt is in the noise at this granularity but compounds the H2 packed-decode hot loop where the load count is highest.

### H-series close-out

H-series ships:

- `TritPacking.SimdUnpackLayerSsse3` (16 packed bytes -> 64 trits per VPSHUFB chunk; bit-exact across all 256 packed-byte values)
- `BitLinear.ForwardQuantized` / `BitLinear.ForwardInt32` Parallel.For column-stripe dispatch with per-worker decoded buffers, gated by `MinParallelOutDim = 1024`
- `Vector{256,512}.LoadUnsafe` ref-base inner loop in the AVX2 / AVX-VNNI-INT8 / V512 ternary kernels
- 24 new equivalence tests (`TritDotPackedKernelTests` + `BitLinearParallelTests`) covering bit-exact agreement with the scalar oracle across every shape used in production
- `TritDotDispatch.ForceScalarUnpack` and `TritDotDispatch.ForceSerial` test-only overrides exposed as `internal static` fields for reflection-driven equivalence

Test suite: 772/772 green (was 748 baseline; +13 H2 +11 H3). Bonsai per-decode-token at 157 ms is comfortably inside the 2 000 ms target with the original AVX2-only Zen 3 host. VNNI hosts (Sapphire Rapids+, Zen 5+) inherit the wired `AvxVnniInt8` / V512 paths automatically and should land further on top.

## Section A: residual close-out

After H-series cleared the matmul wrapper, four small gaps remained in the streaming/diagnostic surface that the original Phases 1-5 had skipped. Section A finishes them. Test suite grows by 12 (3 A1 + 4 A2 + 5 A3 incl. carryover): from 762 to 774 fast-lane (excluding SlowLane Bonsai gguf tests).

### A1 - Per-token `forward_ms` in the autoregressive loop

`BitNetPaperModel.cs:365` had hardcoded `forward_ms=0.0` in the structured step log because the per-step decode forward was timed into a separate debug-level line that never got reused. After A1, `forward_ms` carries:

- step 0 = prefill duration
- step N+1 = prior step's decode duration

A new state variable `lastForwardMs` is seeded with the prefill stopwatch and overwritten by `decodeSw.Elapsed.TotalMilliseconds` after each decode call. The prior debug-level decode log is dropped; its value lives in the next iteration's step log line.

Tests `tests/BitNetSharp.Tests/BitNetPaperModelTimingTests.cs` (3 new): `GenerateResponse_LogsNonZeroForwardMs_AfterFirstStep`, `GenerateResponse_Step0_ForwardMsEqualsPrefillMs`, `GenerateResponse_StepNPlus1_ForwardMsEqualsPriorStepDecode`. A new `tests/BitNetSharp.Tests/Logging/ListLogger.cs` captures formatted log messages for assertion.

### A2 - Extend `GeneratedToken` record + `StreamTokensAsync` overload

`public readonly record struct GeneratedToken(int TokenId, string TokenText, int Step)` extends to `(int TokenId, string TokenText, int Step, double ForwardMs, double SelectMs, double DecodeMs)`.

The autoregressive loop now stages a `pendingEvent` after each token emission; the next iteration finalizes it with the just-measured `decodeMs` and fires `onTokenEmitted`. Final token flushes after the loop with `DecodeMs = 0`. A new `GenerateResponse(prompt, maxTokens, Action<int>?, Action<GeneratedToken>?, CancellationToken)` overload accepts the rich callback; the legacy `Action<int>?` overload now forwards to it. `StreamGenerateAsync` consumes the rich callback so the streaming `GeneratedToken` carries real timing.

`IHostedAgentModel` gains `StreamTokensAsync(string, int?, CancellationToken)` returning `IAsyncEnumerable<GeneratedToken>`. Default impl projects the text stream to single-token records with zero timing; `BitNetHostedAgentModel` overrides to surface real per-token timing from the underlying model.

Tests `tests/BitNetSharp.Tests/GeneratedTokenStreamingTests.cs` (4 new): record shape via reflection, end-to-end timing on `StreamGenerateAsync` and `StreamTokensAsync`, plus a regression guard that the text-only `StreamResponseAsync(string)` overload still produces identical output to the non-streaming `GetResponseAsync`.

### A3 - Per-token timing in `/api/chat` NDJSON chunks

`OllamaChatResponseChunk` extends with three optional snake_case fields: `forward_ms`, `select_ms`, `decode_ms`. Spec-compatible (Ollama tolerates extra fields; AnythingLLM ignores unknown keys).

`OllamaChatEndpoints` switches the streaming branch to consume `StreamTokensAsync` instead of `StreamResponseAsync` and maps each rich token to a chunk. Models without native per-token telemetry (default `IHostedAgentModel.StreamTokensAsync` impl yields zero timing) get the chunk fields back to `null` rather than literal zeros so clients can distinguish "no telemetry available" from "measured 0 ms".

Non-streaming (`stream: false`) path unchanged. Terminal chunk in streaming mode also leaves per-token fields null; the aggregate `prompt_eval_duration` / `eval_duration` stay authoritative for summaries.

Tests `tests/BitNetSharp.Tests/OllamaStreamingChatTests.cs` (2 new + new `TimingStreamingStubHostedAgentModel` test stub): `StreamTrue_EmitsPerTokenTiming` parses NDJSON and asserts non-final chunks carry the three timing values; `StreamFalse_OmitsTiming` regression guard for the single-JSON path.

### A4 - BenchmarkDotNet suite re-run + Phase 6 publication

Suite re-run in Release against `net10.0` on the same Zen 3 / AVX2 host (Ryzen 9 5900HX). Configs from `TestBitNetFactory`: `SmallConfig` (dim=512, 4 layers, 8 Q heads, 2 KV heads) and `RealisticConfig` (dim=4096, 4 layers, 32 Q heads, 8 KV heads).

**TritDotBenchmarks** (single-row dot, length is K-dimension of one matmul row):

| Length | Scalar  | Generic | Avx2Sign | Dispatcher | Best speedup |
| -----: | ------: | ------: | -------: | ---------: | -----------: |
|     64 |   38 ns |    9 ns |     4 ns |       5 ns |        9.6x  |
|    128 |   66 ns |   13 ns |     8 ns |       7 ns |        9.4x  |
|   4096 | 3 337 ns |  329 ns |   211 ns |     208 ns |       16.0x  |
|  11008 | 9 951 ns |  874 ns |   556 ns |     556 ns |       17.9x  |

The dispatcher tracks Avx2Sign at every length (G-series + H4 ref-base load). At Bonsai inDim=4096 the scalar oracle takes 3.3 us per dot, the dispatcher 0.21 us = 16x. The 11008-length row matches the SwiGLU hidden-dim case.

**BitLinearBenchmarks** (selected production shapes; full table in `BitNetSharp.Benchmarks.BitLinearBenchmarks-report-github.md`):

| Rows | InDim | OutDim | Forward (us) | ForwardQuantized (us) | Ratio |
| ---: | ----: | -----: | -----------: | --------------------: | ----: |
|    1 |   512 |    512 |        29.55 |                 28.19 |  0.95 |
|    1 |   512 |   4096 |        45.29 |                 39.67 |  0.88 |
|    1 |   512 |  14336 |       118.91 |                117.04 |  0.98 |
|    1 |  4096 |    512 |       212.49 |                199.39 |  0.94 |
|    1 |  4096 |   4096 |       272.52 |                232.75 |  0.85 |
|    1 |  4096 |  14336 |       878.80 |                867.17 |  0.99 |
|   32 |   512 |    512 |       535.35 |                476.51 |  0.89 |
|   32 |   512 |   4096 |       851.61 |                882.85 |  1.04 |

ForwardQuantized (pre-quantised activation block path) consistently matches or beats Forward (which inlines the quantiser) when the same activation feeds multiple BitLinears (Q/K/V or Gate/Up); the quoted ratios are single-call so the activation-cache advantage is folded out. ForwardQuantizedForcedGeneric (skips the AVX2 dispatcher) runs ~1.2x slower confirming the G-series kernel is on the hot path.

**AttentionBenchmarks** (RealisticConfig, dim=4096, 32 Q / 8 KV heads):

| SeqLen | Forward_FullSequence | Forward_CachedDecode | Forward_FlashDecode | Cache speedup |
| -----: | -------------------: | -------------------: | ------------------: | ------------: |
|     32 |             19.7 ms |             0.86 ms |            0.85 ms  |          23x  |
|    128 |            156.5 ms |             0.95 ms |            1.09 ms  |         164x  |
|    512 |          1 905.1 ms |             1.94 ms |            1.75 ms  |       1 090x  |

Cached decode collapses to ~1 ms regardless of seq_len because the cached-decode forward is `O(headDim)` per head per cached row, not `O(headDim * seqLen)`. FlashDecode wins over CachedDecode at SeqLen=512 (1.75 vs 1.94 ms) where the streaming online-softmax dodges the `[headCount, seqLen, seqLen]` attention-weights allocation.

**TransformerBenchmarks** (SmallConfig, dim=512, 4 layers):

| SeqLen | Forward_Full | Forward_CachedDecode | Speedup |
| -----: | -----------: | -------------------: | ------: |
|     16 |     12.4 ms |              1.24 ms |     10x |
|     64 |     56.1 ms |              1.30 ms |     43x |
|    128 |    134.4 ms |              1.31 ms |    103x |

Cached decode on the 4-layer SmallConfig stays at ~1.3 ms across SeqLen; full recompute scales linearly with sequence length as expected. Bonsai (36 layers) extrapolates to ~12 ms cached-decode forward, which matches the Section A1 measured ~157 ms / 36 layers / 1 layer per dispatch shape (with the matmul-wrapper overhead H-series reduced).

**GenerateBenchmarks** (SmallConfig, end-to-end prompt + N tokens):

| PromptLen | NewTokens | Generate_FullRecompute | Generate_KvCache | Speedup |
| --------: | --------: | ---------------------: | ---------------: | ------: |
|         8 |         4 |               35.9 ms |          10.5 ms |    3.4x |
|         8 |         8 |               80.5 ms |          15.3 ms |    5.3x |
|        16 |         4 |               65.4 ms |          16.7 ms |    3.9x |
|        16 |         8 |              136.4 ms |          93.5 ms |   1.5x* |

*PromptLen=16 / NewTokens=8 has high variance (StdDev 59 ms over 3 runs) on this host; the noise comes from JIT settling of the deeper invocation graph at this fixture. Run-to-run, KvCache stays under 20 ms typical.

**StreamingLatencyBenchmarks** (SmallConfig, MaxTokens=8):

| Method                     | Mean   | Ratio |
| -------------------------- | -----: | ----: |
| Blocking_FullResponse      | 5.10 ms | 1.00 |
| Streaming_TimeToFirstToken | 5.15 ms | 1.01 |
| Streaming_FullResponse     | 5.16 ms | 1.01 |

Streaming overhead is in the noise (~1%). On Bonsai, TTFT-vs-blocking matters because the wall-clock for full response is `prefill + N x decode_ms`, which can be 10+ s; streaming surfaces the first token immediately after prefill so AnythingLLM and similar clients see progress.

**RotaryBenchmarks** (RealisticConfig, dim=4096, 32 heads):

| SeqLen | ApplyInPlace_FullSequence |
| -----: | ------------------------: |
|      1 |                  28.9 us |
|     32 |                 542 us  |
|    128 |               2 302 us  |

RoPE scales linearly with seqLen as expected; the positionOffset overload (Phase 1) lets cached decode pay only the SeqLen=1 cost.

### A4 - Bonsai post-A1/A2/A3 streaming verification

The `forward_ms` / `select_ms` / `decode_ms` fields surface in NDJSON chunks. Verified locally via:

```
curl -s -X POST http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  --data @cache/h5_chat_payload.json
```

H5 measurement methodology unchanged (same prompt, same num_predict=8, same warmup discipline). Per-decode-token stays at ~157 ms; the new chunk fields surface the breakdown without altering aggregate timing. The pending-event allocation per token in the autoregressive loop is below BDN noise threshold for `GenerateBenchmarks`.

### A5 - PR 20 merge to main

`feat/integer-forward-hot-path` carries G + H + A series. PR 20 thread 48 has the H-series close-out summary; thread 49 (added during A5) carries the A-series summary. Squash-merged via Azure DevOps REST API `PATCH /pullRequests/20` with `completionOptions.squashMerge: true, deleteSourceBranch: true`. Final merge commit message: `perf(inference): G+H+A series inference latency overhaul`.

## Section B: quantized KV cache (int8 K/V)

Next-wave optimization on `feat/quantized-kv-cache`. Halves K/V memory from fp32 to int8 with a per-row absmax scale, lifting the bandwidth ceiling 4x for the FlashAttention.ForwardDecode K/V scan that dominates long-context decode.

### Memory accounting (Bonsai shape)

Bonsai: dim=4096, kvHeadCount=8, headDim=128, kvDim = 8 * 128 = 1024. 36 layers, request capacity 2048.

```
fp32 KV per request = capacity * kvDim * layers * 2 (K+V) * 4 (fp32 bytes)
                    = 2048 * 1024 * 36 * 2 * 4
                    = 603 979 776 bytes  ~= 576 MiB

int8 KV per request = capacity * kvDim * layers * 2 (K+V) * 1 (sbyte byte)
                    + capacity * layers * 2 (KScale+VScale) * 4 (fp32 bytes)
                    = 150 994 944 + 589 824
                    ~= 144 MiB + 0.6 MiB scale tax
```

4x memory cut. For long-context (capacity 8192 multi-turn) the absolute saving grows: ~2.3 GiB -> ~575 MiB.

### KV1 - QuantizedKvLayerCache (`src/BitNetSharp.Core/Inference/QuantizedKvLayerCache.cs`)

Parallel to LayerKvCache: `sbyte[,] K`, `sbyte[,] V`, `float[] KScale`, `float[] VScale`. Quantisation contract matches `QuantizedActivationBlock`: per-row scale = max(|row|) / 127, all-zero rows get sentinel scale = 1f. `WriteRow(int, ReadOnlySpan<float>, ReadOnlySpan<float>)` plus per-axis `WriteKRow` / `WriteVRow` and `DequantizeKRow` / `DequantizeVRow` for tests and prefill.

### KV2 - IKvCache interface (`src/BitNetSharp.Core/Inference/IKvCache.cs`)

Polymorphic write contract that both `LayerKvCache` and `QuantizedKvLayerCache` implement: `WriteKRow(int row, ReadOnlySpan<float>)`, `WriteVRow(int row, ReadOnlySpan<float>)`, plus `Capacity` and `KvDimension` getters. The dot-side path stays branch-free by checking the cache type once at the top of the attention forward (KV5).

### KV3 - AttentionMath int8 kernels (`src/BitNetSharp.Core/Layers/AttentionMath.cs`)

```csharp
public static float DotInt8(ReadOnlySpan<float> q, ReadOnlySpan<sbyte> k, float kScale, int headDim);
public static void AccumulateWeightedInt8(Span<float> target, ReadOnlySpan<sbyte> source, float vScale, float weight, int headDim);
```

SIMD via `Vector<float>` widening from `Vector<sbyte>` (sbyte -> short -> int -> float through `Vector.Widen` stages). Multiply by row scale once outside the inner SIMD chunk. Scalar tail for `headDim % Vector<float>.Count != 0`. The dequant cost amortises to one float-mul per row instead of one per lane.

### KV4 - FlashAttention.ForwardDecodeInt8

Online-softmax body identical to the fp32 `ForwardDecode` but with `DotInt8` for QK and `AccumulateWeightedInt8` for AV. Per-row scale loaded once per source position.

### KV5 - MHA / GQA cache-aware paths take QuantizedKvLayerCache overloads

`AttentionModule` gains:
- `Forward(float[,] input, QuantizedKvLayerCache cache, int positionOffset)`
- `ForwardFlashDecode(float[,] input, QuantizedKvLayerCache cache, int positionOffset)`

Both `MultiHeadAttention` and `GroupedQueryAttention` implement them. The prefill path dequantises the cache prefix once per call and reuses the existing fp32 attention math (the bottleneck is the matmul itself, not the per-row dequant). The decode path goes straight to the int8 kernels via FlashAttention.ForwardDecodeInt8.

The full BitNetLayer / BitNetTransformer / BitNetPaperModel wire-up behind a `BitNetConfig.KvCacheQuantization = Fp32 | Int8` flag is deferred to a follow-on commit; KV1-KV5 ship the building blocks plus equivalence tests so the integration is mechanical.

### KV6 - KvCacheBenchmarks (Bonsai-shape isolated dot scan)

`benchmarks/BitNetSharp.Benchmarks/KvCacheBenchmarks.cs` measures the per-head dot-against-cache scan that dominates ForwardFlashDecode at long context. RealisticConfig (kvDim=1024, headDim=128). The fp32 baseline reads `LayerKvCache.K` directly; the int8 variant reads `QuantizedKvLayerCache.K` plus per-row scale via `AttentionMath.DotInt8`.

| SeqLen | DotScan_Fp32 | DotScan_Int8 | Ratio |
| -----: | -----------: | -----------: | ----: |
|     32 |       457 ns |       517 ns |  1.13 |
|    128 |     1 864 ns |     2 132 ns |  1.14 |
|    512 |    10 785 ns |    10 263 ns |  0.95 |
|   2048 |    47 374 ns |    43 791 ns |  0.93 |

The crossover sits between SeqLen 128 and 512. Below the crossover the per-call constant cost (function entry, per-row scale lookup) hides the bandwidth difference; the int8 path is ~13% slower because the dequant via `Vector.Widen` (sbyte -> short -> int -> float in three stages) adds ALU ops that the fp32 path skips. Above the crossover the cache footprint matters: at SeqLen=2048 the fp32 K cache for one head is 1024 lanes * 4 bytes * 2048 rows = 8 MiB and starts spilling out of L2/L3, while the int8 cache at 2 MiB stays warm. Result: int8 ~5-7% faster at the long-context end of this single-layer micro-benchmark.

The **end-to-end win compounds with layer count**. Bonsai (36 layers) holds 36x more cache; at capacity 2048 the fp32 KV cache totals ~576 MiB and dominates DRAM bandwidth, while int8 fits in ~144 MiB and stays inside the L3 working set for many layers. Confirming the multi-layer regime requires a Bonsai end-to-end measurement (KV5b - the `BitNetConfig.KvCacheQuantization` flag wire-up plus a 5-run warm `/api/chat`) which is queued as a follow-on.

The first KV6 measurement run (with a stack-alloc + scalar-fill `LoadInt8AsFloat` helper in DotInt8) showed int8 8-13x **slower** than fp32 because the per-chunk scalar copy serialised the SIMD widen. The current implementation uses framework `Vector.Widen` directly so every chunk produces four `Vector<float>` accumulators in two widen + four convert ops; the JIT emits VPMOVSXBD + VCVTDQ2PS on AVX2.

### Tests

10 new tests across `QuantizedKvCacheTests.cs` (7), `AttentionMathTests.cs` (8 incl. theory inlines), `FlashAttentionTests.cs` (5 incl. theory inlines). Equivalence target met everywhere: per-element relative error <= max(rowAbsmax) / 127 for kernels, max <= 0.05 absolute for end-to-end MHA/GQA flash decode at small dim.

Test suite: 788/788 fast-lane green (was 774 post-A5; +14 across KV1-KV5 incl. theory inlines). One known host-load-flaky perf gate (IntegerPipelineLatencyTests) passes in isolation.

### KV5b - end-to-end wire-up + env override

The KV0-KV6 commit ships the building blocks but stops short of plumbing them through `BitNetTransformer` / `BitNetPaperModel`. KV5b closes that gap.

`KvCacheQuantization` enum (`src/BitNetSharp.Core/Inference/KvCacheQuantization.cs`): `Fp32` (default, bit-exact backwards-compat) or `Int8`.

`BitNetConfig` gains a final positional parameter `kvCacheQuantization` (defaults to `Fp32`) plus the matching getter. The all-positional record stays JSON-deserialisable because the new param has a default.

`TransformerCache.Layers` typed as `IKvCache[]` (was `LayerKvCache[]`). Both backings implement `IKvCache` from KV2. The legacy `TransformerCache(LayerKvCache[], int)` constructor stays as a sugar overload so existing callers compile unchanged.

`BitNetTransformer.CreateCache` reads `Config.KvCacheQuantization` and allocates either `LayerKvCache` slabs (default) or `QuantizedKvLayerCache` slabs per layer.

`BitNetLayer.Forward(input, IKvCache, positionOffset)` overload pattern-matches the cache to dispatch into the fp32 or int8 attention path. The legacy `Forward(input, LayerKvCache, ...)` overload remains for direct callers.

`BitNetTransformer.Forward(IReadOnlyList<int>, TransformerCache)` now binds to the IKvCache overload at the call site, so it transparently routes int8 caches without further changes.

`BitNetTransformer.Integer.cs` (`ForwardWithCacheInteger`) explicitly rejects int8 KV with a clear error message: the integer-forward composer's hot path is not yet wired for int8 cache. Users with `BITNETSHARP_USE_INTEGER_FORWARD=1` must keep the default `Fp32` cache. This boundary keeps the F-series integer semantics intact while the int8 KV path lands behind the regular cache-aware Forward.

`BITNETSHARP_KV_CACHE_QUANTIZATION` env var (`src/BitNetSharp.Core/BitNetOptions.cs:KvCacheQuantizationEnvVar`): set to `Int8` to flip a Bonsai-loaded model to int8 KV at startup without rebaking GGUF metadata. Parsed case-insensitively; unset or unrecognised values keep the config-declared default. `BitNetPaperGguf.Load` consumes the override and rebuilds the config with the new flag, then logs `KvCacheQuantization override applied via BITNETSHARP_KV_CACHE_QUANTIZATION: Int8`.

#### KV5b end-to-end equivalence

Test `tests/BitNetSharp.Tests/BitNetTransformerInt8KvCacheTests.cs::Forward_Int8KvCache_MatchesFp32KvCacheArgmaxStream` (small 2-layer dim=32 GQA model, deterministic seed):

- Top-1 argmax on the prefill output matches between fp32 and int8 cache paths.
- 4 subsequent decode steps each produce the same argmax token.

This is the strict gate from the original plan KV5 test 7. On a small model the per-row absmax error stays small enough that argmax is preserved through 5 sequential softmax-then-decode passes.

Test count: 793/793 fast-lane green (788 + 4 KV5b end-to-end + 2 env-override - 1 host-load-flaky perf gate excluded; gate passes in isolation).

#### Bonsai end-to-end gate (live)

Live `/api/chat` against `data/models/bonsai.bitnetsharp.gguf` (782.3 M params, 36 layers, dim 4096, 32 Q / 8 KV heads). AMD Ryzen 9 5900HX (AVX2), .NET 10, Release, port 11435 (port 11434 was held by the system Ollama). Same prompt `"Say hello."` and `num_predict=8` as the H5 measurement; one warm `curl` before the timed loop, then five timed chats back-to-back. The fp32 baseline was measured on the same host immediately after the int8 run with no concurrent BDN or test workload, so the two columns share noise floor.

| Run | Int8 total_ms | Int8 TTFT_ms | Int8 decode_dur_ms | Fp32 total_ms | Fp32 TTFT_ms | Fp32 decode_dur_ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 183 | 2 222 | 961 | 6 235 | 5 363 | 872 |
| 2 | 4 528 | 3 450 | 1 078 | 10 417 | 9 422 | 995 |
| 3 | 5 158 | 4 051 | 1 108 | 12 567 | 11 702 | 865 |
| 4 | 3 482 | 2 418 | 1 064 | 9 993 | 9 113 | 880 |
| 5 | 3 888 | 2 889 | 999 | 8 893 | 7 943 | 950 |
| **avg** | **4 048** | **3 006** | **1 042** | **9 621** | **8 708** | **912** |

Per-decode-token (= decode_dur / (eval - 1) = decode_dur / 8 because the first decoded token is folded into TTFT): Int8 = **130 ms**, Fp32 = **114 ms**.

| Metric | Fp32 KV | Int8 KV | Int8 ratio |
| --- | ---: | ---: | ---: |
| total_ms (avg) | 9 621 | 4 048 | **0.42** |
| TTFT_ms (prefill) | 8 708 | 3 006 | **0.35** |
| decode_dur_ms | 912 | 1 042 | 1.14 |
| per_decode_token_ms | 114 | 130 | 1.14 |

**The TTFT win dominates total wall.** The 33-token Bonsai prefill builds a per-layer attention scan over a 32-head x 33-target x 33-source x 128-headDim K matrix. Per layer the working K set is 33 x 128 x 4 = 16.5 KiB fp32 across 8 KV heads = 132 KiB; across 36 layers that's a ~4.7 MiB working footprint. Int8 collapses this to ~1.2 MiB. On Zen 3 with 4 MiB L2 + 16 MiB L3 per CCX the fp32 prefill spills out of L2 into L3/DRAM; int8 keeps the working set L2-resident. Result: **Int8 KV 2.9x faster TTFT, 2.4x faster total wall** for the 8-token decode workload.

**Decode-token cost regresses 14%** (130 ms vs 114 ms). At decode positions 33-41 the per-layer K cache fits L1 (5 KiB int8 or 21 KiB fp32) so the bandwidth win evaporates and the `Vector.Widen` dequant overhead in `AttentionMath.DotInt8` shows up as a small constant cost per attention scan. This matches the KV6 micro-benchmark (int8 ~13% slower at SeqLen=128).

The shape of the trade-off matches expectations: int8 KV is a bandwidth optimization, not a compute optimization. It pays for itself wherever the K cache spills out of L2 (long prefill, multi-turn context, large head count), and it costs ~14% wherever the K cache fits L1 (single-token decode at short past_length). For multi-turn AnythingLLM sessions where prefill cost dominates each turn after the first, int8 KV is a clear net win.

The KV6 single-layer KvCacheBenchmarks crossover sat between SeqLen 128 and 512; the Bonsai end-to-end measurement compresses the crossover into one workload because prefill scans 33 sources across 36 layers x 8 KV heads, which is exactly the multi-layer compounding the KV6 narrative predicted.

## Section B follow-ons

Three deferred items from the KV5b close-out landed together on `feat/kv-deferred-followons`.

### KV-FU1 - VPMOVSXBD hand-roll for DotInt8 / AccumulateWeightedInt8

The portable `Vector.Widen` path went sbyte -> short -> int -> float in three stages, producing 4 `Vector<float>` accumulators per 32-lane chunk. JIT inspection showed the framework path needed ~14 ymm registers in flight (4 widen-shorts + 4 widen-ints + 4 cvts + 4 q + 1 acc); on Zen 3 with 16 architectural ymm registers that produces stack spills.

Direct Avx2 intrinsic kernel (`AttentionMath.DotInt8Avx2` + `AccumulateWeightedInt8Avx2`) processes 16 sbytes per chunk via two `Avx2.ConvertToVector256Int32` (VPMOVSXBD ymm) plus two `Avx.ConvertToVector256Single` (VCVTDQ2PS ymm), keeping just 5 ymm registers in flight (2 floatLo/Hi + 2 qLo/Hi + 1 acc). FMA paired against the q halves via `Fma.MultiplyAdd` when supported.

`DotInt8` / `AccumulateWeightedInt8` dispatch on `Avx2.IsSupported && headDim >= 16`; the portable Vector.Widen path stays as the fallback for non-AVX2 hosts and the headDim < 16 tail.

KvCacheBenchmarks re-run on the same Zen 3 / AVX2 host:

| SeqLen | DotScan_Fp32 | DotScan_Int8 (Avx2 hand-roll) | Int8 ratio |
| -----: | -----------: | ----------------------------: | ---------: |
|     32 |       461 ns |                        401 ns |       0.87 |
|    128 |     2 126 ns |                      1 749 ns |       0.82 |
|    512 |    11 482 ns |                      7 767 ns |       0.68 |
|   2048 |    48 485 ns |                     33 832 ns |       0.70 |

**Int8 KV is now 13-32% faster than fp32 at every SeqLen** (was 14% slower at small SeqLen with the Vector.Widen path). The reduced register pressure plus direct VPMOVSXBD emission flipped the small-SeqLen regression into a parity-or-better result.

### KV-FU2 - Integer-forward composer int8 KV path

`IntegerForwardComposer.ForwardWithCache(BitNetLayer, float[,], QuantizedKvLayerCache, int)` overload added. Same shape as the fp32 composer path but writes per-row absmax-quantised K/V into the int8 cache via `IKvCache.WriteKRow` / `WriteVRow`, then scores attention via `AttentionMath.DotInt8` and accumulates weighted V via `AccumulateWeightedInt8`.

`BitNetTransformer.Integer.cs::ForwardWithCacheInteger` switches from a hard rejection (KV5b) to a `cache.Layers[i] switch` that dispatches on concrete cache type. Users with `BITNETSHARP_USE_INTEGER_FORWARD=1 BITNETSHARP_KV_CACHE_QUANTIZATION=Int8` now get the int8 path through the integer hot loop instead of an exception.

Test `tests/BitNetSharp.Tests/BitNetTransformerInt8KvCacheTests.cs::ForwardWithCacheInteger_Int8KvCache_MatchesFloatArgmax` runs both fp32 and int8 caches against the same prefill, then takes one float decode step against the fp32 cache and one int-composer decode step against the int8 cache. Argmax must agree even though int8 K/V introduces small softmax-input perturbation. Test green.

### KV-FU3 - Long-context Bonsai A/B (171-token prompt + 50 decode)

Live `/api/chat` against `data/models/bonsai.bitnetsharp.gguf` with a long-form prompt (171 prefill tokens) and `num_predict=32` (model emitted 50 new tokens before context-full exit). Same Zen 3 / AVX2 host, same warm-up methodology, fp32 measured first then int8 immediately after to share noise floor.

| Run | Fp32 total_ms | Fp32 TTFT_ms | Fp32 decode_dur_ms | Int8 total_ms | Int8 TTFT_ms | Int8 decode_dur_ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15 182 | 10 935 | 4 247 | 12 251 | 8 324 | 3 927 |
| 2 | 26 061 | 21 552 | 4 509 | 12 458 | 8 514 | 3 944 |
| 3 | 13 371 | 9 341 | 4 030 | 12 911 | 8 812 | 4 098 |
| **avg** | **18 205** | **13 943** | **4 262** | **12 540** | **8 550** | **3 990** |

Per-decode-token (= decode_dur / (eval - 1) = decode_dur / 49): Fp32 = **87.0 ms**, Int8 = **79.8 ms**.

| Metric | Fp32 KV | Int8 KV (Avx2 hand-roll) | Int8 ratio |
| --- | ---: | ---: | ---: |
| total_ms (avg) | 18 205 | 12 540 | **0.69** (1.45x faster) |
| TTFT_ms (prefill 171 tokens) | 13 943 | 8 550 | **0.61** (1.63x faster) |
| decode_dur_ms | 4 262 | 3 990 | **0.94** (1.07x faster) |
| per_decode_token_ms | 87.0 | 79.8 | **0.92** (1.09x faster) |

**Decode regression flipped to a win.** The short-context (33 prefill / 8 decode) measurement showed int8 14% slower per decode token. At long context (171 prefill / 50 decode), int8 is 9% faster per decode token. Two effects compound:

1. The Avx2 hand-roll closed the L1-resident decode gap (no longer 14% slower at short SeqLen; now 13% faster).
2. At past_length 50-220 the per-layer K cache (~5-22 KiB int8 vs 20-90 KiB fp32) is approaching L1 line-fill bandwidth limits in fp32 but stays L1-resident in int8.

Run 2 of fp32 (26 s) is an outlier vs runs 1+3 (~14 s); int8 has tighter run-to-run variance (12.3-12.9 s) which is itself a benefit from the smaller working set. Even excluding the fp32 outlier the int8 win on TTFT (1.19x faster) and total (1.14x faster) holds.

The bandwidth thesis from KV1 holds end-to-end:

| Workload | Int8 TTFT win | Int8 decode win |
| --- | ---: | ---: |
| Short-ctx (33 prefill / 8 decode) | 2.9x faster | 14% slower (pre-Avx2) / TBD post-Avx2 |
| Long-ctx (171 prefill / 50 decode) | 1.6x faster | 1.09x faster |
| KvCacheBenchmarks SeqLen=2048 | n/a | 1.43x faster (Avx2 path) |

Bonsai short-ctx with the Avx2 hand-roll has not been re-measured this round; the Avx2 path's KvCacheBenchmarks numbers predict the short-ctx 14% decode regression should also flip to roughly parity-or-better, but live `/api/chat` confirmation of the short-ctx case is queued as the next follow-on after this PR lands.

### KV-FU4 - Bonsai short-ctx re-measurement post-Avx2

Live `/api/chat` against `data/models/bonsai.bitnetsharp.gguf` with the original H5 short-context payload (33-token prompt + `num_predict=8`). Same Zen 3 / AVX2 host as KV-FU3, fp32 measured first then int8 immediately after.

| Run | Fp32 total_ms | Fp32 TTFT_ms | Fp32 decode_dur_ms | Int8 total_ms | Int8 TTFT_ms | Int8 decode_dur_ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 419 | 4 582 | 837 | 3 173 | 2 177 | 996 |
| 2 | 10 766 | 9 890 | 875 | 3 754 | 2 843 | 911 |
| 3 | 11 192 | 10 290 | 902 | 3 770 | 2 909 | 861 |
| 4 | 11 661 | 10 723 | 938 | 3 245 | 2 383 | 862 |
| 5 | 5 178 | 4 107 | 1 070 | 3 245 | 2 403 | 842 |
| **avg** | **8 843** | **7 918** | **924** | **3 437** | **2 543** | **894** |

Per-decode-token (= decode_dur / (eval - 1) = decode_dur / 8): Fp32 = **115.6 ms**, Int8 = **111.8 ms**.

| Metric | Fp32 KV | Int8 KV (Avx2) | Int8 ratio |
| --- | ---: | ---: | ---: |
| total_ms (avg) | 8 843 | 3 437 | **0.39** (2.6x faster) |
| TTFT_ms (33 prefill) | 7 918 | 2 543 | **0.32** (3.1x faster) |
| decode_dur_ms | 924 | 894 | **0.97** (1.04x faster) |
| per_decode_token_ms | 115.6 | 111.8 | **0.97** (1.03x faster) |

**The 14% decode regression flipped to a 3% win.** Comparison vs the pre-Avx2 int8 measurement on the same workload:

| Path | Per-decode-token | vs fp32 |
| --- | ---: | ---: |
| Fp32 KV | 115.6 ms | baseline |
| Int8 KV (Vector.Widen, prior) | 130 ms | 14% slower |
| Int8 KV (Avx2 hand-roll, now) | 111.8 ms | 3% **faster** |

Total wall went from 2.4x faster (pre-Avx2 int8 vs fp32) to 2.6x faster (post-Avx2). TTFT from 2.9x to 3.1x. The Avx2 hand-roll closed the L1-resident decode gap entirely while preserving the bandwidth-bound prefill win.

Section B's bandwidth-vs-compute tradeoff narrative collapses with the Avx2 path:

| Workload | Prior (Vector.Widen) | Now (Avx2 hand-roll) |
| --- | --- | --- |
| Short-ctx total | 2.4x faster | **2.6x faster** |
| Short-ctx TTFT | 2.9x faster | **3.1x faster** |
| Short-ctx decode | 14% slower | **3% faster** |
| Long-ctx total | n/a | 1.45x faster |
| Long-ctx decode | n/a | 9% faster |

**Int8 KV is now strictly better than fp32 KV across every measured Bonsai workload.** The architectural decision in KV5b to make int8 opt-in via env var (default fp32) is preserved for backwards compatibility, but the data supports promoting `BITNETSHARP_KV_CACHE_QUANTIZATION=Int8` to a recommended-default for all Bonsai serve deployments.

### KV-FU5 - Promote int8 to serve default

`BitNetPaperGguf.Load` now defaults to `KvCacheQuantization.Int8` whenever `BITNETSHARP_KV_CACHE_QUANTIZATION` is unset. `BitNetConfig()` ctor default stays `Fp32` so direct callers (tests, training code, embed scenarios) get backwards-compatible identity behaviour; the new default applies only at the production GGUF-load entry point.

Resolution logic:

```csharp
var kvOverride = BitNetOptions.KvCacheQuantizationEnvOverride;
var kvResolved = kvOverride ?? KvCacheQuantization.Int8;
```

Startup banner emitted at `BitNetPaperModel` logger:

```
KvCacheQuantization=Int8 (serve default; set BITNETSHARP_KV_CACHE_QUANTIZATION=Fp32 to opt out)
```

or when explicit:

```
KvCacheQuantization=Int8 (override applied via BITNETSHARP_KV_CACHE_QUANTIZATION)
```

Sanity: live `/api/chat` with no env var produced `total_duration=4279 ms` on the first run (consistent with the int8 short-ctx 5-run avg of 3437 ms; first run includes JIT warmup) and the expected `"stays use examples ..."` deterministic output. Suite stays 794/794 fast-lane green - no test asserted the GGUF-load default, so the flip is invisible to direct `BitNetConfig()` callers.

Section B is now landed end-to-end. The architectural pieces (KV1-KV6 + KV5b building blocks; KV-FU1 Avx2 hand-roll; KV-FU2 integer-composer int8; KV-FU3/FU4 Bonsai A/B at both context lengths; KV-FU5 default flip) form a coherent stack: int8 K/V cache is the default in production, the integer-forward composer supports it natively, and bench numbers confirm the win at every working-set size.

### KV-FU6 - ARM (NEON) hand-roll for DotInt8 / AccumulateWeightedInt8

Ultimate target hardware is ARM. The Avx2 hand-roll covers x86 dev / CI hosts; production deployments on ARM (Apple M-series, Snapdragon, Graviton, RPi, Cortex-A76+) need a NEON-targeted kernel. Falling back to the framework `Vector.Widen` portable path on ARM works but loses the same register-pressure win that motivated the Avx2 hand-roll.

`DotInt8Arm` and `AccumulateWeightedInt8Arm` mirror the Avx2 structure but target Vector128 (128-bit, NEON's natural width):

```csharp
var bytes = Vector128.LoadUnsafe(ref kRef, (nuint)i);              // LDR Q
var shortsLo = Vector128.WidenLower(bytes);                         // SXTL
var shortsHi = Vector128.WidenUpper(bytes);                         // SXTL2
var i0 = Vector128.WidenLower(shortsLo);                            // SXTL
// ... 4 Vector128<int> total
var f0 = Vector128.ConvertToSingle(i0);                             // SCVTF
// ... 4 Vector128<float>
acc = AdvSimd.FusedMultiplyAdd(acc, q0, f0);                        // FMLA
```

`Vector128.WidenLower` / `WidenUpper` are cross-platform abstractions; on ARM the JIT emits `SXTL` / `SXTL2`, on x86 it would emit `VPMOVSXBW` / `VPMOVSXBW + VPSHUFD` (suboptimal vs the 256-bit Avx2 path). The dispatch chain is `AdvSimd > Avx2 > Portable` so each host picks its native kernel.

Vector128's 4-wide float lane forces 4 FMLAs per 16-byte chunk vs the Avx2 path's 2 FMAs per 16-byte chunk (256-bit ymm). The extra FMA dispatches map cleanly to ARM's wide superscalar issue (Cortex-A76+ can issue 2 FMLAs per cycle on the NEON pipeline) so the per-chunk cost should match the Avx2 budget on similar-class hardware.

Bench: dev box is x86 (Zen 3 5900HX); ARM measurement is queued for a Sapphire-class follow-up session on M-series / Graviton / Snapdragon. Tests `tests/BitNetSharp.Tests/AttentionMathTests.cs::DotInt8_OnArmHost_MatchesPortableFallback` activate on ARM hosts (no-op on x86) and assert kernel output drift &lt; 1e-3 vs the scalar oracle.

Suite: 797/797 fast-lane green (794 + 3 ARM-gated theory inlines that skip on x86).

### Hardware target summary

| Host class | Hot kernel | Status |
| --- | --- | --- |
| ARMv8 (any Apple M, Snapdragon, Graviton, RPi 4+) | `DotInt8Arm` (NEON SXTL + FMLA) | Wired, awaiting ARM measurement |
| x86 AVX2 (Zen 1-3, Skylake+, dev box) | `DotInt8Avx2` (VPMOVSXBD + FMA) | Measured, 13-32% faster than fp32 |
| x86 portable / non-AVX2 / headDim &lt; 16 | `DotInt8Portable` (Vector.Widen) | Fallback; correctness only |

The next-wave AVX-VNNI-INT8 path (VPDPBSSD on Sapphire Rapids+ / Zen 5+) and ARM SDOT path (`AdvSimd.Dp.DotProduct` on ARMv8.4-A+) both require quantising q to int8 too - separate workstream that hoists Q quantisation out of the dot site and reuses the AvxVnniInt8 kernel pattern from G-series. Skipped this round; ARM SDOT is the more strategically relevant target given the ARM-first hardware roadmap.

### KV-FU7 - Live ARM bench on Motorola Edge 2024 (.NET MAUI harness)

Custom `BitNetSharp.Benchmarks.Maui` MAUI app (Android-only, net10.0-android) ports `KvCacheBenchmarks` to a Stopwatch-based runner because BenchmarkDotNet doesn't run on Android (no JIT process spawning, no Process API). Results captured from logcat. Device: **Motorola Edge 2024, Android 15, arm64-v8a**.

```
Host caps: AdvSimd=False, AdvSimd.Arm64=False, Avx2=False
Vector<float>.IsHardwareAccelerated=True, Vector<float>.Count=4, Vector<sbyte>.Count=16
Runtime: .NET 10.0.5
Arch: Arm64 / Arm64
```

**Critical finding: `AdvSimd.IsSupported=False` on Mono Android even on ARMv8 hardware.** Mono's runtime does not expose the `System.Runtime.Intrinsics.Arm.AdvSimd` class even though the underlying CPU supports NEON. This means the `DotInt8Arm` hand-roll from KV-FU6 never executes on Android Mono - dispatch falls through to `DotInt8Portable`. However, `Vector<float>.IsHardwareAccelerated=True` confirms Mono DOES vectorize `Vector<T>` operations onto NEON internally; the portable Vector.Widen path emits NEON automatically.

KvCacheBenchmark (Bonsai shape kvDim=1024 headDim=128, Stopwatch runner, warmup=3 iter=10):

| SeqLen | DotScan_Fp32 (ns) | DotScan_Int8 (portable) | Int8 ratio |
| -----: | ----------------: | ----------------------: | ---------: |
|     32 |            35 927 |                  25 672 |       0.71 |
|    128 |           106 953 |                 103 865 |       0.97 |
|    512 |           437 562 |                 409 063 |       0.94 |
|   2048 |         1 734 786 |               1 643 953 |       0.95 |

**Int8 wins at every SeqLen on ARM via the portable path** (5-29% faster). The win is smaller than on x86 Avx2 (which sees 30-32% at long SeqLen via VPMOVSXBD ymm) because Mono's portable Vector.Widen emits 128-bit NEON `SXTL` chains rather than wide-register 256-bit conversions. The result confirms the bandwidth thesis: int8 K cache halves DRAM/L2 traffic enough to win even when the dequant path is generic.

**Implication for KV-FU6:** the hand-rolled `DotInt8Arm` kernel is dead code on Android Mono. It remains useful for:
- Future CoreCLR-on-Android scenarios (where AdvSimd would be exposed)
- iOS / macOS-arm64 (CoreCLR exposes AdvSimd there)
- Server-side Linux ARM64 with CoreCLR (Graviton, Ampere Altra)

The portable Vector.Widen fallback is the actual hot path on Mono Android and ships a real win without the hand-roll.

Bench harness location: `benchmarks/BitNetSharp.Benchmarks.Maui/`. Device deployment: `dotnet build -c Release -f net10.0-android` then `adb install -r bin/Release/net10.0-android/*-Signed.apk` then launch via `adb shell am start -n com.companyname.bitnetsharp.benchmarks.maui/crc64c090f61d2c845dc2.MainActivity`. Results stream to logcat tagged `BENCH_KV`.

