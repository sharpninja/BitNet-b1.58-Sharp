# BitNet-b1.58-Sharp - Session Handoff

**Date:** 2026-04-26
**Session:** `ClaudeCode-20260426T033736Z-plugin` (MCP server `http://PAYTON-LEGION2:7147`)
**HEAD:** `cf6c801` (pushed to Azure DevOps `origin`)
**Branch:** `feat/integer-forward-hot-path`
**Open PR:** [PR 20](https://dev.azure.com/McpServer/McpServer/_git/BitNet-b1.58-Sharp/pullrequest/20) (thread 48 carries the close-out summary)
**Tests:** 772/772 green (xunit, net10.0)

## What was just shipped: H-series matmul wrapper close-out

Bonsai 782M (36 layers, dim=4096, 32 Q / 8 KV heads) per-decode-token latency on the same Zen 3 / AVX2 host (PAYTON-LEGION2):

| Metric             | G3 baseline | H5 (H2+H3+H4) |   Delta |
| ------------------ | ----------: | ------------: | ------: |
| total_ms (avg)     |      72 135 |        11 251 |    6.4x |
| TTFT_ms (prefill)  |      25 363 |         9 996 |    2.5x |
| decode_dur_ms      |      46 771 |         1 255 |   37.3x |
| per_decode_token_ms|       5 197 |           157 |   33.1x |

The 2 s/token gate is met with **12.7x margin**. 5-run warm `/api/chat` measurement.

### Pieces in the H stack

- **H1 dropped.** Matmul-wrapper-cache analysis showed ~3.6 us/token win against a 3 200 ms gap. Not worth the risk surface.
- **H2 - SSSE3 fast unpack** (`src/BitNetSharp.Core/Quantization/TritPacking.cs`, new `SimdUnpackLayerSsse3`). Pre-H2 `SimdUnpackLayer` was a scalar shift/store loop costing ~4096 ops per output column at inDim=4096 vs ~770 ops for the AVX2 dot. New kernel decodes 16 packed bytes (= 64 trits) per chunk via mask+shift slot extraction, VPSHUFB sign-extend lookup against a 16-byte LUT `[0, 1, -2, -1, ...]`, and `VPUNPCKL/H` byte+i16 interleave to restore positional order. SSSE3 chosen over AVX2 to dodge the 128-bit lane-crossing wart of VPSHUFB on YMM. LUT keeps the legacy `0b10 -> -2` contract so it stays bit-exact with the scalar oracle even though `SimdPackLayer` never emits 0b10. 13 equivalence tests cover all 256 byte values + lengths 1..11009.
- **H3 - Parallel.For column stripes** (`src/BitNetSharp.Core/Layers/BitLinear.cs`, both `ForwardQuantized` and `ForwardInt32`). Outer column loop wrapped in `Parallel.For` with `localInit` (ArrayPool-rent decoded buffer) / body (decode + dot per row) / `localFinally` (return to pool). Gated by `MinParallelOutDim = 1024` to skip overhead on small shapes. New `TritDotDispatch.UseParallelColumnStripes` flag (with internal `ForceSerial` backdoor so tests can pin determinism). 11 equivalence tests confirm parallel matches serial across production shapes.
- **H4 - `Vector{256,512}.LoadUnsafe` ref-base loads** (`src/BitNetSharp.Core/Quantization/TritPacking.Avx512.cs`, all three of `TernaryDotAvx2Sign`, `TernaryDotAvxVnniInt8`, `TernaryDotAvxVnniInt8V512`). `Vector256.Create<sbyte>(span.Slice(...))` emits a length check on every iteration; `LoadUnsafe(ref T, nuint)` trusts the caller. Outer `(length >= laneCount)` and `chunks * laneCount <= length` gates already guarantee in-bounds, so the per-iteration check was dead weight.

### Files touched in commit `cf6c801`

```
src/BitNetSharp.Core/Layers/BitLinear.cs
src/BitNetSharp.Core/Quantization/TritDotDispatch.cs
src/BitNetSharp.Core/Quantization/TritPacking.Avx512.cs
src/BitNetSharp.Core/Quantization/TritPacking.cs
tests/BitNetSharp.Tests/BitLinearParallelTests.cs       (new, 11 tests)
tests/BitNetSharp.Tests/TritDotPackedKernelTests.cs     (new, 13 tests)
docs/research/inference-latency.md                       (H-series section appended)
```

## What is still open

The G/H series knocked the matmul wrapper down ~33x. The remaining gap to "human-fast" sits at higher layers - the same items that were scoped in the original `fuzzy-orbiting-parrot.md` plan. With matmul effectively saturated on AVX2, the next dollar is in the wrapper around it:

1. **KV cache** for attention K/V across decode steps. Currently every decode step re-projects the full growing context. Plan section: Phase 1 of `~\.claude\plans\fuzzy-orbiting-parrot.md` (data model, RoPE position-offset overload, cache-aware `Forward` overloads on `MultiHeadAttention` / `GroupedQueryAttention` / `BitNetLayer` / `BitNetTransformer` / `BitNetPaperModel`). 5 red tests pre-defined.
2. **Activation-quantisation cache** so the shared layer input is quantised once and reused across Q/K/V (and Gate/Up). New `QuantizedActivationBlock`, `BitLinear.ForwardQuantized(QuantizedActivationBlock)`. 3 red tests pre-defined. Noted that H-series already added a `ForwardQuantized` path; phase-2 should consume it from the attention/FFN sites.
3. **SIMD attention inner loop** - `for d in headDim` dot products in `GroupedQueryAttention` lines 110-113, 139-142 are still scalar. New `AttentionMath` static class. 3 red tests pre-defined.
4. **Fused flash-style attention** for the decode case (query length = 1) so no N×N attention-weight tensor is materialised at decode time. Online-softmax kernel. 2 red tests pre-defined.
5. **Streaming `/api/chat`** - emit one NDJSON chunk per token so clients see progress before the full generation finishes. 3 red tests pre-defined.
6. **BenchmarkDotNet harness** - new `benchmarks/BitNetSharp.Benchmarks/` project, 6 suites covering every component above. Pinned at `[SimpleJob(RuntimeMoniker.Net100, warmupCount: 3, iterationCount: 10)]`.

The plan is still binding. Each phase follows Byrd TDD (red tests first), publishes deltas to `docs/research/inference-latency.md`, and does not land until the suite stays green.

## Decode kernel shape after H-series

```
BitLinear.ForwardQuantized(QuantizedActivationBlock input)
  -> Parallel.For(0, outDim, MinParallelOutDim=1024 gate)
       per-worker: ArrayPool<sbyte>.Rent(inDim)
       per-column:
         SimdUnpackLayerSsse3(packedRow, decoded)        // H2
         per-row: TernaryDotSimdUnpacked(decoded, act)   // H4 ref-base load inside
```

Dispatcher knobs in `TritDotDispatch`:

| Flag                        | Default   | Test override |
| --------------------------- | --------- | ------------- |
| `UseParallelColumnStripes`  | `true`    | `ForceSerial` field flips it |
| (existing G-series gates)   | unchanged | unchanged     |

## How to resume

```powershell
# On PAYTON-LEGION2 (current dev box):
cd F:\GitHub\BitNet-b1.58-Sharp
git fetch origin
git checkout feat/integer-forward-hot-path
dotnet build BitNet-b1.58-Sharp.slnx -c Release
dotnet test tests/BitNetSharp.Tests -c Release -f net10.0 --filter "Category!=SlowLane"

# Re-run the H5 Bonsai measurement (server must be up):
dotnet run --project src/BitNetSharp.App -c Release -- serve
# In a second shell:
curl -sS -X POST http://127.0.0.1:11434/api/chat `
  -H "Content-Type: application/json" `
  --data @cache/h5_chat_payload.json
```

Plan to pick up next: `~\.claude\plans\fuzzy-orbiting-parrot.md` Phase 1 (KV cache). All red tests are listed; start with `TransformerKvCacheTests.cs` and bring the cache-aware overloads up.

## Key architectural decisions (this round)

1. **SSSE3 over AVX2 for trit unpack.** YMM-VPSHUFB has the lane-crossing wart and was not worth a 256-bit version once the 128-bit kernel cleared the bottleneck.
2. **Per-worker ArrayPool buffer in Parallel.For.** Per-iteration rent/return would dominate at outDim=4096; `localInit/localFinally` amortizes it across the worker's iteration block.
3. **`MinParallelOutDim = 1024` gate.** Below that the spawn overhead beats the win. Tunable; held constant for the close-out.
4. **`ForceSerial` test backdoor on dispatcher.** Lets the parallel-vs-serial equivalence tests pin determinism without polluting production behaviour.
5. **LUT preserves legacy `0b10 -> -2` contract.** The packer never emits 0b10 but the historical scalar oracle's sign-extension would, so the SSSE3 LUT keeps it for bit-exact equivalence.
6. **H1 dropped on cost-vs-risk.** Matmul-wrapper-cache surfaces a coherent invalidation surface for ~3.6 us/token. Not worth it against the 3 200 ms gap.

## MCP session log

Session `ClaudeCode-20260426T033736Z-plugin` on `http://PAYTON-LEGION2:7147` carries 5 turns: H2 (SSSE3 unpack), H3 (Parallel.For stripes), H4 (LoadUnsafe ref-base loads), H5 (Bonsai measurement + commit + push + PR 20 thread), and this handoff turn. Posted via `POST /mcpserver/sessionlog` with full `UnifiedSessionLogDto`.

## Reference

- Plan: `~\.claude\plans\fuzzy-orbiting-parrot.md` (KV cache + activation-quant cache + SIMD attention + flash + streaming + BDN harness)
- Latency log: `docs/research/inference-latency.md` (G-series + H-series tables)
- PR 20 thread 48: H-series summary on Azure DevOps
- Distributed-training context (now historical): see commit `adc801a` and earlier session `Claude-20260415T120000Z-bitnet-distributed-training`
