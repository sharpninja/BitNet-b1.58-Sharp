# Architecture

BitNet b1.58 Sharp targets the paper-aligned BitNet b1.58 decoder-only transformer architecture and no longer presents the earlier toy-model path as part of the supported runtime surface.

## Core design

`/src/BitNetSharp.Core` contains the paper-model runtime and transformer building blocks:

- `BitNetPaperModel` wraps tokenizer state, the seeded transformer, and next-token inspection output
- `DataGenGenerator` expands JSON seed examples into synthetic JSONL records while recording BitNet model provenance
- `VerbosityLevel` exposes exactly three interaction levels: `Quiet`, `Normal`, and `Verbose`
- `BitLinear` implements absmean-scaled ternary projections with signed int8 activation quantization
- `RmsNorm`, `RotaryPositionEmbedding`, `MultiHeadAttention`, `SwiGLUFeedForward`, `BitNetLayer`, and `BitNetTransformer` implement the decoder-only paper architecture

## Hosting design

`/src/BitNetSharp.App` prioritizes hosting through Microsoft Agent Framework packages:

- `Microsoft.Agents.AI`
- `Microsoft.Agents.AI.Hosting`

The app registers a local `IChatClient` implementation so the paper-aligned BitNet model can be hosted through Agent Framework conventions while remaining runnable as a standalone console application.

The hosting layer now resolves multiple local model types behind the same agent wrapper:

- the seeded paper-aligned BitNet model
- a traditional local tensor-based comparison model trained on the default corpus with `System.Numerics.Tensors`
- local command models described by JSON configuration files

This lets BenchmarkDotNet measure host construction, querying, streaming, and local training through one shared path.

The same app surface also exposes a `datagen` command that keeps synthetic data generation local to the repository checkout.

## Language and interaction model

The built-in vocabulary and command output default to American English. That keeps prompts, diagnostics, and help text aligned with the requirement for a primary U.S. English interface.
