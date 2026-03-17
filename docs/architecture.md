# Architecture

BitNet b1.58 Sharp targets the paper-aligned BitNet b1.58 decoder-only transformer architecture and no longer presents the earlier toy-model path as part of the supported runtime surface.

## Core design

`/src/BitNetSharp.Core` contains the paper-model runtime and transformer building blocks:

- `BitNetPaperModel` wraps tokenizer state, the seeded transformer, and next-token inspection output
- `VerbosityLevel` exposes exactly three interaction levels: `Quiet`, `Normal`, and `Verbose`
- `BitLinear` implements absmean-scaled ternary projections with signed int8 activation quantization
- `RmsNorm`, `RotaryPositionEmbedding`, `MultiHeadAttention`, `SwiGLUFeedForward`, `BitNetLayer`, and `BitNetTransformer` implement the decoder-only paper architecture

## Hosting design

`/src/BitNetSharp.App` prioritizes hosting through Microsoft Agent Framework packages:

- `Microsoft.Agents.AI`
- `Microsoft.Agents.AI.Hosting`

The app registers a local `IChatClient` implementation so the paper-aligned BitNet model can be hosted through Agent Framework conventions while remaining runnable as a standalone console application.

## Language and interaction model

The built-in vocabulary and command output default to American English. That keeps prompts, diagnostics, and help text aligned with the requirement for a primary U.S. English interface.
