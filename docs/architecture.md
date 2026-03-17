# Architecture

BitNet b1.58 Sharp uses a deliberately small architecture so the repository can provide an end-to-end .NET 10 reference implementation without hiding the core ideas behind a large dependency graph.

## Core design

`/src/BitNetSharp.Core` contains the model, tokenizer, trainer, visualization support, and the new additive transformer scaffold:

- `BitNetModel` stores a ternary weight matrix using `sbyte` values in `-1`, `0`, and `+1`
- `BitNetTrainer` fits the matrix from prompt/response pairs and records loss history
- `BitNetVisualizer` renders ASCII charts and CSV output for quick inspection
- `VerbosityLevel` exposes exactly three interaction levels: `Quiet`, `Normal`, and `Verbose`
- `BitLinear` implements absmean-scaled ternary projections with signed int8 activation quantization
- `RmsNorm`, `RotaryPositionEmbedding`, `MultiHeadAttention`, `SwiGLUFeedForward`, `BitNetLayer`, and `BitNetTransformer` provide a paper-aligned decoder-only scaffold without changing the existing toy model path

## Hosting design

`/src/BitNetSharp.App` prioritizes hosting through Microsoft Agent Framework packages:

- `Microsoft.Agents.AI`
- `Microsoft.Agents.AI.Hosting`

The app registers a local `IChatClient` implementation so the BitNet model can be hosted through Agent Framework conventions while remaining runnable as a standalone console application.

## Language and interaction model

The built-in training corpus and command output default to American English. That keeps prompts, diagnostics, and help text aligned with the issue requirement for a primary U.S. English interface.
