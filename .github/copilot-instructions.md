# Copilot instructions for BitNet-b1.58-Sharp

- Keep changes focused and surgical. Do not refactor unrelated code while addressing a task.
- This repository targets .NET 10 and C#. Projects currently enable nullable reference types and implicit usings.
- The supported runtime surface is the paper-aligned BitNet b1.58 transformer path. Do not reintroduce the retired toy or bigram workflow into the active application surface.
- Core model and transformer code lives in `src/BitNetSharp.Core`. Hosting and CLI entry points live in `src/BitNetSharp.App`. Tests live in `tests/BitNetSharp.Tests`.
- From the repository root, validate code changes with:
  - `dotnet build BitNet-b1.58-Sharp.slnx`
  - `dotnet test BitNet-b1.58-Sharp.slnx`
- Repository documentation is maintained in GitBook format under `docs/`. When adding or moving documentation pages, keep `docs/README.md` and `docs/SUMMARY.md` in sync.
- Preserve the existing American English tone in user-facing prompts, diagnostics, and documentation unless a task explicitly requires otherwise.
