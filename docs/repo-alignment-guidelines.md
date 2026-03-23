# Repository alignment guidelines

## Purpose

This repository should stay focused on the paper-aligned BitNet b1.58 runtime and the local tooling needed to build, inspect, benchmark, and document it. These guidelines keep contributions consistent, Windows-first, and domain-agnostic.

## Core alignment rules

### Preserve the paper-aligned runtime surface

Changes should reinforce the active BitNet b1.58 transformer path in `src\BitNetSharp.Core` and the hosting or CLI entry points in `src\BitNetSharp.App`.

Do not reintroduce retired toy, bigram, or unrelated experimental workflows into the active application surface.

### Keep the repository domain-agnostic

The core runtime, built-in training data, benchmark positioning, and top-level documentation should remain general-purpose rather than anchored to a single business vertical, product, or proprietary workflow.

Examples can stay illustrative, but defaults should not hard-code product-specific assumptions into the repository's main experience.

### Prefer Windows-first guidance

When adding or updating documentation, favor PowerShell and `dotnet` CLI examples that work from a standard Windows clone.

If a document needs a concrete path example, use Windows-style paths such as `C:\src\BitNet-b1.58-Sharp` or repository-relative paths such as `src\BitNetSharp.Core`.

### Keep repository-local validation authoritative

Use the repository solution for the standard validation flow:

```powershell
dotnet build BitNet-b1.58-Sharp.slnx
dotnet test BitNet-b1.58-Sharp.slnx
```

If a change affects user-facing behavior, diagnostics, benchmarks, or fixtures, update the relevant tests or documentation alongside the code.

### Keep GitBook navigation in sync

When you add, remove, or rename pages under `docs\`, update both `docs\README.md` and `docs\SUMMARY.md` in the same change so the documentation map stays accurate.

## Review checklist

Before opening a pull request, confirm the following:

- The change keeps the repository aligned to BitNet b1.58 and the current .NET application surface.
- The change does not add domain-specific defaults to the core runtime or benchmark story.
- New or updated documentation uses American English and Windows-first instructions when concrete shell examples are needed.
- Documentation navigation files were updated if the contents of `docs\` changed.
- The repository still builds and tests cleanly with the standard solution commands.
