## Summary

Describe the change and its user-visible impact.

## Validation

- [ ] `dotnet build BitNet-b1.58-Sharp.slnx`
- [ ] `dotnet test BitNet-b1.58-Sharp.slnx`

## Repository alignment checklist

- [ ] The change preserves the paper-aligned BitNet b1.58 runtime and does not reintroduce retired toy or bigram workflows into the active application surface.
- [ ] The change keeps the repository domain-agnostic at the core runtime, benchmark, and top-level documentation level.
- [ ] New or updated docs use Windows-first wording, PowerShell-oriented commands, and Windows-style paths when concrete path examples are needed.
- [ ] If I added, removed, or renamed pages under `docs\`, I updated both `docs\README.md` and `docs\SUMMARY.md`.
- [ ] Any new prompts, diagnostics, or examples keep the repository's American English tone.
