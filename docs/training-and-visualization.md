# Training and Visualization

## Paper-model status

The repository runtime now only uses the paper-aligned BitNet transformer path. The earlier toy prompt/response trainer is no longer part of the documented workflow.

## Inspect the seeded transformer

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- visualize
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- paper-audit
```

This command prints the current paper-model configuration and an aggregated ternary weight histogram across every `BitLinear` projection in the seeded transformer.
The `paper-audit` command adds a structured checklist on top of that inspection output so the repository can report which paper-aligned architecture requirements are currently implemented and which end-to-end reproduction items are still pending.

## Inspect next-token predictions

To inspect the seeded transformer with a prompt:

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "how are you hosted"
```

The chat surface reports the paper model's top next-token predictions for the supplied prompt instead of using the retired toy prompt/response path.

## Training roadmap

The training command is intentionally explicit about the current state:

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- train
```

At the moment it reports that the paper-aligned training loop is not yet implemented in this branch. This keeps the runtime honest: the repository only uses the full transformer path from the paper, and it does not fall back to the retired toy trainer.
