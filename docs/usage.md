# Usage

## Build

```bash
dotnet build /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp.slnx
```

## Chat

The chat command can host the seeded paper-aligned transformer or another local comparison model and report that model's response for the supplied prompt.

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "how are you hosted"
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "how are you hosted" --model=traditional-local
```

Optional verbosity:

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "hello" --verbosity=quiet
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- chat "hello" --verbosity=verbose
```

## Host summary

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- host
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- host --model=traditional-local
```

This command confirms that the application is wired for Microsoft Agent Framework hosting and reports the selected model, language, and verbosity configuration.

## Transformer inspection

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- visualize
```

This command prints the current model summary. When the selected model is the paper-aligned BitNet transformer, it also prints the ternary weight histogram across the transformer's `BitLinear` projections.

## Benchmark

```bash
dotnet run --configuration Release --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- benchmark --model=bitnet-b1.58-sharp --compare-model=traditional-local --prompt="how are you hosted"
```

This command runs BenchmarkDotNet over the same hosted-model operations covered by the SpecFlow scenarios so you can compare local models under one agent wrapper.

## Benchmark report

```bash
dotnet run --configuration Release --project src/BitNetSharp.App/BitNetSharp.App.csproj -- benchmark-report --model=bitnet-b1.58-sharp --compare-model=traditional-local --output=/absolute/path/to/benchmark-report
```

This command runs the BenchmarkDotNet suite, evaluates both built-in models against the shared default training corpus/query script, and writes HTML, Markdown, and JSON comparison reports to the selected output directory.

## DataGen

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen --domain "medical-diagnosis" --count 25 --seeds examples/seed-examples.json --output data/synthetic-medical.jsonl
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen --domain "medical-diagnosis" --count 25 --seeds examples/seed-examples.json --output data/synthetic-medical.jsonl --lora medical-lora.bin
```

This command reads a JSON array of seed examples, expands them into synthetic instruction-response pairs, and writes JSONL output for downstream local fine-tuning or evaluation. See the [DataGen guide](datagen-guide.md) for accepted seed aliases and the output schema.

## Train the traditional comparison model

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- train --model=traditional-local
```

The paper-aligned transformer still reports that training is not implemented in this branch. The `traditional-local` model trains a small tensor-based local language model on the default corpus for 24 epochs so its training and query performance can be benchmarked on the same dataset.
