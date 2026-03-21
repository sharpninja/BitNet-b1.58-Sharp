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
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- paper-audit
```

The `visualize` command prints the current model summary. For the built-in BitNet and `traditional-local` models, it also prints a signed weight histogram so both comparison models expose the same inspection surface from the CLI.

The `paper-audit` command turns the paper checklist into an executable report. It confirms the implemented architecture requirements that the repository currently satisfies and also verifies the repository-local runtime surface for paper-model fine-tuning, named perplexity fixture measurements, zero-shot fixture evaluation, and checkpoint round-tripping.

## Benchmark

```bash
dotnet run --framework net10.0 --configuration Release --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- benchmark --model=bitnet-b1.58-sharp --compare-model=traditional-local --prompt="how are you hosted"
```

This command runs BenchmarkDotNet over the same hosted-model operations covered by the SpecFlow scenarios so you can compare local models under one agent wrapper.

## Benchmark report

```bash
dotnet run --framework net10.0 --configuration Release --project src/BitNetSharp.App/BitNetSharp.App.csproj -- benchmark-report --model=bitnet-b1.58-sharp --compare-model=traditional-local --output=/absolute/path/to/benchmark-report
```

This command runs the BenchmarkDotNet suite, evaluates both built-in models against the shared default training corpus/query script, and writes HTML, Markdown, and JSON comparison reports to the selected output directory.
For the paper-aligned BitNet model, the generated report also includes a paper-alignment audit section with architecture checks and benchmark-pipeline coverage for training, perplexity fixtures, zero-shot fixtures, and checkpoint export/import validation.

## DataGen

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen --domain "medical-diagnosis" --count 25 --output data/synthetic-medical.jsonl
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen --domain "medical-diagnosis" --count 25 --seeds examples/seed-examples.json --output data/synthetic-medical.jsonl --constraint "Use American English" --lora medical-lora.bin
```

This command reads optional seed examples, merges the built-in pattern prompts with the repository template, and writes JSONL output for downstream local fine-tuning or evaluation. Optional flags include `--task-type`, `--constraint`, `--constraints`, `--output-schema`, `--template`, `--candidate-count`, `--min-quality`, `--max-tokens`, and `--lora`. The emitted JSONL includes both the core generator fields (`seedInstruction`, `variation`, `generatorModel`, `tags`) and the merged prompt metadata (`prompt`, `taskType`, `qualityScore`, `generationTimestamp`, `groundingContext`). See the [DataGen guide](datagen-guide.md) for accepted seed aliases and the merged output schema.

## Refresh full TinyLlama and WikiText-2 corpora

```bash
python scripts/process_full_corpora.py \
  --tinyllama-source /absolute/path/to/tinyllama.jsonl \
  --wikitext-source-dir /absolute/path/to/wikitext-2 \
  --commit \
  --commit-message "Vendor full TinyLlama and WikiText-2 corpora"
```

The script writes normalized TinyLlama train/validation/test JSONL files under `src/BitNetSharp.Core/Data/TinyLlama/`, refreshes the vendored WikiText-2 token files under `src/BitNetSharp.Core/Data/WikiText2/`, preserves the blank separator rows from the tokenized WikiText-2 corpus, and can optionally stage and commit the updated data files from your local clone. If you do not already have local WikiText-2 files, pass `--download-wikitext` to pull the tokenized `wiki.train.tokens`, `wiki.valid.tokens`, and `wiki.test.tokens` files from the default public source before writing them into the repository.

## Train the traditional comparison model

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- train --model=traditional-local
```

The paper-aligned transformer now exposes repository-local output-head fine-tuning on the default corpus so the benchmark pipeline can exercise its training path alongside inference. The `traditional-local` model still runs its 24-epoch tensor-based training loop for comparison on the same dataset.
