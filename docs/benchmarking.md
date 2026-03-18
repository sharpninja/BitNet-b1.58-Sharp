# Benchmarking and model comparison

## Overview

`BitNetSharp.App` can now host more than one local model shape through the same Microsoft Agent Framework wrapper:

- `bitnet-b1.58-sharp` for the paper-aligned seeded transformer
- `traditional-local` for a local tensor-based comparison model trained on the default training corpus
- an absolute path to a local command model JSON file for other locally available models

The benchmark command uses BenchmarkDotNet to measure the same hosted-model operations that the SpecFlow scenarios exercise:

- training a selected trainable model on the default dataset
- generating a response for a prompt
- streaming a response for a prompt
- building the agent host

The manual GitHub Actions benchmark report workflow runs the same benchmark suite for both built-in models, then publishes a static comparison site through GitHub Pages. That report combines:

- efficacy, measured as non-empty responses across the shared default query script
- accuracy, measured as exact-match and expected-token recall against the default corpus responses
- performance, measured from the exported BenchmarkDotNet results

## Run the built-in comparison benchmark

```bash
dotnet run --configuration Release --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- benchmark --model=bitnet-b1.58-sharp --compare-model=traditional-local --prompt="how are you hosted"
```

This runs the BenchmarkDotNet suite over both local models so their hosted response and host-construction costs can be compared directly.

## Generate the comparison report site

```bash
dotnet run --configuration Release --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- benchmark-report --model=bitnet-b1.58-sharp --compare-model=traditional-local --output=/home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/artifacts/benchmark-report
```

This command writes a static report site with:

- `index.html` for GitHub Pages publishing
- `comparison-report.md` and `comparison-report.json` summaries
- raw BenchmarkDotNet HTML, CSV, and GitHub-flavored Markdown exports under `BenchmarkDotNet.Artifacts/results/`

The repository also includes a manual trigger workflow at `.github/workflows/benchmark-report.yml` that builds, tests, generates the same report, uploads it as an artifact, and deploys it with GitHub Pages.

## Train the traditional local model

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- train --model=traditional-local
```

The traditional local model trains over `BitNetTrainingCorpus.CreateDefaultExamples()` for 24 epochs using `System.Numerics.Tensors` softmax and dot-product primitives so it can be benchmarked and queried against the same dataset every time.

## Compare another local model

Pass the absolute path to a JSON file that describes how to execute a locally available model runner:

```bash
dotnet run --configuration Release --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- benchmark --model=/absolute/path/to/local-model.json --compare-model=traditional-local
```

Example configuration:

```json
{
  "modelId": "my-local-model",
  "displayName": "My local model",
  "executablePath": "/absolute/path/to/model-runner",
  "arguments": [
    "--model",
    "/absolute/path/to/model.bin"
  ],
  "promptTransport": "StandardInput",
  "primaryLanguage": "en-US"
}
```

### Prompt transport options

- `StandardInput`: the prompt is written to the process standard input
- `FinalArgument`: the prompt is appended as the final command-line argument

This keeps model comparison local-only and avoids endpoint or API-key based integrations.
