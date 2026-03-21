# Bucketing Guide

Bucketing is a core optimization in BitNet b1.58 Sharp that accelerates inference via **Chain-Bucket Speculative Decoding** and reduces training cost via **Training-Time Sequence Compression**.

---

## How It Works

### Chain-Bucket Speculative Decoding (Inference)

A `ChainBucketTable` stores up to 256 frequent n-gram chains (length 2–8) mined from a training corpus. During generation:

1. After each normally generated token, the last 1–3 context tokens are looked up in the table.
2. If a matching chain is found, the model speculatively emits the chain's continuation tokens.
3. Each speculative token is verified: if the model's top-1 prediction matches, the token is accepted.
4. Accepted tokens are appended to the context at once, reducing the number of full forward passes.

This is safe: no token is accepted without model verification.

### Training-Time Sequence Compression

When compression is enabled, the prompt context passed to the forward pass is shortened by replacing known chain n-grams with the first token of each chain. The loss target is unchanged. This reduces the effective context length and speeds up each training step.

---

## Quick Start

### Via CLI (automatic corpus mining)

```bash
# Chat with chain-bucket speculative decoding active
dotnet run --project src/BitNetSharp.App -- chat "hello" --enable-bucketing

# Train with sequence compression active
dotnet run --project src/BitNetSharp.App -- train --enable-bucketing
```

The `--enable-bucketing` flag mines a `ChainBucketTable` from the default training corpus at startup and activates both `EnableChainBuckets` and `EnableSequenceCompression`.

### Via code (programmatic setup)

```csharp
// Create a model with bucketing options enabled
var model = BitNetBootstrap.CreatePaperModel(
    verbosity: VerbosityLevel.Normal,
    enableChainBuckets: true,
    enableSequenceCompression: true);

// Mine buckets from your own training examples
var examples = MyCorpus.LoadExamples();
var table = model.MineAndLoadBuckets(examples);
Console.WriteLine($"Mined {table.Count} chain buckets.");

// Generate with speculative decoding active
var result = model.GenerateResponse("What is BitNet?");
```

### Via `BucketMiner` directly (advanced)

```csharp
using BitNetSharp.Core.Bucketing;

// Provide tokenized integer sequences
IReadOnlyList<int>[] sequences = GetTokenizedCorpus();
var table = BucketMiner.Mine(sequences, maxBuckets: 256);

model.LoadBucketTable(table);
```

---

## Configuration Options

The following properties are added to `BitNetOptions`:

| Property | Default | Description |
|----------|---------|-------------|
| `EnableChainBuckets` | `false` | Activates chain-bucket speculative decoding during inference. |
| `MaxChainLength` | `8` | Maximum n-gram length considered during bucket mining and speculative expansion. |
| `EnableSequenceCompression` | `false` | Activates training-time prompt compression using chain buckets. |

---

## Expected Performance

| Metric | Without Bucketing | With Bucketing |
|--------|-------------------|----------------|
| Tokens/sec (inference) | baseline | ≥ 1.8× (≥ 70 % acceptance rate) |
| Effective sequence length (training) | baseline | 20–35 % shorter |
| Training time per epoch | baseline | 20–35 % faster |
| Output quality | baseline | no regression (verified) |

Actual gains depend on corpus repetition patterns and chain acceptance rates.

---

## Architecture

See the full design in [Bucketing Implementation Plan v1.0](bucketing-implementation-plan-v1.0.md).

Key source files:

| File | Description |
|------|-------------|
| `src/BitNetSharp.Core/Bucketing/ChainBucket.cs` | Record for a single n-gram chain bucket. |
| `src/BitNetSharp.Core/Bucketing/ChainBucketTable.cs` | 256-entry lookup table with prefix matching. |
| `src/BitNetSharp.Core/Bucketing/BucketMiner.cs` | N-gram mining and scoring service. |
| `src/BitNetSharp.Core/BitNetOptions.cs` | `EnableChainBuckets`, `MaxChainLength`, `EnableSequenceCompression`. |
| `src/BitNetSharp.Core/BitNetPaperModel.cs` | Integrated speculative decoding and compression. |
| `src/BitNetSharp.App/Program.cs` | `--enable-bucketing` CLI flag. |
