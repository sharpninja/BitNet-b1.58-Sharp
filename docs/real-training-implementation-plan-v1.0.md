# Implementation Plan for Real Training in BitNet-b1.58-Sharp v1.0
**Replace Stub Training with Full Epochs, STE Backprop, Optimizer & Perplexity Validation**  
**Core Repository – Domain-Agnostic**

**Version:** 1.0  
**Date:** March 20, 2026  
**Status:** Ready-to-execute blueprint

---

## Table of Contents

1. [Executive Summary & Success Criteria](#1-executive-summary--success-criteria)
2. [Prerequisites & Current State](#2-prerequisites--current-state)
3. [Overall Training Architecture](#3-overall-training-architecture)
4. [Phase 1: WikiText-2 Data Loader & Tokenization (2–3 days)](#4-phase-1-wikitext-2-data-loader--tokenization-23-days)
5. [Phase 2: Real Train Method with Epochs, Batches & STE (5–7 days)](#5-phase-2-real-train-method-with-epochs-batches--ste-57-days)
6. [Phase 3: AdamW Optimizer & Gradient Updates (3–4 days)](#6-phase-3-adamw-optimizer--gradient-updates-34-days)
7. [Phase 4: Perplexity Evaluation on WikiText-2 (2–3 days)](#7-phase-4-perplexity-evaluation-on-wikitext-2-23-days)
8. [Phase 5: BenchmarkDotNet Integration & Reporting (3–4 days)](#8-phase-5-benchmarkdotnet-integration--reporting-34-days)
9. [Phase 6: Final Validation & CI Integration (2 days)](#9-phase-6-final-validation--ci-integration-2-days)
10. [Full UML Catalog](#10-full-uml-catalog)
11. [Risk Register & Mitigation](#11-risk-register--mitigation)
12. [Timeline & Effort Estimates](#12-timeline--effort-estimates)

---

## 1. Executive Summary & Success Criteria

Goal: Replace the current stub training with a **real, measurable training loop** that performs multiple epochs, computes loss, applies STE backprop, updates weights via AdamW, and reports perplexity on WikiText-2.

### Success Criteria

- Training runs multiple epochs and visibly reduces loss
- Perplexity on WikiText-2 validation is computed and reported (BitNet vs FP16 baseline)
- BenchmarkDotNet measures training time, tokens/sec, memory, and perplexity delta
- Report includes side-by-side TinyLlama-1.1B comparison
- Training no longer finishes in seconds — realistic duration on CPU/GPU

---

## 2. Prerequisites & Current State

- Existing `BitNetModel` and `BitLinear` with STE forward pass already implemented
- WikiText-2 raw validation set downloaded and tokenized (one-time step)
- BenchmarkDotNet already added to the test project (from prior benchmark patches)

---

## 3. Overall Training Architecture

```mermaid
flowchart TD
    A[WikiText-2 Validation Tokens] --> B[DataLoader (Batching)]
    B --> C[BitNetModel.Train(epochs)]
    C --> D[For each epoch]
    D --> E[Forward Pass (quantized)]
    E --> F[Cross-Entropy Loss]
    F --> G[STE Backward]
    G --> H[AdamW Optimizer Step]
    H --> I[Periodic Re-quantization]
    I --> J[Perplexity Calculation]
    J --> K[Benchmark Report]
```

---

## 4. Phase 1: WikiText-2 Data Loader & Tokenization (2–3 days)

1. Download `wikitext-2-raw-v1.zip` from the official source.
2. Add a tokenizer helper to convert raw text to token IDs by reusing the existing tokenizer.
3. Create a `WikiTextDataLoader` class that yields batches of shape `(batchSize, seqLen)`.
4. Cache the tokenized validation set in the test project for fast loading.

---

## 5. Phase 2: Real Train Method with Epochs, Batches & STE (5–7 days)

Update `BitNetModel` with a training API shaped like this:

```csharp
public TrainingReport Train(int epochs, IDataLoader dataLoader)
{
    var optimizer = new AdamWOptimizer(lr: 3e-4f, weightDecay: 0.1f);
    var report = new TrainingReport();

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalLoss = 0;
        int tokenCount = 0;

        foreach (var batch in dataLoader.GetBatches())
        {
            var logits = Forward(batch.Input);           // quantized forward
            var loss = CrossEntropyLoss(logits, batch.Target);
            totalLoss += loss.Value * batch.Size;
            tokenCount += batch.Size;

            loss.BackwardWithSTE();                      // straight-through estimator
            optimizer.Step(Parameters);
            optimizer.ZeroGrad();
        }

        report.AddEpoch(epoch, totalLoss / tokenCount);
        ReQuantizeAllLayers();                           // periodic re-quantization
    }

    return report;
}
```

---

## 6. Phase 3: AdamW Optimizer & Gradient Updates (3–4 days)

Implement a simple `AdamWOptimizer` class, or reuse an existing one if present, with:

- Momentum
- Variance
- Weight decay
- Support for ternary weight scaling (`γ`)
- In-place updates compatible with `BitLinear`

---

## 7. Phase 4: Perplexity Evaluation on WikiText-2 (2–3 days)

Add a validation method to `BitNetModel`:

```csharp
public double CalculatePerplexity(IDataLoader validationLoader)
{
    double totalNLL = 0;
    int tokenCount = 0;

    foreach (var batch in validationLoader.GetBatches())
    {
        var logits = Forward(batch.Input);
        var loss = CrossEntropyLoss(logits, batch.Target);
        totalNLL += loss.Value * batch.Size;
        tokenCount += batch.Size;
    }

    return Math.Exp(totalNLL / tokenCount);
}
```

---

## 8. Phase 5: BenchmarkDotNet Integration & Reporting (3–4 days)

Update `TinyLlamaBenchmark.cs`, or create it if it is missing, with:

```csharp
[Benchmark]
public double PerplexityBitNet() => _bitnetModel.CalculatePerplexity(wikiLoader);

[Benchmark]
public void TrainingEpoch() => _bitnetModel.Train(1, trainingLoader);
```

Enhance the report generator to include:

- Training time per epoch
- Perplexity before and after training
- BitNet vs FP16 baseline comparison

---

## 9. Phase 6: Final Validation & CI Integration (2 days)

- Add an integration test that runs 3 epochs and verifies loss decreases
- Update CI to run the full benchmark suite on a nightly schedule
- Generate HTML and JSON reports with tables and charts

---

## 10. Full UML Catalog

### Training Loop Flow

```mermaid
flowchart TD
    A[WikiText-2 Loader] --> B[Epoch Loop]
    B --> C[Batch Forward (BitLinear)]
    C --> D[Cross-Entropy Loss]
    D --> E[STE Backward]
    E --> F[AdamW Step]
    F --> G[Re-quantize]
    G --> H[Perplexity Calc]
```

---

## 11. Risk Register & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training still too fast | High | High | Enforce a minimum of 3 epochs and a real WikiText loader |
| STE gradient issues | Medium | High | Add a unit test that verifies gradient flow on a small batch |
| Memory explosion | Low | Medium | Use a small batch size (8–32) plus gradient clipping |

---

## 12. Timeline & Effort Estimates

| Phase | Estimate |
|------|----------|
| Phase 1: WikiText-2 Data Loader & Tokenization | 2–3 days |
| Phase 2: Real Train Method with Epochs, Batches & STE | 5–7 days |
| Phase 3: AdamW Optimizer & Gradient Updates | 3–4 days |
| Phase 4: Perplexity Evaluation on WikiText-2 | 2–3 days |
| Phase 5: BenchmarkDotNet Integration & Reporting | 3–4 days |
| Phase 6: Final Validation & CI Integration | 2 days |
| **Total** | **17–23 days** |

This plan is intentionally scoped to the core repository and remains domain-agnostic. It focuses on replacing stubbed training behavior with a measurable, benchmarked, paper-aligned training path.
