# DataGen guide

## Overview

`datagen` introduces a domain-agnostic synthetic dataset workflow for BitNet-b1.58-Sharp. The command accepts the target domain at runtime, grounds prompts with optional seed examples, runs the selected hosted model multiple times per sample, applies lightweight quality gates, and writes JSONL output that is ready for instruction-response training pipelines.

## Command

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen --domain=code-review --count=100 --output=/absolute/path/to/data/code-review.jsonl
```

## Supported options

- `--domain=...` required target domain
- `--count=...` number of accepted examples to write
- `--output=...` output JSONL file
- `--task-type=...` optional task family label such as `instruction-response`, `qa`, or `classification`
- `--seeds=...` optional JSON array of seed examples using either `instruction` or `prompt` plus `response`
- `--constraint=...` repeatable natural-language constraints
- `--constraints=a,b,c` comma-separated constraint shorthand
- `--output-schema=...` optional schema description injected into the prompt template
- `--template=...` optional absolute or relative path to a JSON prompt template
- `--lora=...` optional LoRA artifact path recorded in output metadata
- `--candidate-count=...` number of self-consistency passes per example, default `3`
- `--min-quality=...` acceptance threshold between `0.0` and `1.0`, default `0.45`
- `--max-tokens=...` optional model output limit
- `--model=...` optional hosted model selector, default `bitnet-b1.58-sharp`

## Seed format

```json
[
  {
    "instruction": "Review a patch for null handling regressions.",
    "response": "Check every nullable input path and add focused regression coverage."
  },
  {
    "prompt": "Summarize a code review finding.",
    "response": "Explain the risk, the affected path, and the proposed fix."
  }
]
```

## Output format

Each JSONL line includes the instruction-response pair plus generation metadata:

```json
{
  "instruction": "Create a instruction-response training example for the code-review domain (sample 1 of 2).",
  "response": "Domain: code-review\nTask type: instruction-response\n...",
  "prompt": "You are DataGen, a domain-agnostic synthetic data generator...",
  "domain": "code-review",
  "task_type": "instruction-response",
  "quality_score": 0.8167,
  "generation_timestamp": "2026-03-20T00:00:00+00:00",
  "grounding_context": [
    "Review a patch for null handling regressions."
  ],
  "lora": "/absolute/path/to/code-review-lora.bin"
}
```

## Templates

The repository ships with a default JSON template at `/templates/datagen/default.json`. Templates expose the placeholders `{domain}`, `{task_type}`, `{seed_examples}`, `{constraints}`, `{output_schema}`, `{count}`, and `{sample_number}` so one neutral template can be reused across domains without hard-coded vertical logic.

## Quality gates

The current implementation applies three acceptance checks on each candidate set:

1. Record validation for required instruction-response metadata
2. Self-consistency voting across repeated generations
3. Lexical diversity scoring against previously accepted responses

This keeps the runtime surface small while still producing traceable, grounded JSONL suitable for bootstrap dataset generation.
