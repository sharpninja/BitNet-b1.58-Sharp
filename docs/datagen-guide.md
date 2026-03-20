# DataGen guide

## Overview

`datagen` is the repository's offline synthetic dataset bootstrapper for the paper-aligned BitNet b1.58 runtime. The merged implementation combines deterministic variation patterns with the repository's prompt-template system so each generated JSONL record carries both a reusable DataGen prompt and structured training metadata. Seeds remain optional at runtime, but when supplied they are used for grounding, prompt rendering, and output attribution.

## Generate a dataset

```bash
dotnet run --project src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen \
  --domain "medical-diagnosis" \
  --count 50000 \
  --seeds examples/seed-examples.json \
  --output data/synthetic-medical.jsonl \
  --constraint "Use American English" \
  --lora medical-lora.bin
```

## Supported options

- `--domain=...` required target domain
- `--count=...` number of accepted examples to write
- `--output=...` output JSONL file
- `--seeds=...` optional JSON array of seed examples using either `instruction`, `prompt`, or `input` plus `response`, `output`, or `answer`
- `--task-type=...` optional task family label such as `instruction-response`, `qa`, or `classification`
- `--constraint=...` repeatable natural-language constraints
- `--constraints=a,b,c` comma-separated constraint shorthand
- `--output-schema=...` optional schema description injected into the prompt template
- `--template=...` optional absolute or relative path to a JSON prompt template
- `--lora=...` optional LoRA artifact path recorded in output metadata
- `--candidate-count=...` number of self-consistency passes per example, default `3`
- `--min-quality=...` acceptance threshold between `0.0` and `1.0`, default `0.45`
- `--max-tokens=...` optional model output limit

## Seed format

Seed files are standard JSON arrays. Each object must include one instruction-like field and one response-like field. DataGen accepts the following aliases:

- instruction: `instruction`, `prompt`, or `input`
- response: `response`, `output`, or `answer`

```json
[
  {
    "prompt": "Summarize the patient's main complaint and likely differential diagnosis.",
    "response": "Restate the complaint, list the most likely causes, and flag any immediate safety concerns."
  },
  {
    "instruction": "Explain what evidence should be gathered before choosing a treatment plan.",
    "answer": "Collect history, exam findings, recent medications, and any contraindications before recommending next steps."
  }
]
```

When `--seeds` is omitted, the command synthesizes a neutral seed from the requested domain and task type so small bootstrap runs still work.

## Output schema

Each JSONL line includes the instruction-response pair plus generation metadata:

```json
{
  "instruction": "Create a medical-diagnosis task that starts from this seed: Summarize the patient's main complaint and likely differential diagnosis. [sample 1]",
  "response": "Use the seed response as the baseline: Restate the complaint, list the most likely causes, and flag any immediate safety concerns. Then adapt it for medical-diagnosis work with extra attention to complaint, diagnosis, safety.",
  "prompt": "You are DataGen, a domain-agnostic synthetic data generator...",
  "domain": "medical-diagnosis",
  "taskType": "instruction-response",
  "qualityScore": 0.8167,
  "generationTimestamp": "2026-03-20T00:00:00+00:00",
  "groundingContext": [
    "Summarize the patient's main complaint and likely differential diagnosis."
  ],
  "lora": "/absolute/path/to/medical-lora.bin",
  "seedInstruction": "Summarize the patient's main complaint and likely differential diagnosis.",
  "seedResponse": "Restate the complaint, list the most likely causes, and flag any immediate safety concerns.",
  "variation": "pattern-1",
  "generatorModel": "bitnet-b1.58-sharp",
  "tags": [
    "synthetic",
    "offline",
    "pattern-1",
    "medical",
    "diagnosis"
  ]
}
```

## Templates

The repository ships with a default JSON template at `/templates/datagen/default.json`. Templates expose the placeholders `{domain}`, `{task_type}`, `{seed_examples}`, `{constraints}`, `{output_schema}`, `{count}`, `{sample_number}`, `{variation}`, `{seed_instruction}`, and `{seed_response}`. The built-in variation patterns from the core generator are injected into that template so the two prompt systems stay merged rather than diverging, while the emitted JSON uses camelCase metadata fields such as `taskType`, `qualityScore`, `generationTimestamp`, and `generatorModel`.

## Quality controls

The current implementation applies lightweight quality scoring to every accepted example:

1. prompt/response schema validation
2. self-consistency scoring across repeated BitNet cue generations
3. lexical diversity scoring against previously accepted responses

Use a smaller preview run first, inspect the JSONL output, and then scale up counts once the prompt template and constraints match your target domain.

## Integration notes

DataGen is intentionally local-first:

- generation runs entirely offline
- output stays in your working directory
- the same built-in BitNet model ID and optional LoRA path are recorded with every example for traceability
