# DataGen guide

DataGen is the repository's offline synthetic dataset bootstrapper for the paper-aligned BitNet b1.58 runtime. It takes a small set of seed examples, applies deterministic variation patterns, and uses the built-in BitNet transformer to condition each batch with lightweight next-token cues.

## Generate a dataset

```bash
dotnet run --project /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/src/BitNetSharp.App/BitNetSharp.App.csproj -- datagen \
  --domain "medical-diagnosis" \
  --count 50000 \
  --seeds /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/examples/seed-examples.json \
  --output /home/runner/work/BitNet-b1.58-Sharp/BitNet-b1.58-Sharp/data/synthetic-medical.jsonl \
  --lora medical-lora.bin
```

The command writes one JSON object per line so the output can flow directly into local fine-tuning or evaluation jobs.

## Seed format

Seed files are standard JSON arrays. Each object must include one instruction-like field and one response-like field. DataGen accepts the following aliases:

- instruction: `instruction`, `prompt`, or `input`
- response: `response`, `output`, or `answer`

Example:

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

## Output schema

Each JSONL line includes:

- `domain`
- `instruction`
- `response`
- `seedInstruction`
- `seedResponse`
- `variation`
- `generatorModel`
- `loraAdapter`
- `tags`

The optional `--lora` argument is recorded in output metadata so runs can stay attributable even when adapter-conditioned execution is handled outside the CLI.

## Quality controls

- Start from diverse seeds that already match the tone and structure you need.
- Generate a smaller preview set first, then inspect the JSONL output before scaling up.
- Filter or deduplicate generated samples before fine-tuning if your target pipeline requires stricter curation.

## Integration notes

DataGen is intentionally local-first:

- generation runs entirely offline
- output stays in your working directory
- the same built-in BitNet model ID is recorded with every example for traceability
