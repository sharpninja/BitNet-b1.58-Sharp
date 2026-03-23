#!/usr/bin/env python3
"""Build a deterministic small training corpus from repo-local JSONL sources.

This script is intentionally separate from process_full_corpora.py so the small
corpus workflow can stay repo-local, fast, and easy to reason about.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class NormalizedExample:
    split: str
    prompt: str
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic small corpus fixture set.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repo-local JSON or JSONL file containing raw records. Repeatable.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where small-corpus.train.jsonl and small-corpus.valid.jsonl should be written.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fallback train ratio for records without an explicit split.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for source in args.source:
        examples.extend(load_examples(Path(source).expanduser().resolve(), args.train_ratio))

    train_path = output_dir / "small-corpus.train.jsonl"
    valid_path = output_dir / "small-corpus.valid.jsonl"

    write_jsonl(train_path, (example for example in examples if example.split == "train"))
    write_jsonl(valid_path, (example for example in examples if example.split == "valid"))

    print(f"Wrote {train_path}")
    print(f"Wrote {valid_path}")
    print(f"Source records processed: {len(examples)}")
    return 0


def load_examples(path: Path, train_ratio: float) -> list[NormalizedExample]:
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    records = list(iter_records(path))
    examples = []
    for record in records:
        prompt, response = normalize_prompt_response(record)
        split = normalize_split(record, prompt, response, train_ratio)
        examples.append(NormalizedExample(split, prompt, response))

    return examples


def iter_records(path: Path) -> Iterator[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", newline=None) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Could not parse JSONL record {line_number} from {path}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL record {line_number} in {path} must be a JSON object.")
                yield value
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8", newline=None) as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError(f"JSON array entries in {path} must be JSON objects.")
                yield item
            return

        if isinstance(payload, dict):
            yield payload
            return

        raise ValueError(f"Unsupported JSON payload in {path}.")

    raise ValueError(f"Unsupported source format for {path}. Use .json or .jsonl.")


def normalize_split(record: dict[str, object], prompt: str, response: str, train_ratio: float) -> str:
    split = first_text(record, "split")
    if split:
        normalized = split.strip().lower()
        if normalized not in {"train", "valid"}:
            raise ValueError(f"Unsupported split '{split}'.")
        return normalized

    digest = hashlib.sha256(f"{prompt}\0{response}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if bucket < train_ratio else "valid"


def normalize_prompt_response(record: dict[str, object]) -> tuple[str, str]:
    prompt = first_text(record, "prompt", "question")
    response = first_text(record, "response", "answer")
    if prompt and response:
        return normalize_text(prompt), normalize_text(response)

    instruction = first_text(record, "instruction")
    output = first_text(record, "output")
    if instruction and output:
        supplemental_input = first_text(record, "input", "context")
        prompt_text = normalize_text(instruction)
        if supplemental_input:
            prompt_text = f"{prompt_text}\n\n{normalize_text(supplemental_input)}"
        return prompt_text, normalize_text(output)

    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        pair = extract_conversation_pair(messages)
        if pair is not None:
            return pair

    raise ValueError("Unsupported small corpus record format.")


def extract_conversation_pair(messages: Iterable[object]) -> tuple[str, str] | None:
    pending_prompt = None
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = first_text(message, "role", "from", "speaker")
        content = first_text(message, "content", "value", "text")
        if not role or not content:
            continue

        if is_user_role(role):
            pending_prompt = normalize_text(content)
            continue

        if is_assistant_role(role) and pending_prompt:
            return pending_prompt, normalize_text(content)

    return None


def is_user_role(role: str) -> bool:
    normalized = role.strip().lower()
    return normalized in {"user", "human", "prompt", "instruction"}


def is_assistant_role(role: str) -> bool:
    normalized = role.strip().lower()
    return normalized in {"assistant", "gpt", "model", "response", "output"}


def first_text(record: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def write_jsonl(path: Path, examples: Iterable[NormalizedExample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            json.dump(
                {"prompt": example.prompt, "response": example.response},
                handle,
                ensure_ascii=False,
                separators=(",", ":"))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
