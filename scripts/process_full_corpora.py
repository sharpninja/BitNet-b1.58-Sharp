#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import lzma
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIKITEXT_BASE_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"
DEFAULT_COMMIT_MESSAGE = "Vendor processed TinyLlama and WikiText-2 corpora"
WIKITEXT_FILES = ("wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens")
USER_ROLES = {"human", "user", "prompt", "instruction"}
ASSISTANT_ROLES = {"assistant", "gpt", "model", "response", "output"}
TRANSCRIPT_PATTERN = re.compile(
    r"(?:###\s*)?(?:Human|User)\s*:\s*(?P<prompt>.+?)\s*(?:###\s*)?(?:Assistant|GPT|Model)\s*:\s*(?P<response>.+?)(?=(?:###\s*)?(?:Human|User)\s*:|\Z)",
    re.DOTALL)


@dataclass(frozen=True)
class TrainingPair:
    prompt: str
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare repository-local TinyLlama and WikiText-2 corpora for commit.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root to update. Defaults to the current script's repository.")
    parser.add_argument(
        "--tinyllama-source",
        action="append",
        default=[],
        help="Path or URL for a monolithic TinyLlama JSON/JSONL source. Repeat as needed.")
    parser.add_argument(
        "--tinyllama-train-source",
        action="append",
        default=[],
        help="Path or URL for a TinyLlama training split source. Repeat as needed.")
    parser.add_argument(
        "--tinyllama-valid-source",
        action="append",
        default=[],
        help="Path or URL for a TinyLlama validation split source. Repeat as needed.")
    parser.add_argument(
        "--tinyllama-test-source",
        action="append",
        default=[],
        help="Path or URL for a TinyLlama test split source. Repeat as needed.")
    parser.add_argument(
        "--tinyllama-train-ratio",
        type=float,
        default=0.98,
        help="Train ratio for deterministic TinyLlama splitting when --tinyllama-source is used.")
    parser.add_argument(
        "--tinyllama-valid-ratio",
        type=float,
        default=0.01,
        help="Validation ratio for deterministic TinyLlama splitting when --tinyllama-source is used.")
    parser.add_argument(
        "--wikitext-source-dir",
        type=Path,
        help="Directory containing wiki.train.tokens, wiki.valid.tokens, and wiki.test.tokens.")
    parser.add_argument(
        "--download-wikitext",
        action="store_true",
        help="Download WikiText-2 from the default public tokenized corpus URLs.")
    parser.add_argument(
        "--wikitext-base-url",
        default=DEFAULT_WIKITEXT_BASE_URL,
        help="Base URL used when --download-wikitext is specified.")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Run git add/commit for the generated corpus files after processing.")
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message to use with --commit.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise ValueError(f"Repository root does not exist: {repo_root}")

    has_monolithic_tinyllama = bool(args.tinyllama_source)
    split_source_flags = [
        args.tinyllama_train_source,
        args.tinyllama_valid_source,
        args.tinyllama_test_source
    ]
    has_split_tinyllama = any(split_source_flags)

    if has_monolithic_tinyllama and has_split_tinyllama:
        raise ValueError("Use either --tinyllama-source or explicit TinyLlama split sources, not both.")

    if not has_monolithic_tinyllama and not has_split_tinyllama:
        raise ValueError("Provide TinyLlama input with --tinyllama-source or explicit split sources.")

    if has_split_tinyllama and not all(split_source_flags):
        raise ValueError("When using explicit TinyLlama splits, provide --tinyllama-train-source, --tinyllama-valid-source, and --tinyllama-test-source.")

    if has_monolithic_tinyllama:
        if not 0 < args.tinyllama_train_ratio < 1:
            raise ValueError("--tinyllama-train-ratio must be between 0 and 1.")
        if not 0 <= args.tinyllama_valid_ratio < 1:
            raise ValueError("--tinyllama-valid-ratio must be between 0 and 1.")
        if args.tinyllama_train_ratio + args.tinyllama_valid_ratio >= 1:
            raise ValueError("TinyLlama train and validation ratios must leave room for a test split.")

    if args.wikitext_source_dir is None and not args.download_wikitext:
        raise ValueError("Provide --wikitext-source-dir or use --download-wikitext.")

    if args.wikitext_source_dir is not None and args.download_wikitext:
        raise ValueError("Use either --wikitext-source-dir or --download-wikitext, not both.")

    if args.wikitext_source_dir is not None and not args.wikitext_source_dir.exists():
        raise ValueError(f"WikiText-2 source directory does not exist: {args.wikitext_source_dir}")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        repo_root = args.repo_root.resolve()

        written_paths = []
        tinyllama_paths = write_tinyllama_splits(repo_root, args)
        written_paths.extend(tinyllama_paths)

        wikitext_paths = write_wikitext_splits(repo_root, args)
        written_paths.extend(wikitext_paths)

        if args.commit:
            commit_outputs(repo_root, written_paths, args.commit_message)
    except Exception as exc:  # pragma: no cover - local operator diagnostics
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def write_tinyllama_splits(repo_root: Path, args: argparse.Namespace) -> list[Path]:
    tinyllama_output_dir = repo_root / "src" / "BitNetSharp.Core" / "Data" / "TinyLlama"
    tinyllama_output_dir.mkdir(parents=True, exist_ok=True)

    if args.tinyllama_source:
        monolithic_examples = collect_training_pairs(args.tinyllama_source)
        splits = split_training_pairs(
            monolithic_examples,
            train_ratio=args.tinyllama_train_ratio,
            valid_ratio=args.tinyllama_valid_ratio)
    else:
        splits = {
            "train": collect_training_pairs(args.tinyllama_train_source),
            "valid": collect_training_pairs(args.tinyllama_valid_source),
            "test": collect_training_pairs(args.tinyllama_test_source)
        }

    written_paths = []
    for split_name in ("train", "valid", "test"):
        output_path = tinyllama_output_dir / f"tinyllama.{split_name}.jsonl"
        write_training_pairs(output_path, splits[split_name])
        print(f"Wrote {len(splits[split_name])} TinyLlama {split_name} examples to {output_path}")
        written_paths.append(output_path)

    return written_paths


def write_wikitext_splits(repo_root: Path, args: argparse.Namespace) -> list[Path]:
    wikitext_output_dir = repo_root / "src" / "BitNetSharp.Core" / "Data" / "WikiText2"
    wikitext_output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for file_name in WIKITEXT_FILES:
        output_path = wikitext_output_dir / file_name
        if args.wikitext_source_dir is not None:
            source_path = args.wikitext_source_dir / file_name
            if not source_path.exists():
                raise FileNotFoundError(f"Missing WikiText-2 source file: {source_path}")
            with source_path.open("r", encoding="utf-8", newline=None) as source:
                write_text_lines(output_path, source)
        else:
            source_url = urljoin(args.wikitext_base_url, file_name)
            with open_text_stream(source_url) as source:
                write_text_lines(output_path, source)

        print(f"Wrote WikiText-2 split to {output_path}")
        written_paths.append(output_path)

    return written_paths


def collect_training_pairs(sources: Iterable[str]) -> list[TrainingPair]:
    pairs = []
    seen = set()

    for source in sources:
        for record in iter_records(source):
            for pair in normalize_record(record):
                key = f"{pair.prompt}\0{pair.response}"
                if key in seen:
                    continue

                seen.add(key)
                pairs.append(pair)

    if not pairs:
        raise ValueError("No TinyLlama training pairs were produced from the supplied sources.")

    return pairs


def split_training_pairs(
    pairs: Iterable[TrainingPair],
    train_ratio: float,
    valid_ratio: float) -> dict[str, list[TrainingPair]]:
    splits = {"train": [], "valid": [], "test": []}

    for pair in pairs:
        split_name = choose_split(pair, train_ratio, valid_ratio)
        splits[split_name].append(pair)

    return splits


def choose_split(pair: TrainingPair, train_ratio: float, valid_ratio: float) -> str:
    digest = hashlib.sha256(f"{pair.prompt}\0{pair.response}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + valid_ratio:
        return "valid"
    return "test"


def iter_records(source: str) -> Iterator[object]:
    source_path = source_path_name(source)
    lower_name = source_path.lower()

    with open_text_stream(source) as handle:
        if lower_name.endswith((".jsonl", ".jsonl.gz", ".jsonl.xz", ".jsonl.lzma")):
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Could not parse JSONL record {line_number} from {source}: {exc}") from exc
            return

        if lower_name.endswith((".json", ".json.gz", ".json.xz", ".json.lzma")):
            payload = json.load(handle)
            if isinstance(payload, list):
                yield from payload
                return

            if isinstance(payload, dict):
                if isinstance(payload.get("data"), list):
                    yield from payload["data"]
                    return
                if isinstance(payload.get("records"), list):
                    yield from payload["records"]
                    return

            raise ValueError(f"Unsupported JSON structure in {source}.")

    raise ValueError(f"Unsupported TinyLlama source format for {source}. Use .json, .jsonl, .json.gz, or .jsonl.gz.")


def normalize_record(record: object) -> list[TrainingPair]:
    if not isinstance(record, dict):
        raise ValueError("TinyLlama records must be JSON objects.")

    prompt = first_text(record, "prompt", "question")
    response = first_text(record, "response", "answer")
    if prompt and response:
        return [TrainingPair(clean_text(prompt), clean_text(response))]

    instruction = first_text(record, "instruction")
    output = first_text(record, "output")
    if instruction and output:
        supplemental_input = first_text(record, "input", "context")
        prompt_text = clean_text(instruction)
        if supplemental_input:
            prompt_text = f"{prompt_text}\n\n{clean_text(supplemental_input)}"
        return [TrainingPair(prompt_text, clean_text(output))]

    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        pairs = list(extract_conversation_pairs(messages))
        if pairs:
            return pairs

    transcript = first_text(record, "text")
    if transcript:
        pairs = list(extract_transcript_pairs(transcript))
        if pairs:
            return pairs

    raise ValueError("Unsupported TinyLlama record format. Expected prompt/response, instruction/output, or conversation messages.")


def extract_conversation_pairs(messages: Iterable[object]) -> Iterator[TrainingPair]:
    pending_prompt: str | None = None

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = first_text(message, "role", "from", "speaker")
        content = first_text(message, "content", "value", "text")
        if not role or not content:
            continue

        normalized_role = role.strip().lower()
        if normalized_role in USER_ROLES:
            pending_prompt = clean_text(content)
            continue

        if normalized_role in ASSISTANT_ROLES and pending_prompt:
            yield TrainingPair(pending_prompt, clean_text(content))
            pending_prompt = None


def extract_transcript_pairs(transcript: str) -> Iterator[TrainingPair]:
    for match in TRANSCRIPT_PATTERN.finditer(transcript):
        prompt = clean_text(match.group("prompt"))
        response = clean_text(match.group("response"))
        if prompt and response:
            yield TrainingPair(prompt, response)


def first_text(record: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def clean_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


@contextlib.contextmanager
def open_text_stream(source: str) -> Iterator[TextIO]:
    parsed = urlparse(source)
    source_name = source_path_name(source)
    raw_handle = None
    text_handle = None

    try:
        if parsed.scheme in {"http", "https"}:
            raw_handle = urlopen(source)
        else:
            raw_handle = Path(source).expanduser().resolve().open("rb")

        if source_name.endswith(".gz"):
            raw_handle = gzip.GzipFile(fileobj=raw_handle)
        elif source_name.endswith((".xz", ".lzma")):
            raw_handle = lzma.LZMAFile(raw_handle)

        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8")
        yield text_handle
    finally:
        if text_handle is not None:
            text_handle.close()
        elif raw_handle is not None:
            raw_handle.close()


def source_path_name(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return parsed.path
    return str(Path(source).expanduser())


def write_training_pairs(path: Path, pairs: Iterable[TrainingPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            json.dump({"prompt": pair.prompt, "response": pair.response}, handle, ensure_ascii=False)
            handle.write("\n")


def write_text_lines(path: Path, source: TextIO) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for line in source:
            destination.write(line.rstrip("\n").rstrip("\r"))
            destination.write("\n")


def commit_outputs(repo_root: Path, written_paths: Iterable[Path], commit_message: str) -> None:
    relative_paths = [str(path.resolve().relative_to(repo_root)) for path in written_paths]
    subprocess.run(["git", "-C", str(repo_root), "add", *relative_paths], check=True)

    diff_result = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--", *relative_paths],
        check=False)

    if diff_result.returncode == 0:
        print("No corpus changes were staged, so no commit was created.")
        return

    subprocess.run(["git", "-C", str(repo_root), "commit", "-m", commit_message], check=True)
    print(f"Created git commit with message: {commit_message}")


if __name__ == "__main__":
    raise SystemExit(main())
