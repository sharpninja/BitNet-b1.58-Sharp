#!/usr/bin/env python3
"""
process_full_corpora.py
Downloads and prepares TinyLlama tokenizer + WikiText-2 validation set for benchmarks.
Saves pre-tokenized data directly into the repository under data/.
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
WIKITEXT_VALID_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
TINYLLAMA_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOKEN_WRITE_CHUNK_SIZE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and tokenize TinyLlama + WikiText-2 for benchmarks.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT, help="Repository root")
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Compatibility flag; required inputs are downloaded automatically when missing.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--no-tokenize", action="store_true", help="Skip tokenization step")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_data_dirs(repo_root: Path) -> tuple[Path, Path, Path]:
    data_dir = repo_root / "data"
    return data_dir, data_dir / "WikiText2", data_dir / "TinyLlama"


def import_snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it with "
            "`python -m pip install huggingface_hub`."
        ) from exc

    return snapshot_download


def import_tokenizer_class() -> type[Any]:
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'tokenizers'. Install it with "
            "`python -m pip install tokenizers`."
        ) from exc

    return Tokenizer


def find_wikitext_validation_file(target_dir: Path) -> Path | None:
    extract_dir = target_dir / "raw"
    candidates = (
        extract_dir / "wikitext-2-raw-v1" / "wiki.valid.raw",
        extract_dir / "wikitext-2-raw-v1" / "wiki.valid.tokens",
        extract_dir / "wiki.valid.raw",
        extract_dir / "wiki.valid.tokens",
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def download_tinyllama_tokenizer(target_dir: Path, force: bool = False) -> None:
    tokenizer_path = target_dir / "tokenizer.json"
    if not force and tokenizer_path.exists():
        print("TinyLlama tokenizer already exists - skipping download.")
        return

    ensure_dir(target_dir)

    print(f"Downloading TinyLlama tokenizer from {TINYLLAMA_MODEL_ID}...")
    snapshot_download = import_snapshot_download()
    snapshot_download(
        repo_id=TINYLLAMA_MODEL_ID,
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
        local_dir=str(target_dir),
        local_dir_use_symlinks=False)
    print("TinyLlama tokenizer downloaded.")


def download_wikitext(target_dir: Path, force: bool = False) -> Path:
    ensure_dir(target_dir)

    existing_validation_file = find_wikitext_validation_file(target_dir)
    if not force and existing_validation_file is not None:
        print("WikiText-2 already exists - skipping download.")
        return existing_validation_file

    zip_path = target_dir / "wikitext-2-raw-v1.zip"
    extract_dir = target_dir / "raw"

    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)

    print("Downloading WikiText-2 raw dataset...")
    urlretrieve(WIKITEXT_VALID_URL, str(zip_path))

    print("Extracting WikiText-2...")
    shutil.unpack_archive(str(zip_path), str(extract_dir))

    validation_file = find_wikitext_validation_file(target_dir)
    if validation_file is None:
        raise FileNotFoundError(
            f"Unable to locate the extracted WikiText-2 validation file under '{extract_dir}'.")

    print("WikiText-2 downloaded and extracted.")
    return validation_file


def write_token_ids(output_file: Path, tokens: list[int]) -> None:
    ensure_dir(output_file.parent)

    with output_file.open("wb") as handle:
        for offset in range(0, len(tokens), TOKEN_WRITE_CHUNK_SIZE):
            chunk = tokens[offset:offset + TOKEN_WRITE_CHUNK_SIZE]
            if any(token < 0 or token > 0xFFFFFFFF for token in chunk):
                raise ValueError("Tokenizer output must fit in unsigned 32-bit integers.")

            if chunk:
                handle.write(struct.pack(f"<{len(chunk)}I", *chunk))


def tokenize_wikitext(valid_file: Path, tokenizer: Any, output_file: Path) -> None:
    print("Tokenizing WikiText-2 validation set with TinyLlama tokenizer...")
    text = valid_file.read_text(encoding="utf-8")

    encoding = tokenizer.encode(text)
    tokens = encoding.ids
    write_token_ids(output_file, tokens)

    print(f"Tokenized {len(tokens)} tokens -> saved to {output_file}")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        print(f"Error: Repository root does not exist: {repo_root}", file=sys.stderr)
        return 1

    data_dir, wikitext_dir, tinyllama_dir = get_data_dirs(repo_root)
    ensure_dir(data_dir)
    ensure_dir(wikitext_dir)
    ensure_dir(tinyllama_dir)

    try:
        # Preserve the current workflow by refreshing the required inputs when they are missing.
        download_tinyllama_tokenizer(tinyllama_dir, args.force)
        valid_raw = download_wikitext(wikitext_dir, args.force)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.no_tokenize:
        print("Tokenization skipped as requested.")
        return 0

    tokenizer_path = tinyllama_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print("Error: TinyLlama tokenizer not found after download.", file=sys.stderr)
        return 1

    try:
        tokenizer_class = import_tokenizer_class()
        tokenizer = tokenizer_class.from_file(str(tokenizer_path))
        tokenized_path = wikitext_dir / "wikitext-2-valid-tokens.bin"
        tokenize_wikitext(valid_raw, tokenizer, tokenized_path)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nAll files ready for benchmarks!")
    print(f"Tokenizer: {tinyllama_dir}")
    print(f"Tokenized WikiText-2 validation: {tokenized_path}")
    print("\nYou can now use these paths in your C# benchmarks and training pipeline.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
