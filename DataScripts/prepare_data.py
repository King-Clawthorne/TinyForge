"""One-time data preparation: download, tokenize, and cache the corpus.

Downloads TinyStories via the HF cache (resumable, retried), trains/loads the
BPE tokenizer, and writes flat uint16 token files that the training runs can
memory-map with zero network access:

  DataOutput/tokens/val_<vocab>.bin    - first --val-docs documents
  DataOutput/tokens/train_<vocab>.bin  - following documents, up to --target-tokens
  DataOutput/tokens/meta_<vocab>.json  - eot_id, counts, provenance

Documents are packed end-to-end, each terminated by <|endoftext|>, matching
the streaming pipeline's layout. Run once before the matrix:

  python DataScripts/prepare_data.py

Default --target-tokens (140M) covers the default run budget
(1000 steps x 4 batch x 16 accum x 2048 block = 131M tokens) with headroom;
raise it if you raise the budget.
"""

import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from modules.utils import train_or_load_bpe

DATASET_PATH = "roneneldan/TinyStories"
ENCODE_BATCH = 1024


def main():
    parser = argparse.ArgumentParser(description="Tokenize the corpus to disk")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--target-tokens", type=int, default=140_000_000)
    parser.add_argument("--val-docs", type=int, default=2000)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "DataOutput" / "tokens"))
    parser.add_argument("--tokenizer-path", default=None)
    args = parser.parse_args()

    if args.vocab_size > 65536:
        raise ValueError("uint16 token files require vocab_size <= 65536")

    from datasets import load_dataset
    # Non-streaming: downloads the parquet shards to the local HF cache once
    # (resumable), then everything below is pure local I/O.
    print(f"Downloading/loading {DATASET_PATH} (full, non-streaming)...")
    ds = load_dataset(DATASET_PATH, split="train")
    print(f"{len(ds)} documents")

    tokenizer_path = args.tokenizer_path or str(
        REPO_ROOT / f"bpe_{args.vocab_size}.json")

    def bpe_corpus():
        for i in range(min(50_000, len(ds))):
            t = ds[i]["text"].strip()
            if t:
                yield t

    tokenizer = train_or_load_bpe(bpe_corpus(), vocab_size=args.vocab_size,
                                  save_path=tokenizer_path)
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def tokenize_range(start_doc, max_tokens=None, label=""):
        """Pack documents [start_doc, ...) into one uint16 token array."""
        chunks, total = [], 0
        batch = []

        def flush():
            nonlocal total
            if not batch:
                return
            for enc in tokenizer.encode_batch(batch):
                ids = enc.ids + [eot_id]
                chunks.append(np.asarray(ids, dtype=np.uint16))
                total += len(ids)
            batch.clear()

        last_doc = start_doc
        for i in range(start_doc, len(ds)):
            last_doc = i
            t = ds[i]["text"].strip()
            if not t:
                continue
            batch.append(t)
            if len(batch) >= ENCODE_BATCH:
                flush()
                if max_tokens is not None and total >= max_tokens:
                    break
                if total and total % 10_000_000 < ENCODE_BATCH * 300:
                    print(f"  {label}: {total / 1e6:.0f}M tokens...")
        flush()
        arr = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
        if max_tokens is not None:
            arr = arr[:max_tokens]
        return arr, last_doc

    print(f"Tokenizing {args.val_docs} validation docs...")
    val_chunks = []
    for enc in tokenizer.encode_batch(
            [ds[i]["text"].strip() for i in range(args.val_docs)
             if ds[i]["text"].strip()]):
        val_chunks.append(np.asarray(enc.ids + [eot_id], dtype=np.uint16))
    val_arr = np.concatenate(val_chunks)
    print(f"Val tokens: {len(val_arr) / 1e6:.1f}M")

    print(f"Tokenizing train docs (target {args.target_tokens / 1e6:.0f}M tokens)...")
    train_arr, last_doc = tokenize_range(args.val_docs, args.target_tokens,
                                         label="train")
    if len(train_arr) < args.target_tokens:
        print(f"NOTE: corpus exhausted at {len(train_arr) / 1e6:.0f}M tokens "
              f"(< target); training will cycle the data.")
    print(f"Train tokens: {len(train_arr) / 1e6:.1f}M (docs {args.val_docs}..{last_doc})")

    val_path = out_dir / f"val_{args.vocab_size}.bin"
    train_path = out_dir / f"train_{args.vocab_size}.bin"
    val_arr.tofile(val_path)
    train_arr.tofile(train_path)
    (out_dir / f"meta_{args.vocab_size}.json").write_text(json.dumps({
        "dataset": DATASET_PATH,
        "vocab_size": args.vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "eot_id": eot_id,
        "val_docs": args.val_docs,
        "val_tokens": len(val_arr),
        "train_tokens": len(train_arr),
        "train_file": train_path.name,
        "val_file": val_path.name,
        "dtype": "uint16",
    }, indent=2))
    print(f"Wrote {train_path} ({train_arr.nbytes / 1e6:.0f}MB), "
          f"{val_path} ({val_arr.nbytes / 1e6:.0f}MB), and meta json.")


if __name__ == "__main__":
    main()
