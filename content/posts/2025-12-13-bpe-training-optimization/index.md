---
title: "BPE Training Optimisation"
date: 2025-12-13
tags: [BPE, Tokenization, Optimisation, Benchmarking, Python]
series: ["Tokenization Deep Dive"]
---

#### Goal

Optimise the BPE training loop (counts, merge pass, and I/O) while preserving correctness. Baseline below is the current version used for benchmarking.

#### Reference

- Repository: [SDcodehub/LM-training](https://github.com/SDcodehub/LM-training)

#### Current version (baseline)

<details>
<summary>View baseline code</summary>

```python
"""
Train a BPE model on a text file
"""
import os
import logging
import json
import time
import multiprocessing
from collections import Counter
from binascii import b2a_hex
from heapq import nlargest
import regex as re
from tqdm import tqdm
from LM_training.utils.logging_config import get_logger
from pretokenization_example import find_chunk_boundaries

log = get_logger()
# GPT 2 tokenizer pattern
# This regex splits the text into chunks of letters numbers or punctuations
# Its designed to keep the spaces attached to the words that follow them
# Compile pattern once for performance
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenise_text(input_path, special_tokens=None):

    if special_tokens is None:
        special_tokens = []

    log.info(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as read_file:
        text = read_file.read()
   

    # Build a regex pattern to split the text by any of the special tokens.
    # re.escape is used in case a special token contains characters with special
    # meaning in regex, like '|'.
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_chunks = re.split(f"({special_pattern})", text)
    else:
        text_chunks = [text]

    # pre tokenize the text chunks seperately
    word_counts = {}

    log.info("Pre-tokenizing text...")
    for chunk in tqdm(text_chunks, desc="Chunking"):
        # Ignore the special tokens
        # handles in the vocab seperately
        if chunk in special_tokens:
            continue

        # find all pre-tokens in the chunk
        for word in SPLIT_PATTERN.findall(chunk):
            word_counts[word] = word_counts.get(word, 0) + 1

    # BPE generally works on the byte sequences to converting the strings into byte sequences
    splits = {word.encode("utf-8"): count for word, count in word_counts.items()}
    return splits

def initialise_vocab(special_tokens):
    # vocab is a mapping from the integer ID to the byte sequence
    vocab = {i: bytes([i]) for i in range(256)}

    #add special tokens
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    return vocab

def get_stats(splits):
    """
    Give n splits pre tokenized, return a dictionary of pairs of byte sequences and their counts
    """
    stats = {}
    for word_part, count in splits.items():
        # A word is represented as the byte sequences
        for i in range(len(word_part)-1):
            # form the pair of adjacent tokens
            pair = (word_part[i], word_part[i+1])
            # increment the count for the pair
            stats[pair] = stats.get(pair, 0) + count
    return stats

def merge_splits(splits, pair, new_token):
    """Replaces all the occuraces of pair in the splits with new_token"""
    p0, p1 = pair
    new_splits = {}
    for words_parts, count in splits.items():
        # Optimization: If the pair isn't in this word, skip the heavy logic
        # Note: This is a heuristic check; p0 might exist without p1 following it.
        # But it saves time for words that contain neither byte.
        if p0 not in words_parts:
             new_splits[words_parts] = count
             continue

        new_words_parts = []
        i = 0
        n = len(words_parts)

        while i < n:
            # Optimized Check: Direct index access is faster than slicing [i:i+2]
            if i < n - 1 and words_parts[i] == p0 and words_parts[i+1] == p1:
                new_words_parts.append(new_token)
                i += 2
            else:
                new_words_parts.append(words_parts[i])
                i += 1
        new_splits[tuple(new_words_parts)] = count
    return new_splits


def save_tokenizer(vocab, merges, prefix):
    """Saves the vocabulary and merges to files."""
    vocab_file = f"{prefix}_vocab.json"
    merges_file = f"{prefix}_merges.txt"

    # 1. Save the vocabulary
    # We need to convert bytes to a JSON-serializable format (list of ints)
    serializable_vocab = {
        token_id: list(byte_sequence) for token_id, byte_sequence in vocab.items()
    }
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    log.info(f"Vocabulary saved to {vocab_file}")

    # 2. Save the merges
    # We save as hex to avoid any issues with special characters or spaces
    with open(merges_file, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            p1_hex = b2a_hex(p1).decode('ascii')
            p2_hex = b2a_hex(p2).decode('ascii')
            f.write(f"{p1_hex} {p2_hex}\n")
    log.info(f"Merges saved to {merges_file}")


def train_bpe(input_path, vocab_size, special_tokens, save_prefix=None):
    """Main function for training BPE model"""

    start_time = time.time()
    vocab_map = initialise_vocab(special_tokens)
    log.info("vocab size: %d", len(vocab_map))

    raw_splits = pretokenise_text(input_path, special_tokens)
    log.info("unique pretokenized byte-sequences: %d", len(raw_splits))

    # Convert raw bytes keys to tuple of bytes for mutability simulation
    splits = {tuple(bytes([b]) for b in word): count for word, count in raw_splits.items()}


    merges = []
    num_merges = vocab_size - len(vocab_map)

    log.info(f"Starting BPE training. Target merges: {num_merges}")

    # WRAPPER: tqdm for progress bar
    progress_bar = tqdm(range(num_merges), desc="Training BPE")
    
    for i in progress_bar:
        # Get the stats of the splits
        pair_stats = get_stats(splits)

        if not pair_stats:
            # If there are no more adjacent-byte pairs to merge, break
            log.info("No more adjacent-byte pairs to merge")
            break

        log.info("unique adjacent-byte pairs: %d", len(pair_stats))

        # Debug-only: top-K pairs
        if log.isEnabledFor(logging.DEBUG):
            top_pairs = nlargest(20, pair_stats.items(), key=lambda kv: kv[1])
            for (a, b), count in top_pairs:
                log.debug("pair (%d,%d) [%02x %02x] -> %d", a[0], b[0], a[0], b[0], count)

        # Get the top pair by the count        
        best_pair = max(pair_stats, key=lambda pair: (pair_stats[pair], pair))

        # Create new token and perform the merge
        p1, p2 = best_pair
        new_token_bytes = p1 + p2
        new_token_id = len(vocab_map)

        # Upsatte vocab, merges, splits
        vocab_map[new_token_id] = new_token_bytes
        merges.append(best_pair)
        splits = merge_splits(splits, best_pair, new_token_bytes)

        # Update tqdm description with current stats rarely (to save rendering time)
        if i % 10 == 0:
            progress_bar.set_postfix({"Best Pair": f"{p1}+{p2}", "Count": pair_stats[best_pair]})
        
        # LOGGING STRATEGY: Only log to console every X steps
        if i % 100 == 0:
             if log.isEnabledFor(logging.DEBUG):
                 log.debug(f"Merge {i+1}: {best_pair} -> {new_token_bytes}")

    total_time = time.time() - start_time
    log.info(f"Finished training in {total_time:.2f}s. Final vocab size: {len(vocab_map)}")

    if save_prefix:
        save_tokenizer(vocab_map, merges, save_prefix)

    return vocab_map, merges

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Train a Byte-Pair Encoding (BPE) tokenizer on a text file."
    )

    # 1. Input File (Positional Argument - Required)
    parser.add_argument(
        "input_path", 
        type=str, 
        help="Path to the training text file (e.g., data/corpus.txt)"
    )

    # 2. Output Directory (Optional)
    parser.add_argument(
        "--output_dir", "-o", 
        type=str, 
        default="bpe_tokenizer",
        help="Directory to save the vocab and merges files (default: bpe_tokenizer)"
    )

    # 3. Vocab Size (Optional)
    parser.add_argument(
        "--vocab_size", "-v", 
        type=int, 
        default=5000,
        help="Target vocabulary size (default: 5000)"
    )

    # 4. Filename Prefix (Optional)
    parser.add_argument(
        "--prefix", "-p", 
        type=str, 
        default="tokenizer",
        help="Prefix for the saved files (default: tokenizer)"
    )

    # 5. Special Tokens (Optional - List)
    parser.add_argument(
        "--special_tokens", "-s", 
        nargs="*", 
        default=["<|endoftext|>"],
        help="List of special tokens to include (default: <|endoftext|>)"
    )

    args = parser.parse_args()

    # Validation: Check if input file exists
    if not os.path.exists(args.input_path):
        log.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    full_save_prefix = os.path.join(args.output_dir, args.prefix)

    log.info(f"Training BPE with vocab_size={args.vocab_size} on {args.input_path}")
    log.info(f"Special tokens: {args.special_tokens}")

    # Run Training
    train_bpe(
        args.input_path, 
        args.vocab_size, 
        args.special_tokens, 
        save_prefix=full_save_prefix
    )
```

</details>

#### Optimized version

<details>
<summary>View optimized code</summary>

```python
"""
Train a BPE model on a text file (Optimized with Parallel Processing & Inverted Index)
"""
import os
import logging
import json
import time
import regex as re
import multiprocessing
import heapq
from typing import BinaryIO, List
from binascii import b2a_hex
from collections import defaultdict, Counter
from tqdm import tqdm
from LM_training.utils.logging_config import get_logger

log = get_logger()

# -----------------------------------------------------------------------------
# Helper: Parallel Chunk Boundary Finder
# -----------------------------------------------------------------------------
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    Ensures boundaries align with the special token (e.g., space) to avoid cutting words.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # Set boundary strictly AFTER the token to ensure the token 
                # stays with the previous chunk (or is the split point)
                chunk_boundaries[bi] = initial_position + found_at
                break
            
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

# -----------------------------------------------------------------------------
# Helper: Worker for Parallel Processing
# -----------------------------------------------------------------------------
def _process_chunk_worker(args):
    """Worker function to process a single file chunk."""
    filename, start, end, special_pattern_str = args
    local_counts = Counter()
    
    # GPT-2 Split Pattern (compiled locally for the worker)
    # Note: We rely on the byte-level processing, so we assume UTF-8 text.
    local_split_pattern = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    
    with open(filename, "rb") as f:
        f.seek(start)
        text_bytes = f.read(end - start)
        # Decode efficiently, ignoring errors at boundaries
        text = text_bytes.decode("utf-8", errors="ignore")

    if special_pattern_str:
        chunks = re.split(special_pattern_str, text)
    else:
        chunks = [text]

    for chunk in chunks:
        if not chunk: continue
        for word in local_split_pattern.findall(chunk):
            local_counts[word] += 1
            
    return local_counts

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------
def initialise_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    return vocab

def save_tokenizer(vocab, merges, prefix):
    vocab_file = f"{prefix}_vocab.json"
    merges_file = f"{prefix}_merges.txt"

    serializable_vocab = {
        token_id: list(byte_sequence) for token_id, byte_sequence in vocab.items()
    }
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    log.info(f"Vocabulary saved to {vocab_file}")

    with open(merges_file, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            p1_hex = b2a_hex(p1).decode('ascii')
            p2_hex = b2a_hex(p2).decode('ascii')
            f.write(f"{p1_hex} {p2_hex}\n")
    log.info(f"Merges saved to {merges_file}")

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------
def train_bpe(input_path, vocab_size, special_tokens, save_prefix=None):
    start_time = time.time()
    
    # 1. Initialize Vocab
    vocab_map = initialise_vocab(special_tokens)
    log.info("Initial vocab size (bytes + special): %d", len(vocab_map))

    # 2. Parallel Pre-tokenization
    log.info(f"Chunking {input_path}...")
    
    # Prepare regex for special tokens
    special_pattern_str = None
    if special_tokens:
        escaped = [re.escape(t) for t in special_tokens]
        special_pattern_str = f"({'|'.join(escaped)})"

    # Find boundaries using 'space' as a safe split point for text
    num_processes = max(1, os.cpu_count() - 1)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b" ") 

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_pattern_str))

    log.info(f"Pre-tokenizing with {num_processes} cores...")
    global_counts = Counter()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for local_count in tqdm(pool.imap_unordered(_process_chunk_worker, tasks), total=len(tasks), desc="Pre-tokenizing"):
            global_counts.update(local_count)
            
    # Convert string words to byte tuples for BPE
    splits = {tuple(bytes([b]) for b in word.encode("utf-8")): count 
              for word, count in global_counts.items()}
    
    log.info(f"Unique words found: {len(splits)}")

    # 3. Build Inverted Index & Initial Stats
    # token_to_words: token -> set of words containing that token
    token_to_words = defaultdict(set)
    pair_stats = defaultdict(int)

    log.info("Building inverted index and stats...")
    for word, count in splits.items():
        # Indexing
        for token in word:
            token_to_words[token].add(word)
        # Stats
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_stats[pair] += count

    # 4. Training Loop
    merges = []
    num_merges = vocab_size - len(vocab_map)
    
    # Heap for O(1) access to best pair. (-count, pair) for Min-Heap simulating Max-Heap
    stats_heap = []
    for pair, count in pair_stats.items():
        heapq.heappush(stats_heap, (-count, pair))

    log.info(f"Starting BPE training. Target merges: {num_merges}")
    
    progress_bar = tqdm(range(num_merges), desc="Training BPE")
    
    for i in progress_bar:
        # A. Get Best Pair (Lazy removal from heap)
        best_pair = None
        current_count = 0
        
        while stats_heap:
            neg_count, pair = heapq.heappop(stats_heap)
            real_count = pair_stats.get(pair, 0)
            # If heap count matches real count, it's valid. Else it's stale.
            if -neg_count == real_count:
                best_pair = pair
                current_count = real_count
                break
        
        if not best_pair:
            log.info("No more pairs to merge.")
            break

        # B. Create New Token
        p0, p1 = best_pair
        new_token_bytes = p0 + p1
        new_token_id = len(vocab_map)
        vocab_map[new_token_id] = new_token_bytes
        merges.append(best_pair)
        
        # Clean up stats for the merged pair itself
        del pair_stats[best_pair]

        # C. Merge in Splits (Using Inverted Index)
        # We only look at words containing p0
        words_to_check = list(token_to_words[p0])
        updates = defaultdict(int) # Track changes to update heap later

        for word in words_to_check:
            # 1. Validation checks
            if word not in splits: continue
            if p1 not in word: continue # Heuristic: p1 must also be in word

            # 2. Rebuild the word merging p0+p1
            new_word_list = []
            i_idx = 0
            changed = False
            n = len(word)
            
            while i_idx < n:
                if i_idx < n - 1 and word[i_idx] == p0 and word[i_idx+1] == p1:
                    new_word_list.append(new_token_bytes)
                    i_idx += 2
                    changed = True
                else:
                    new_word_list.append(word[i_idx])
                    i_idx += 1
            
            if changed:
                new_word = tuple(new_word_list)
                count = splits[word]
                
                # 3. Update Stats: Remove old pairs
                del splits[word]
                for j in range(len(word) - 1):
                    old_pair = (word[j], word[j+1])
                    pair_stats[old_pair] -= count
                    updates[old_pair] = pair_stats[old_pair]

                # 4. Update Stats: Add new pairs
                splits[new_word] = count
                for j in range(len(new_word) - 1):
                    new_pair = (new_word[j], new_word[j+1])
                    pair_stats[new_pair] += count
                    updates[new_pair] = pair_stats[new_pair]

                # 5. Update Index (Add new word to relevant buckets)
                # We don't remove `word` from p0/p1 buckets to save time (lazy removal)
                for token in new_word:
                    token_to_words[token].add(new_word)

        # D. Push updates to heap
        for pair, count in updates.items():
            if count > 0:
                heapq.heappush(stats_heap, (-count, pair))

        # Update progress bar occasionally
        if i % 100 == 0:
            progress_bar.set_postfix({"Best Count": current_count})

    total_time = time.time() - start_time
    log.info(f"Finished training in {total_time:.2f}s")

    if save_prefix:
        save_tokenizer(vocab_map, merges, save_prefix)

    return vocab_map, merges

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train BPE tokenizer.")
    parser.add_argument("input_path", type=str, help="Path to training text")
    parser.add_argument("--output_dir", "-o", type=str, default="bpe_tokenizer")
    parser.add_argument("--vocab_size", "-v", type:int, default=5000)
    parser.add_argument("--prefix", "-p", type=str, default="tokenizer")
    parser.add_argument("--special_tokens", "-s", nargs="*", default=["<|endoftext|>"])

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        log.error(f"Input file not found: {args.input_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    full_save_prefix = os.path.join(args.output_dir, args.prefix)

    train_bpe(
        args.input_path, 
        args.vocab_size, 
        args.special_tokens, 
        save_prefix=full_save_prefix
    )
```

</details>

#### Why this helps

- **Removed logging**: The loop now only updates `tqdm` every 100 steps. No more console spam.
- **Inverted index**: Instead of scanning 100,000 words to find where to merge "t" and "h", it jumps directly to the ~5,000 words that contain "t".
- **Heap for best pair**: Getting the best pair becomes \(O(1)\) amortized via a heap with lazy updates, instead of scanning the entire stats dictionary \(O(N)\).

#### Benchmarking

Use `hyperfine` to compare baseline vs future optimisations. This runs within the `uv` environment from the repo.

```bash
hyperfine --warmup 1 --runs 2 --show-output \
  'uv run -- python -u LM_training/tokenizer/bpe/training.py ./data/TinyStoriesV2-GPT4-train.txt \
    --output_dir bpe_tokenizer \
    --vocab_size 32000 \
    --prefix TinyStoriesV2-GPT4-train-v1 \
    --special_tokens "<|endoftext|>"'
```

#### Results

```text
Time (mean ± σ):     1157.925 s ±  4.846 s    [User: 1155.042 s, System: 4.063 s]
Range (min … max):   1154.498 s … 1161.352 s    2 runs
```

#### Results (Optimized)

```text
Time (mean ± σ):     21.320 s ±  0.389 s    [User: 464.053 s, System: 65.252 s]
Range (min … max):   21.045 s … 21.595 s    2 runs
```

#### Notes

- See repo usage for end‑to‑end workflows: [USAGE.md](https://github.com/SDcodehub/LM-training/blob/main/USAGE.md).


