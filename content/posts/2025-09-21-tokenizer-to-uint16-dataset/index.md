---
title: From tokenizer to uint16 dataset with encode_iterable
date: 2025-09-21
author: Sagar Desai
categories: [LLM, Tokenization, Data]
tags: [BPE, Tokenizer, NumPy, Dataset]
---

#### Goal

- **Convert raw text** → **token IDs** → **`uint16` dataset** using your `Tokenizer`'s `encode_iterable`.

#### What you already have

- **`encode_iterable(iterable_of_text)`**: streams token IDs from any iterable of strings (e.g., a file handle). This is memory-efficient and ideal for large corpora.

---

#### Option A: Simple end-to-end script (`.npy` file)

- **Good for**: small/medium datasets that fit in RAM.

```python
import numpy as np
from tokonizer import Tokenizer  # ensure tokonizer.py is on PYTHONPATH or same folder

# 1) Load tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath="bpe_tokenizer/tinystories_vocab.json",
    merges_filepath="bpe_tokenizer/tinystories_merges.txt",
    special_tokens=["<|endoftext|>"]
)

# 2) Stream-encode the dataset
input_file_path = "path/to/your/tinystories_train.txt"
with open(input_file_path, "r", encoding="utf-8") as f:
    token_ids = list(tokenizer.encode_iterable(f))  # collects into memory

# 3) Convert to uint16 and save
train_ids = np.array(token_ids, dtype=np.uint16)
np.save("tinystories_train_ids.npy", train_ids)
```

---

#### Option B: Streaming-safe (`.bin` file with uint16)

- **Good for**: very large datasets that don't fit in RAM.
- **Load later with**: `np.fromfile(path, dtype=np.uint16)`.

```python
from array import array
from tokonizer import Tokenizer

tokenizer = Tokenizer.from_files(
    vocab_filepath="bpe_tokenizer/tinystories_vocab.json",
    merges_filepath="bpe_tokenizer/tinystories_merges.txt",
    special_tokens=["<|endoftext|>"]
)

input_file_path = "path/to/your/tinystories_train.txt"
output_file_path = "tinystories_train_ids_uint16.bin"

buffer = array("H")  # unsigned short = uint16
flush_every = 1_000_000  # tune per memory

with open(input_file_path, "r", encoding="utf-8") as fin, open(output_file_path, "wb") as fout:
    for tid in tokenizer.encode_iterable(fin):
        if tid > 0xFFFF:
            raise ValueError(f"Token id {tid} exceeds uint16 range (65535)")
        buffer.append(tid)
        if len(buffer) >= flush_every:
            buffer.tofile(fout)
            buffer = array("H")

    if buffer:
        buffer.tofile(fout)

# Later: np.fromfile("tinystories_train_ids_uint16.bin", dtype=np.uint16)
```

---

#### Notes

- **Tokenizer API**: `encode_iterable` accepts any Python iterable of text lines, not only files.
- **`uint16` bound**: ensure your vocabulary size ≤ 65536 so token IDs fit `uint16`.
- **`.npy` vs `.bin`**: `.npy` includes NumPy header (easy `np.load`), `.bin` is raw; use `np.fromfile` with `dtype=np.uint16`.

---

#### Code reference

- Repository: [SDcodehub/assignment1-basics](https://github.com/SDcodehub/assignment1-basics)
- Script: [cs336_basics/compute_bytes_per_token.py](https://github.com/SDcodehub/assignment1-basics/blob/main/cs336_basics/scripts/tokenize_dataset.py)

---


