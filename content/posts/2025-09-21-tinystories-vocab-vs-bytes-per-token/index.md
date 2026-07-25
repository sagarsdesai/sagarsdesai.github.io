---
title: 10K vs 32K Tokenizers Yield Similar Bytes per Token
date: 2025-09-21
author: Sagar Desai
categories: [LLM, Tokenization]
tags: [BPE, Tokenizer, TinyStories]
series: ["Tokenization Deep Dive"]
---

#### TL;DR

- **Observation:** On TinyStories, 10K and 32K tokenizers produce almost identical compression (bytes/token).
- **Why:** The corpus is simple and repetitive; a 10K vocab already captures common subwords, so the extra 22K tokens rarely trigger.
- **Bigger gap when:** Corpora are complex/diverse (OpenWebText, code, multilingual), where larger vocabs reduce token count noticeably.

---

#### At a glance: TinyStories results

| Tokenizer | Vocab size | Bytes/token | Sample |
| --- | --- | --- | --- |
| BPE | 5K | 3.970 | 10 docs (seed 42) |
| BPE | 10K | 4.058 | 10 docs (seed 42) |
| BPE | 32K | 4.072 | 10 docs (seed 42) |

All are in the same ballpark; small differences arise from sampling, tokenizer variant (5K vs 10K/32K), and dataset slice. Net: little benefit from increasing vocab on this dataset.

---

#### Why the gap is small on TinyStories

1. **Vocabulary saturation:** The lexicon is limited and repetitive. A 10K vocab already covers almost all common words/subwords ("the", "and", "play", simple names), so merges capture most compression.
2. **Diminishing returns:** The extra 22K tokens in a 32K vocab skew toward rare/complex segments that barely occur in TinyStories, so they seldom apply.
3. **Short, simple morphology:** Few long, rare words means fewer opportunities for single-token replacements of multi-token fragments.

---

#### When you will see a bigger gap

- **OpenWebText / web-scale prose:** Broader vocabulary and topics. Larger vocabs can tokenize complex words as single units (e.g., "transformer", "backpropagation", "jurisdiction").
- **Code corpora:** Identifiers and symbols benefit from longer subword units; larger vocabs capture common stems/snippets (e.g., "get_user_id", "</div>").
- **Multilingual or domain jargon:** Medical/legal or multilingual text has many low-frequency segments that a 10K vocab would split into many pieces.

Result: On complex corpora, a 32K tokenizer often yields fewer tokens for the same bytes, improving bytes/token.

---

#### How bytes/token is computed

- **Definition:** Average UTF-8 byte length of the raw text divided by the number of tokens produced by the tokenizer.
- For simple English, average bytes per character ≈ 1 (ASCII), so differences mainly come from token count, not byte size.
- On multilingual text, characters may be 2–4 bytes in UTF-8; both numerator and denominator shift.

Formula: `bytes_per_token = total_utf8_bytes / total_tokens`

---

#### Code reference

- Repository: [SDcodehub/assignment1-basics](https://github.com/SDcodehub/assignment1-basics)
- Script: [cs336_basics/compute_bytes_per_token.py](https://github.com/SDcodehub/assignment1-basics/blob/main/cs336_basics/scripts/compute_bytes_per_token.py)

---

#### Practical guidance

- If your deployment domain resembles TinyStories (simple, repetitive), a 10K vocab is often sufficient and cheaper.
- For real-world text (OpenWebText), code, or multilingual corpora, prefer larger vocabs (e.g., 32K) for better compression.

---

#### Learned points from the latest run

- A 5K TinyStories tokenizer measured 3.970 bytes/token, close to prior 10K/32K numbers.
- This reinforces vocabulary saturation on TinyStories: smaller vocabs already capture frequent patterns.
- Differences across 5K/10K/32K on TinyStories are modest and sensitive to sampling and tokenizer variant.
- OpenWebText needs explicit paths; include the required flags to measure cross-domain differences.
