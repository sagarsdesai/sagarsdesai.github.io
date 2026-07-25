---
title: "make_over — Name Generation"
date: 2024-01-01
description: "Character-level language models for name generation: from random baseline to learned bigrams"
tags: ["LLM", "NLP", "PyTorch", "Hugging Face"]
showDate: false
showReadingTime: false
---

A deep dive into character-level language models for name generation — built from scratch to understand the fundamentals of sequence modeling before scaling up to transformers.

## What it covers

- **Random baseline** — uniform sampling over the character vocabulary
- **Bigram model** — count-based next-character prediction from training data
- **Neural bigram** — same logic, but learned via gradient descent
- Comparison of all three on a held-out name dataset

## Why I built it

Andrej Karpathy's makemore tutorial, done from scratch with my own notes and extensions. The goal was to own the mental model end-to-end, not just run the code.

## Stack

Python · PyTorch · Hugging Face (model card)

## Links

- [GitHub](https://github.com/SDcodehub/make_over)
- [Hugging Face](https://huggingface.co/SDcodehub)
