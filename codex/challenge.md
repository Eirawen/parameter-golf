# Challenge Rules & Constraints

## Objective
Train the best language model measured by **bits per byte (BPB)** on the FineWeb validation set (first 50k documents). Lower = better.

## Hard Constraints
- **Artifact size**: code bytes + compressed model bytes <= 16,000,000 bytes (decimal 16MB, not 16 MiB)
- **Training time**: <= 10 minutes on 8xH100 SXM
- **Evaluation time**: <= 10 minutes on 8xH100 SXM (separate from training budget)
- **No network calls** during evaluation. Artifact must be fully self-contained.
- **No accessing validation data during training** (no "paid prefix" tricks)
- **No accessing training data during evaluation** unless paid for in the 16MB budget

## What's Allowed
- Any architecture, tokenizer, optimizer, quantization scheme
- Any evaluation method (sliding window, any sequence length)
- Any Python package (include requirements.txt; can't sneak in extra compute via custom libs)
- Test-time training on validation tokens **already evaluated** (already graded)
- Offline hyperparameter tuning is fine; brute-forcing seeds is not
- Eval at any sequence length

## Submission Requirements
- Beat SOTA by >= 0.005 nats
- Statistical significance p < 0.01 (typically 3+ seeds)
- PR adding a folder to `/records/track_10min_16mb/`
- Must include: README.md, submission.json, train log, train_gpt.py

## Key Dates
- Challenge runs: March 18 – April 30, 2026
- OpenAI hiring cohort: June 2026

## Scoring
- BPB = bits_per_token / log(2) * tokens_per_byte
- Tokenizer-agnostic: accounts for actual byte counts via SentencePiece LUTs
- Compression = prediction. Better BPB = better language model.

## Compute
- OpenAI sponsoring $1M in RunPod credits
- Quick start: 8 compute hours (free)
- Development grant: $500 / 160 hours (need concrete approach + prior experiments)
- Advanced competitor: actively competing near top
