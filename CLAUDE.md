# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenAI Parameter Golf challenge: train the best language model fitting in a **16MB artifact** (code + compressed weights) in **≤10 minutes on 8xH100 SXM**, scored by **bits per byte (BPB)** on FineWeb validation set. Lower BPB = better. Deadline: April 30, 2026.

## Commands

```bash
# Download dataset (sp1024 vocab, --train-shards 1 for minimal local subset)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Train on single GPU
RUN_ID=test \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Train on 8xH100 (leaderboard submission)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# MLX local smoke test (Apple Silicon only)
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

All training config is via environment variables. Key overrides: `MAX_WALLCLOCK_SECONDS` (default 600), `ITERATIONS`, `NUM_LAYERS`, `MODEL_DIM`, `MLP_MULT`, `TRAIN_SEQ_LEN`, `VAL_LOSS_EVERY`, `SEED`.

## Architecture

**train_gpt.py** (~1100 lines) contains the entire model, optimizer, training loop, quantization, and evaluation in one file.

**Model:** Transformer with GQA (grouped query attention), RoPE, RMSNorm, relu² MLP, logit softcap, U-Net skip connections, tied embeddings. Default: 9 layers, 512 dim, 8 heads, 4 KV heads, 1024 vocab.

**Optimizer:** Three-group setup:
- Muon (Newton-Schulz orthogonalization) for weight matrices
- AdamW for token embeddings
- AdamW for scalar/vector params (norms, scales, mixing weights)

**Quantization → Compression → Evaluation flow:**
1. Train in bf16 (CastedLinear stores fp32 master weights)
2. Post-training: per-row int8 quantize 2D weights, per-tensor for rest, fp16 for tied embeddings
3. Compress with zlib-9 (top submissions use zstd-22)
4. Decompress, dequantize, evaluate on full validation set
5. Report `val_bpb` = (val_loss / ln2) × tokens_per_byte

**Data pipeline:** Binary shards with 256-int header + uint16 tokens. TokenStream reads shards sequentially. DistributedTokenLoader slices across DDP ranks.

**Distributed:** PyTorch DDP via torchrun, NCCL backend, gradient all-reduce over NVLink. torch.compile with fullgraph=True. FlashAttention only.

## Submission Structure

```
records/track_10min_16mb/YYYY-MM-DD_RunName/
├── README.md           # Approach explanation + ablations
├── submission.json     # Author, val_bpb, artifact size metadata
├── train_gpt.py        # Self-contained training script
└── train_seed*.log     # Logs from 3+ seeds for p<0.01 significance
```

Must beat SOTA by ≥0.005 nats. Artifact = code bytes + compressed model ≤ 16,000,000 bytes (decimal).

## Agent Working Memory

The `/codex` directory is shared working memory across agents. Always read `codex/INDEX.md` first for a filemap. Log experiments, decisions, and learnings there so other agents have context.

## Key Gotchas

- 16MB is decimal (16,000,000), not 16 MiB (16,777,216)
- FP16 for tied embeddings is critical — quantizing them tanks performance
- Sliding window eval (stride=64) is a free ~0.03 BPB gain; always use it
- Weight decay (0.04) directly improves quantization quality
- QAT (quantization-aware training) reduces quant penalty from ~0.016 to ~0.001 BPB
- zstd-22 compresses ~5% better than zlib-9
- Tokenizer changes are heavily scrutinized — must prove val_bpb correctness
- No validation data access during training; TTT only on already-graded tokens
