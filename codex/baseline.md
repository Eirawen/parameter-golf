# Baseline Architecture Reference

Source: `/home/khaled/parameter-golf/train_gpt.py` (~1126 lines)

## Model
- 9 transformer blocks, 512 dim, 8 attention heads, 4 KV heads (GQA)
- 2x MLP expansion (hidden=1024), relu^2 activation
- RoPE positional embeddings (base=10000)
- RMSNorm for layer normalization
- Logit softcap at 30.0 (tanh-bounded logits)
- U-Net skip connections (encoder-decoder with reverse skip reuse)
- Tied embeddings, vocab size 1024 (SentencePiece BPE)
- ~13M parameters

## Training
- Muon optimizer for weight matrices (momentum=0.95, lr=0.02, newton-schulz orthogonalization)
- AdamW for embeddings (lr=0.6) and scalars (lr=0.04)
- No weight decay in baseline
- Warmup: 20 steps, warmdown: 1200 iters
- Sequence length: 1024, batch: 524K tokens
- Gradient accumulation: 8 // world_size steps
- 10-minute wallclock cap (override with MAX_WALLCLOCK_SECONDS=0)
- DDP with torchrun, tf32 matmul, FlashAttention

## Evaluation
- BPB metric via SentencePiece LUTs for byte counting
- Validates on fixed first-50k-doc FineWeb validation set

## Quantization (post-training)
- Per-row int8 for 2D weight matrices
- Per-tensor int8 for vectors/scalars
- Tied embeddings kept fp16
- zlib-9 compression
- Final: ~16MB artifact, val_bpb ~1.2244

## Key Config Env Vars
- RUN_ID, DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE
- ITERATIONS, MAX_WALLCLOCK_SECONDS
- VAL_LOSS_EVERY, VAL_BATCH_SIZE
- TRAIN_BATCH_TOKENS, SEQ_LEN
- NUM_LAYERS, MODEL_DIM, NUM_HEADS, NUM_KV_HEADS
- MLP_MULT, LOGIT_SOFTCAP

## Running Baseline
```bash
# 8xH100
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 1xGPU (local testing)
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
