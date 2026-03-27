# Gotchas & Pitfalls (Updated 2026-03-26, scout run 4)

## The README Leaderboard is STALE
- Last updated 2026-03-20, shows 1.1428 as SOTA
- Actual pure-neural frontier: ~1.0226 (PR #875, unverified) or ~1.1093 (PR #857, verified)
- Actual hybrid frontier: ~0.0887 BPB (PR #913, "Cache Is All You Need")
- ~920 PRs open. ALWAYS check GitHub PRs for current state.

## N-gram Legality — EVOLVING SITUATION
- **Will DePue (OpenAI, March 25)**: eval-time memory is unlimited, 16MB limit only applies to train→eval transfer
- **valerio-oai**: "leaning towards accepting [n-gram caches] as legal"
- **PR #886 RFC**: proposes 64MB cap on eval-time state — NO RULING YET
- **Explicitly illegal**: Oracle/min-NLL selection (picking neural vs n-gram per-token based on which is lower)
- **Must**: commit to one mixture distribution before scoring each token
- **Disputed**: Full-rescore (two-pass) approaches, order-12+ n-grams
- **PR #883 (0.0308 BPB)**: Prefills n-gram from 8B TRAINING tokens → 384MB sidechannel → almost certainly illegal
- **Safe**: Backward-looking single-pass n-gram cache (orders 2-7), hedge mixer
- **Bottom line**: Implement conservatively first, aggressive later if legality confirmed

## Artifact Size
- 16MB is **decimal** (16,000,000 bytes), NOT 16 MiB (16,777,216). Easy to overshoot.
- Artifact = code bytes + compressed model bytes. Code lives in train_gpt.py.
- PR #768 seed 1337 hit 16,024,522 — over the limit! Watch closely.

## Quantization
- FP16 for tied embeddings is critical — quantization-sensitive.
- Last-layer key projection benefits from FP16.
- **At int6, GPTQ is near-optimal** — Qronos, CDQuant don't help (PR #756).
- Random calibration data only 0.002 BPB worse than real data for GPTQ.
- 131K random tokens sufficient for calibration.
- **nGPT Hypersphere weights fail int6** (+0.35 gap, PR #831).
- **Hourglass FFN split weights fail int6** (PR #831).

## Evaluation
- Sliding window eval stride=64 is a free ~0.034 BPB improvement. Always use it.
- Eval budget is separate 10 minutes — TTT, n-gram cache, rescoring can use this.
- You CANNOT train on validation data before evaluating it.
- N-gram cache must be backward-looking — update counts AFTER scoring each window.
- **Two-pass full-rescore** is legal per Will DePue's statement but under debate (PR #886).

## Two-Pass Full-Rescore Gotchas (NEW)
- Pass 1 must score ALL tokens before pass 2 begins
- Self-inclusion handling: use `min_count >= 2` threshold for rare n-grams (PR #870)
- np.bincount is 10-50x faster than dict-based approaches for cache build
- Eval time constraint: full rescore of 62M tokens must fit in 600s
- GPU-shared n-gram tables: all 8 ranks update same tables (PR #907)

## Dirichlet vs Linear Interpolation (NEW)
- Linear interpolation for n-gram mixing is 8.9x WORSE than Dirichlet posterior (PR #900)
- Concentration parameter must decrease with order (c=50 bigrams → c=1.0 phrases)
- This is the most impactful single technique choice in hybrid systems

## Throughput Tax (NEW — PR #831 research)
- At 83ms/step baseline, each ms overhead ≈ 7 lost training steps ≈ 0.007 BPB
- Techniques must clear 0.007 BPB/ms threshold to be worthwhile
- This is why GDN (+240%), nGPT (+47%), Hourglass (+11%) all fail
- Only techniques that integrate with torch.compile + int6 + tensor cores survive

## Tokenizer Changes
- Don't change the tokenizer. Stock v1024 is optimal.
- Any changes will be "examined much more carefully."

## Submission
- Need p < 0.01 significance. Typically 3+ seeds.
- Must beat SOTA by >= 0.005 nats (not BPB — nats).
- Submissions accepted chronologically by PR creation time.

## Training
- LZMA preset=9 gives better compression than zstd-22. Use LZMA.
- Weight decay (0.04) directly improves quantization quality.
- Muon momentum warmup (0.85→0.99 over ~1500 steps).
- GPTQ calibration time counts toward 600s budget (reserve 14-40s).
- **Late QAT activation causes torch.compile recompilation → OOM**. Enable from start or budget carefully (PR #892).

## TTT (Test-Time Training)
- **XSA-all and TTT are sub-additive** — expected -0.028, actual -0.022 (PR #892)
- PR #756: ZERO TTT gain on XSA-all stack across 25+ attempts
- If using XSA-4 instead, aggressive TTT (lr=1.0, 30 epochs) can give -0.041 BPB (PR #757, unverified)
- SGD works better than AdamW for TTT (momentum cold-start per document)
- Don't freeze early blocks (ttt_freeze_blocks=0)
- Chunk size: 256 (128 is wasteful)
- Score-first protocol: score tokens before training (definitively legal)
- Meta-TTT (MAML-style) DOES NOT WORK — 0.085 BPB worse (PR #384)
- **E2E TTT meta-learning** (PR #873): achieves 1.0467 BPB with MAML + 7-gram + kNN-LM

## N-gram Backoff
- Must be backward-looking: update counts AFTER scoring each window
- Entropy-adaptive alpha > fixed alpha (~0.015 BPB better)
- **Orders 2-12** is the new sweet spot (was 2-7)
- **Dirichlet mixing >> linear interpolation** (8.9x improvement)
- Zero artifact cost — cache built from eval tokens at test time
- N-gram-aware training (learned gate, PR #834) > fixed-alpha complementary training (PR #811)
- **FineWeb n-gram statistics**: 99.5% of 2-grams repeated, 90.5% of 3-grams, 66.1% of 4-grams, 18.1% of 5-grams

## Depth Recurrence
- 1 block × 12 repeats = TERRIBLE (1.4061, PR #386)
- Minimal repeat (layers 4,5 → 13 virtual from 11) = 1.1182 (PR #752)
- **BI-guided tying** (layers 9-13 share 1 block → 15 from 11) = **1.1093** (PR #857) ← NEW BEST
- 3 blocks × 3 loops = 1.2659 (PR #855) — first viable but needs Output-LN
- Progressive depth (2→5 repeats) works but only competitive in 4hr track (PR #895)
- **Output-LN at MLP output is CRITICAL** for loop identity (PR #855)
- Birkhoff-constrained mixing prevents quantization blowup in deep recurrence
- Capped timestep scaling: zero pre-quant effect but -26-30% Q-gap
- Gradient clip 0.3 for stability
- Seq_len=1024 preferred for speed (140ms vs 253ms at 2048)

## Activation Functions
- **LeakyReLU(0.5)² is standard** — ~0.004 BPB better than relu²
- Some top submissions use **LeakyReLU(0.9)²** (PR #907, PR #885)
- Drop-in replacement, no other changes needed

## Architecture
- Failed: Spectral Init, Gated Attention (standalone), DiffTransformer, SLOT Bias, nGPT, Hourglass FFN
- **GatedAttn + ValueResid** together = +0.018 BPB (PR #824) — but only with XSA6/HedgeMixer
- **XSA-all > XSA-4** (~0.006 BPB) but has TTT interaction
- **Higher-Rank Output Heads** lose to standard tied head (PR #908)
- **JEPA**: negative result (PR #906)
- **Diffusion LM**: not competitive (PR #905, PR #904)

## SWA/Checkpoint Averaging (NEW)
- SWA from 38 checkpoints = -0.060 BPB (!!!) in 4hr track (PR #895)
- This is larger than most architectural innovations
- Must store checkpoints during training → memory pressure
- EMA + SWA are redundant — choose one (PR #892)

## Local Development
- 3070 is ~100-150x slower than 8xH100. Use reduced configs.
- MLX path exists for Mac users but we're on Linux/CUDA.
- Dataset: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1` for minimal local subset.
