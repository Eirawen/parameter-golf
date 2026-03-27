# State of the Art — As of 2026-03-26 (scout run 4)

## IMPORTANT: The README leaderboard is STALE (last updated 2026-03-20)
## The real frontier is in open PRs. ~919+ PRs now. Check GitHub PRs for current SOTA.

## Official Leaderboard (merged, from README)
| Rank | Score | Author | Key Innovation |
|------|-------|--------|---------------|
| 1 | 1.1428 | thwu1 | 10L, int5/int6, BigramHash(10240), SWA(0.4) |
| 2 | 1.1458 | Raahil Shah | 9L, int6+zstd-22, SmearGate+BigramHash(4096) |
| 3 | 1.1502 | aruniyer | 11L, int6 QAT, zstd-22 |

## Unmerged PRs — Pure Neural Frontier (no n-gram, as of 2026-03-26, scout run 4)
| PR | Score | Author | Key Innovation |
|----|-------|--------|---------------|
| #875 | **1.0226** | shalyhinpavel | **GatedDeltaNet (GDN)** — SSM replacing attention, dynamic batch/chunk curriculum, NO TTT. ⚠️ torch.compile issues? |
| #857 | **1.1093** | aruniyer | **15L Depth Recurrence** via BI-guided weight tying (layers 9-13 share 1 block), cosine TTT 20 epochs |
| #720 | **1.1078** | agalimova | XSA6 + BigramHash4K on Hedge Mixer Stack (5-expert) |
| #824 | **1.0897** | sahiee-dev | GatedAttn + ValueResid + XSA6 + HedgeMixer + Legal TTT (has BigramHash4K) |
| #757 | **1.1124** ⚠️DRAFT | fielding | **Aggressive SGD TTT** lr=1.0, 30 epochs, all unfrozen |
| #728 | **1.1142** | abaybektursun | Val-Calibrated Full GPTQ, XSA-all, BigramHash 3072×112, Parallel Muon, NO TTT |
| #783 | 1.1171 | petergpt | PR703 base + shard-order curriculum + GPTQ cache-backout |
| #790 | 1.1172 | danialht | Residual Input Mixing, mixed int6 GPTQ, grouped TTT, MLP 3.5x |
| #703 | 1.1176 | Gusanidas | MiLe decay + 8-bit Muon + 1.04x LR |
| #713 | 1.1180 | hypery11 | Batched LoRA TTT (10L) |
| #752 | 1.1182 | Naazimsnh02 | Depth Recurrence (layers 4,5 repeated → 13 virtual from 11 physical) |
| #693 | 1.1186 | EthanYangTW | CROWN-Q + Full GPTQ + SWA/EMA Blend |

## Unmerged PRs — Neural + N-gram Hybrid (as of 2026-03-26, scout run 4)

### Sub-0.1 BPB (!!!) — THE NEW FRONTIER
| PR | Score | Author | Key Innovation |
|----|-------|--------|---------------|
| #883 | **0.0308** ⚠️DISPUTED | THUQiXuan | Order-13 N-gram Oracle prefilled from 8B training tokens. **384MB sidechannel** — likely illegal |
| #913 | **0.0887** | RoyiRa | **Cache Is All You Need** — 2L/128d tiny GPT + dual-cache (n-gram 2-12 + phrase 16-64). 622KB artifact! |
| #870 | **0.0935** | simon-marcus | **BROADSIDE Full-Rescore** — two-pass, ALL 62M tokens rescored with complete cache. np.bincount optimization |
| #907 | **0.0960** | resouer | **Shared N-gram Tables** — order-12 + entropy-adaptive alpha, LeakyReLU(0.9)² |
| #881 | **0.0990** | simon-marcus | **WaterLOO** — full-rescore with self-exclusion |
| #888 | **0.0942** | aamodbhatt | Fast Full-Rescore N-gram |

### Sub-0.2 BPB
| PR | Score | Author | Key Innovation |
|----|-------|--------|---------------|
| #880 | **0.1003** | RoyiRa | PhraseCache + OrderAdaptive N-gram + RegimeTracker |
| #900 | **0.1181** | Robby955 | **Two-Level Dirichlet Posterior Mixing** — Bayesian smoothing, orders 2-15, c=5.0/1.0 |
| #868 | **0.1181** | aamodbhatt | Budgeted Two-Pass N-gram Backoff |
| #869 | **0.1290** | THUQiXuan | N-gram Two-Pass Score-First |
| #893 | **0.1310** | aryanbhosale | Two-Pass Order-12 + Parallel Muon |
| #853 | **0.1315** | quietsmile | Two-Pass Order-12 + 256K Chunks |
| #846 | **0.1434** | himanshudongre | Two-Pass N-gram Rescoring |
| #859 | **0.1582** | bigbag | Learned Mixer Head + No TTT |
| #918 | **0.1653** | haikosys | TurboQuant + Full-Rescore N-gram |
| #834 | **0.1663** | AnirudhRahul | **N-gram-Aware Training** — frozen n-gram oracle + learned 7-expert gate head |

### Sub-0.5 BPB
| PR | Score | Author | Key Innovation |
|----|-------|--------|---------------|
| #809 | 0.295 ⚠️ | AayushBaniya2006 | Chunk-Based N-gram Backoff + Score-First TTT (suspiciously good) |
| #850 | 0.3212 | callithyia | Complementary N-gram 65K + Int5 GPTQ + LoRA TTT |
| #916 | 0.3461 | Bortlesboat | 10L + PPM Full-Rescore Order-12 |
| #890 | 0.4405 | sofiabod | Order-Adaptive 9-gram + Distributed Prefill |
| #811 | 0.4377 | quietsmile | Complementary Training + Backoff N-gram Mixer |
| #803 | 0.4416 | pentxayc | Complementary Training + Backoff N-gram Mixer |
| #814 | 0.4820 | newjordan | X-WING 3D Cubric + Complementary Training |

### Sub-1.0 BPB
| PR | Score | Author | Key Innovation |
|----|-------|--------|---------------|
| #798 | 0.5466 | travispchen | Order-Adaptive Entropy Gating + BackoffNgramMixer |
| #818 | 0.5527 | lucamignatti | HWNODE |
| #800 | 0.5644 | newjordan | X-WING: Shared N-gram Tables + Cubric |
| #808 | 0.6364 | Naazimsnh02 | Depth Recurrence + Multi-Order N-gram Backoff |
| #813 | 0.6671 | hypery11 | 11L MHA 8/8 XSA-all LeakyReLU(0.5)² MLP3.5x + BackoffNgramMixer orders 2-7 |
| #806 | 0.6678 | ibarrajo | Backoff N-gram Cache + LeakyReLU(0.9)² |
| #770 | 0.6672 | minh-stakc | 11L + Multi-Order N-gram + Entropy-Adaptive Alpha |
| #779 | 0.6683 | deanbrr | BackoffNgramMixer + Drift-Free TTT |
| #909 | 0.8609 | sunnypatneedi | 11-gram Eval Cache + Hedge Mixer |
| #828 | 0.9076 | bigbag | 10L + N-gram Backoff + Matrix LR 0.03 |
| #889 | 0.9642 | anthony-maio | N-gram Backoff + VRL + LeakyReLU² |
| #885 | 0.9958 | lolrazh | LeakyReLU(0.9)² + N-gram Cache + Entropy-Reg QAT |

## Unmerged PRs — Notable Non-Competitive / Research
| PR | Score | Author | Key Innovation |
|----|-------|--------|---------------|
| #892 | — | robbiebusinessacc | **Technique Taxonomy** — S/A/B/C tier list, interaction effects matrix, BPB verification tools |
| #886 | — | abaybektursun | **RFC: N-gram legality framework** — proposes 64MB eval-time state cap. NO RULING YET. |
| #831 | — | sseanliu | **Why Novel Architectures Fail at 16MB** — throughput-quant co-optimization is binding constraint |
| #875 | 1.0226 | shalyhinpavel | GatedDeltaNet pure neural (torch.compile issues per PR #831) |
| #873 | 1.0467 | gowtham0992 | E2E TTT: MAML-style meta-learning + 7-gram + kNN-LM |
| #895 | 1.0889 | iverbovoy | 4-Hour Progressive Depth — 3 blocks × 2-5 repeats, dim=832, Hedge Mixer |
| #855 | 1.2659 | aazizyan | **First Viable 3-Loop Recurrence** — Birkhoff + Output-LN + Timestep Scaling |
| #812 | 1.2297 | andrewmouldon | BankLinear: cross-layer shared weight bank |
| #911 | — | Akhilesh-Gogikar | Ternary Reasoner — 12L/768d Capsule-Feedback, no eval results |
| #914 | 1.1873 | mkenney2 | Hymba-LongContext: hybrid SSM + SWA, 32K context |
| #906 | — | andrew-medrano | Pure raw-byte JEPA negative result |
| #908 | — | albertorkive | Higher-Rank Output Heads: standard tied head wins |

## The New Meta (updated 2026-03-26, scout run 4)

### Two-Pass Full-Rescore N-gram is THE dominant technique
- **Pass 1**: Run neural model, score all tokens, build complete n-gram cache from all ~62M val tokens
- **Pass 2**: Rescore EVERY token using the complete cache
- Key optimization: `np.bincount` vectorization → 33s cache build (10-50x faster)
- Prior approaches only rescored subsets (15/63 chunks → 50/237 → now ALL)
- Best scores: 0.0887-0.0960 BPB (vs 0.44 with complementary training alone)

### N-gram order escalation: 2-7 → 2-12 → 2-15
- Order-12 is now standard in top submissions (was 2-7 a day ago)
- Order-15 used in Dirichlet Posterior Mixing (PR #900)
- Higher orders have diminishing returns but still measurable

### Dirichlet Posterior Mixing (PR #900) — principled Bayesian approach
- Recursive Bayesian smoothing: each order's posterior becomes prior for next higher order
- Concentration parameter decreases with order (c=50 bigrams → c=1.0 phrases)
- Linear interpolation is 8.9x WORSE than Dirichlet mixing
- Two levels: n-gram backoff + phrase suffix matching

### N-gram-Aware Training (PR #834) — learned complementary training
- Small Linear(512→7) gate head predicts per-token expert weights
- Frozen n-gram oracle built once from training data
- Neural model learns WHEN n-grams are unreliable via gradient through gate
- More principled than fixed alpha complementary training

### "Cache Is All You Need" (PR #913) — minimal neural, maximal cache
- Tiny 2L/128d GPT (~500K params, 622KB artifact!) + massive dual-cache system
- N-gram cache (orders 2-12) + phrase cache (lengths 16-64)
- Shows the neural model barely matters when cache is comprehensive
- 0.0887 BPB from a 622KB model!

### GatedDeltaNet (PR #875) — potential pure neural breakthrough
- Replaces attention with GatedDeltaBlock (SSM architecture)
- 1.0226 BPB pure neural, no TTT, no n-gram
- BUT: PR #831 research found GDN breaks torch.compile (+240% overhead, 1.2516 BPB)
- PR #875 may have found workarounds (dynamic batch/chunk curriculum, FastLoader)
- ⚠️ Needs verification — conflicting evidence

### BI-Guided Depth Recurrence (PR #857) — principled weight tying
- Block Influence scores identify most redundant layers (9-13, BI: 0.10-0.16)
- Tie those layers → 15 effective from 11 unique → same params, more depth
- Dedup-aware quantization: tied weights stored once
- 1.1093 BPB pure neural — best depth recurrence result

### Progressive Depth Training (PR #895) — dynamic recurrence
- Shared-weight blocks with increasing repeats during training
- Phase 1: 2 repeats → Phase 2: 3 → Phase 3: 4-5 repeats
- 36% more steps in same wallclock (5861 vs 4300)
- SWA from 38 checkpoints contributed -0.060 BPB alone (!!)
- Not yet competitive in 10min track (1.0889 in 4hr)

### 3-Loop Recurrence Stabilization (PR #855)
- **Output-LN**: move RMSNorm to MLP output — "critical piece" for loop identity
- **Birkhoff-constrained mixing**: sigmoid parameterization, spectral norm ≤ 1
- **Capped Timestep Scaling**: [-4,+4] float16, zero pre-quant effect but -26-30% Q-gap
- 1.2659 BPB (not competitive but first viable 3-loop)

### N-gram Legality Status (CRITICAL UPDATE)
- **Will DePue (OpenAI, March 25)**: "All that matters is eval runs in 10 minutes and info between train/eval is under 16MB... runtime eval memory is unlimited"
- **valerio-oai**: "leaning towards accepting [n-gram caches] as legal"
- **PR #886 RFC**: proposes 64MB cap on eval-time state. No ruling yet.
- **Explicitly illegal**: Oracle/min-NLL selection. Must commit to one mixture distribution before scoring.
- **Disputed**: Full-rescore (two-pass) and order-12+ approaches
- **PR #883 (0.0308 BPB)**: Uses 384MB sidechannel from training data — likely illegal

### Throughput Tax Formula (PR #831 research)
- At 83ms/step baseline, each ms overhead ≈ 7 lost steps ≈ 0.007 BPB cost
- Techniques must clear this threshold per ms of overhead
- GDN: +240% overhead → ~199ms extra → ~1393 lost steps. Quality gain insufficient.
- nGPT: +47% → incompatible with int6 quantization
- This is why the converged recipe is so hard to beat on pure neural

### Technique Tier List (PR #892 meta-analysis)
**S-Tier** (0.005-0.020 BPB): Sliding window eval, int6 quant, 3x MLP, FP16 embeddings, 11 layers, seq_len 2048
**A-Tier** (0.002-0.005 BPB): Muon WD 0.04, EMA/SWA, ortho init, SmearGate, BigramHash, XSA-4, LeakyReLU(0.5)², QAT
**B-Tier** (<0.002 BPB): Partial RoPE, LN Scale, Value embeddings, LZMA, Warmdown
**C-Tier** (paradigm shift): Legal TTT (-0.020 to -0.050), N-gram backoff (-0.050 to -0.700), GPTQ (-0.002 to -0.005)

### Interaction Effects (PR #892)
- **Sub-additive**: XSA-all + TTT (expected -0.028, actual -0.022), EMA + SWA (choose one)
- **Nearly additive**: LeakyReLU² + 3xMLP, Partial RoPE + seq=2048, TTT + N-gram
- **Rule**: Training improvements stack additively; eval improvements stack additively. Training × eval interactions are sub-additive.

## Converged Recipe (base for all top pure-neural runs, updated 2026-03-26)
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion (hidden=1536), **LeakyReLU(0.5)²** (some use 0.9)
- SmearGate + BigramHash (3072×112 or larger)
- **XSA on ALL layers** (but note TTT interaction)
- Orthogonal init, muP-scaled outputs
- **Parallel Muon**: lr=0.025, WD=0.04, momentum=0.99 (warmup 0.85→0.99)
- EMA decay=0.997
- **Full Hessian GPTQ int6** + **LZMA** compression
- Seq_len=2048, batch ~786K tokens
- Sliding window eval stride=64
- Value Embeddings: 128-dim, 5 sets
- Partial RoPE (16/64 dims)
- TTT: SGD all blocks unfrozen (but XSA-all interaction — gains reduced)

## TTT + XSA-all Interaction (CRITICAL finding from PR #756)
- XSA-all already captures inter-document adaptation patterns that TTT targets
- PR #756 tested TTT on XSA-all stack: **zero improvement** across 25+ attempts
- PR #757 claims big TTT gains but uses XSA-4, not XSA-all
- This means: **XSA-all and aggressive TTT may be substitutes, not complements**
- PR #892 quantifies: expected -0.028, actual -0.022 (sub-additive by 0.006)

## Negative Results & Failed Experiments (updated 2026-03-26)
- **Meta-TTT (MAML-style, PR #384)**: FAILED. +0.085 BPB degradation.
- **Tokenizer optimization (PR #384)**: NULL. Stock v1024 optimal.
- **Depth recurrence 1×12 (PR #386)**: TERRIBLE. 1.4061 BPB.
- **Qronos quantization (PR #756)**: FAILED. +0.0007 BPB at int6.
- **CDQuant (PR #756)**: FAILED. +0.0005 BPB.
- **Spectral Init λ=10 (PR #756)**: TERRIBLE. 1.52 BPB, 650ms/step.
- **SLOT Bias (PR #756)**: FAILED. +0.0013 BPB.
- **TTT on XSA-all stack (PR #756)**: FAILED. Zero gain across 25+ attempts.
- **Gated Attention**: +0.0026 worse (but PR #824 uses it successfully with other techniques).
- **DiffTransformer**: 1.5x slower.
- **Attention Residuals**: 54% slower.
- **Gibbs-refined GPTQ calibration**: Worse than random tokens.
- **nGPT Hypersphere (PR #831)**: 1.6915 BPB. Unit-norm fails int6 (+0.35 gap).
- **Hourglass FFN (PR #831)**: 1.4519 BPB. Split weights incompatible with int6.
- **MUD Optimizer (PR #831)**: 1.1581 BPB. Tensor core incompatibility.
- **SSM/GDN without torch.compile (PR #831)**: 1.2516 BPB (+240% overhead).
- **Higher-Rank Output Heads (PR #908)**: Standard tied head wins.
- **JEPA (PR #906)**: Negative result for raw-byte JEPA.
- **Diffusion LM (PR #905)**: 1.8587 BPB. Not competitive.
- **AR-Diffusion Hybrid (PR #904)**: 1.2734 BPB. Not competitive.

## Key Observations (Updated 2026-03-26, scout run 4)
1. **Two-pass full-rescore n-gram** pushes hybrid scores to sub-0.1 BPB — 10x better than our last update
2. Competition has ~920 PRs now — massive acceleration
3. The neural model is becoming a MINORITY contributor in top hybrid submissions (PR #913: 622KB model!)
4. **N-gram legality is mostly confirmed** but full-rescore and high-order (12+) approaches are "disputed"
5. **Pure neural frontier**: GDN at 1.0226 (unverified), depth recurrence at 1.1093 (verified 3-seed)
6. **Throughput-quantization co-optimization** is the binding constraint for pure neural (PR #831)
7. **Dirichlet Posterior Mixing** is the most principled hybrid approach (0.1181 BPB)
8. **Progressive depth** enables 36% more training steps via dynamic recurrence scheduling
9. **SWA from many checkpoints** (-0.060 BPB) rivals architectural innovations
10. **Output-LN** is critical for deep recurrence stability (PR #855)
11. **N-gram-aware training** (learned gate) > fixed-alpha complementary training
12. **LeakyReLU(0.9)²** appearing in some top submissions (was 0.5)
