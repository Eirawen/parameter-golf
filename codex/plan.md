# Attack Plan (Updated 2026-03-26, scout run 4)

**Target**: Sub-0.10 BPB hybrid OR sub-1.10 BPB pure neural
**Hybrid frontier**: 0.0887 BPB (PR #913, "Cache Is All You Need")
**Pure neural frontier**: 1.0226 BPB (PR #875, GDN — unverified) or 1.1093 (PR #857, depth recurrence — verified)
**Starting point**: Fork converged recipe (11L + LeakyReLU(0.5)² + XSA-all + Full GPTQ + LZMA)

## The Meta Has Shifted MASSIVELY (since March 25)
1. **Two-pass full-rescore** is the new dominant technique — sub-0.1 BPB now achievable
2. **Order-12+ n-grams** are standard (was 2-7)
3. **Cache-dominated models** work — 622KB neural + massive cache gets 0.0887 BPB
4. **Dirichlet Posterior Mixing** is the most principled approach (0.1181 BPB)
5. **N-gram legality mostly confirmed** but full-rescore is "disputed"
6. **Pure neural**: GDN at 1.0226, depth recurrence at 1.1093
7. **Throughput-quantization co-optimization** is the pure-neural binding constraint

## STRATEGIC CHOICE: Hybrid vs Pure Neural

### Path A: Hybrid (N-gram + Neural) — HIGHEST BPB POTENTIAL
- **Target**: Sub-0.10 BPB
- **Risk**: Legality dispute on full-rescore / high-order. PR #886 RFC proposes 64MB cap.
- **Reward**: 10x better than pure neural

### Path B: Pure Neural — SAFER, MORE NOVEL
- **Target**: Sub-1.10 BPB (beat PR #857's 1.1093)
- **Risk**: Throughput-quant co-optimization is very hard to improve
- **Reward**: Unambiguously legal, more prestigious

### RECOMMENDATION: Do both. Start with hybrid (faster to implement, higher impact), have pure neural as fallback if n-gram legality changes.

## Phase 1 — Setup & Reproduce (DONE)
- [x] Get compute access (RunPod RTX PRO 6000 Blackwell)
- [x] Download dataset (sp1024 variant)
- [x] Run baseline training
- [ ] Fork PR #728's approach and reproduce

## Phase 2 — Hybrid Path: Two-Pass Full-Rescore N-gram

### 2a. Implement basic n-gram backoff (orders 2-12)
- Backward-looking n-gram cache built during eval
- Start with entropy-adaptive alpha mixing
- Expected: ~0.65-0.90 BPB from baseline

### 2b. Add two-pass full-rescore
- Pass 1: Score all tokens + build complete n-gram cache
- Pass 2: Rescore ALL tokens with complete cache using np.bincount
- Expected: ~0.10-0.15 BPB (this is the big unlock)

### 2c. Dirichlet Posterior Mixing (PR #900 approach)
- Replace linear interpolation with Dirichlet-Multinomial posterior
- Per-order concentration parameters (high for bigrams, low for phrases)
- Expected: ~0.02-0.03 BPB better than linear interp

### 2d. Phrase cache (PR #913 approach)
- Probe lengths 16-64 tokens for phrase matches
- Separate Dirichlet mixing for phrase level (c=1.0)
- Expected: pushes toward 0.09 BPB

### 2e. N-gram-aware training (PR #834 approach)
- Add Linear(512→7) gate head to predict expert weights
- Freeze n-gram oracle, train gate end-to-end
- Neural learns to complement n-gram weaknesses
- Expected: better than fixed complementary training

## Phase 3 — Pure Neural Path

### 3a. BI-Guided Depth Recurrence (PR #857 approach)
- Compute Block Influence scores to find redundant layers
- Tie highest-BI layers → more effective depth, same params
- Dedup-aware quantization stores shared weights once
- Target: beat 1.1093

### 3b. Progressive Depth Training (PR #895 approach)
- Start with 2 repeats, increase to 4-5 during training
- 36% more steps in same wallclock
- Combine with converged recipe

### 3c. GatedDeltaNet investigation (PR #875)
- PR #831 says GDN breaks torch.compile (+240%)
- PR #875 claims 1.0226 BPB — need to verify how they handled compile
- If viable: massive pure neural improvement
- HIGH RISK: may waste GPU time investigating

### 3d. SWA from many checkpoints
- PR #895 showed SWA from 38 checkpoints = -0.060 BPB (!!!)
- This is a FREE technique if we store checkpoints
- Combine with any architecture

### 3e. GatedAttn + ValueResid (PR #824)
- Per-head learned scalar for attention contribution
- Per-block learned scalar injecting initial embedding into residual
- ~0.018 BPB improvement over baseline
- Drop-in additions to converged recipe

## Phase 4 — Moonshots
- [ ] Ternary quantization with iterative correction (PR #911)
- [ ] MoE (sparse, more capacity per byte)
- [ ] Cubric (from newjordan's Podracing series)
- [ ] kNN-LM combined with n-gram (PR #873 uses both)
- [ ] PPM (Prediction by Partial Matching) full-rescore (PR #916)

## Decision Log
- 2026-03-22: Discovered SOTA has moved to 1.1213 via open PRs. Plan completely revised.
- 2026-03-22: TTT is the biggest unlock. Must be part of our approach.
- 2026-03-22: GPTQ-lite is free. Add it to any submission.
- 2026-03-22: Depth recurrence was tried badly (PR #386). Opportunity to do it right.
- 2026-03-22: Applied for quick-start compute grant (8 hours).
- 2026-03-25: **N-gram backoff is the new biggest revolution** — sub-0.5 BPB possible.
- 2026-03-25: Pure-neural frontier moved to 1.1078 (PR #720, Hedge Mixer Stack).
- 2026-03-25: LeakyReLU(0.5)², XSA-all, Full GPTQ, LZMA are now standard recipe.
- 2026-03-25: **CRITICAL**: XSA-all and TTT may conflict (PR #756: zero TTT gain on XSA-all).
- 2026-03-25: Depth recurrence works! PR #752 achieved 1.1182 with minimal layer repeat.
- 2026-03-25: N-gram backoff is zero artifact cost and ~0.14 BPB gain. Top priority.
- 2026-03-25: Complementary training on top of n-gram pushes to 0.44 BPB.
- 2026-03-25: Random GPTQ calibration only 0.002 BPB worse — avoids legality issues.
- 2026-03-25: PR #757 (aggressive TTT lr=1.0) is DRAFT — potential rule violation, wait for resolution.
- 2026-03-26: **MASSIVE META SHIFT**: Two-pass full-rescore pushes hybrid to sub-0.1 BPB.
- 2026-03-26: N-gram legality mostly confirmed by Will DePue (OpenAI) — eval-time memory unlimited.
- 2026-03-26: Full-rescore and order-12+ approaches are "disputed" — PR #886 RFC proposes 64MB cap.
- 2026-03-26: PR #913 shows 622KB neural model + cache = 0.0887 BPB. Neural model barely matters.
- 2026-03-26: PR #857 depth recurrence beats PR #752: 1.1093 via BI-guided weight tying.
- 2026-03-26: PR #875 GDN claims 1.0226 pure neural but conflicts with PR #831 research (torch.compile issues).
- 2026-03-26: PR #892 technique taxonomy provides S/A/B/C tier list and interaction matrix. Gold reference.
- 2026-03-26: PR #831 research: each ms overhead costs ~0.007 BPB. Throughput is king for pure neural.
- 2026-03-26: SWA from 38 checkpoints = -0.060 BPB (PR #895). Massive free gain.
- 2026-03-26: Output-LN critical for deep recurrence stability (PR #855).
- 2026-03-26: Dirichlet mixing >> linear interpolation (8.9x better, PR #900).
- 2026-03-26: N-gram-aware training with learned gate (PR #834) > fixed-alpha complementary training.
