# Experiment Log

Format: `[date] experiment name — result — takeaway`

---

## EXP-001: Baseline Smoke Test (2 min)
- **Date**: 2026-03-26
- **Hardware**: 1x RTX PRO 6000 Blackwell (96GB VRAM), RunPod spot
- **Config**: Default 9L/512d/8h/4kv, relu², 1024 seq_len, 1 shard, MAX_WALLCLOCK=120s
- **RUN_ID**: baseline_smoke
- **Results**:
  - Steps: 236 in 120s (510ms/step)
  - val_bpb: 1.9581 (int8+zlib roundtrip)
  - Artifact: 7.96 MB
  - Peak VRAM: 10.3 GB
- **Takeaway**: Environment works. Confirmed PyTorch 2.11+cu128 supports Blackwell sm_120.

## EXP-002: Baseline Full (10 min)
- **Date**: 2026-03-26
- **Hardware**: 1x RTX PRO 6000 Blackwell (96GB VRAM), RunPod spot
- **Config**: Default 9L/512d/8h/4kv, relu², 1024 seq_len, 1 shard, MAX_WALLCLOCK=600s
- **RUN_ID**: baseline_full
- **Results**:
  - Steps: 1,176 in 600s (510ms/step)
  - val_bpb (pre-quant): 1.3652
  - val_bpb (int8+zlib roundtrip): **1.3665**
  - Quantization penalty: 0.0013 BPB
  - Artifact: 13.0 MB / 16 MB limit (3 MB headroom)
  - Peak VRAM: 10.3 GB / 102 GB available
- **Takeaway**:
  - 1 GPU gets ~1/6th the steps of 8xH100 → 1.37 vs ~1.17 BPB. Gap is purely from fewer steps.
  - Quant penalty (0.0013) is excellent — matches top submissions.
  - 92 GB VRAM unused — massive room for bigger models.
  - 3 MB artifact headroom — can add layers or widen MLP.
  - Architecture changes on 1 GPU should transfer to 8xH100 (relative improvements valid).
