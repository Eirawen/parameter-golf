# Parameter Budget Math

Generated 2026-03-22. Budget: 16,000,000 bytes total, ~55KB code, ~15.95MB for model.

## Effective bits per parameter (after compression)
| Quant | Raw Bits | zstd Ratio | Eff Bits/Param | Max Params in 15.95MB |
|-------|----------|-----------|----------------|----------------------|
| int4  | 4        | ~2.0x     | 2.00           | 63.8M |
| int5  | 5        | 1.88x    | 2.66           | 48.0M |
| int6  | 6        | 1.51x    | 3.97           | 32.1M |
| int8  | 8        | 1.2x     | 6.67           | 19.1M |
| fp16  | 16       | 1.0x     | 16.0           | 8.0M  |

## SOTA #1 uses ~25.5M params, compresses to ~15.9MB (int5 MLP + int6 attn + fp16 embed)

## What fits at int5/int6 mixed quant (current best recipe)
| Config | Params | Compressed | Headroom |
|--------|--------|-----------|----------|
| 10L 3x MLP (SOTA) | 25.5M | 11.5MB | 4.5MB |
| 12L 3x MLP | 30.2M | 13.4MB | 2.6MB |
| 14L 3x MLP | 35.0M | 15.2MB | 0.8MB |
| 10L 4x MLP | 30.8M | 13.3MB | 2.7MB |
| 10L dim=576 3x MLP | 31.9M | 14.1MB | 1.9MB |

## With int4 MLP (aggressive, needs QAT)
| Config | Params | Compressed | Headroom |
|--------|--------|-----------|----------|
| 14L 3x MLP | 35.0M | 13.4MB | 2.6MB |
| 10L 4x MLP | 30.8M | 11.5MB | 4.5MB |

## Depth Recurrence (weight-tied layers)
| Config | Unique Params | Eff Layers | Compressed | Headroom |
|--------|--------------|-----------|-----------|----------|
| 5 unique x 4 = 20 eff | 13.7M | 20 | 6.2MB | 9.8MB |
| 4 unique x 5 = 20 eff | 11.3M | 20 | 5.4MB | 10.6MB |
| 3 unique x 7 = 21 eff | 9.0M | 21 | 4.6MB | 11.4MB |
| 5 unique x 4, 4x MLP | 16.3M | 20 | 7.1MB | 8.9MB |

## Key Insights
1. SOTA only uses 11.5MB of the 16MB budget — there's ~4.5MB of headroom for more layers
2. 12-14 layers should fit easily with current quantization
3. The #1 submission chose 10 layers, not more — likely hit training time limit, not byte limit
4. Depth recurrence is dramatically parameter-efficient: 20 effective layers for 6.2MB
5. The open question: does recurrence quality match unique layers? Research says ~80-90% effective
6. With recurrence, the freed bytes could go to wider dims (768+) or bigger embeddings
7. int4 MLP would be a further multiplier but needs careful QAT to avoid quality loss
