# Mask v5 (no SAM at inference): test-split metrics (pre-registered in scripts/eci/v2/mask5_prereg.md)

## ants: route (b) probe, cfg {'t': 0.5, 't_core': 0.5, 'or_n2': True}; PASS = False

| arm | token_f1 | token_recall | token_precision | thin_recall | bg_tok_per_frame | iso_bg_comp_per_frame | tok_per_frame |
|---|---|---|---|---|---|---|---|
| N2 | 0.867 | 0.818 | 0.921 | 0.479 | 0.800 | 0.000 | 41.600 |
| new | 0.834 | 0.868 | 0.802 | 0.547 | 6.100 | 0.000 | 50.600 |
| B_s_crop | 0.924 | 0.874 | 0.981 | 0.581 | 0.200 | 0.000 | 41.700 |

test: 10 frames, 117 thin points; store: {'union5_over_t': 1.2477472441190494, 'new_inside_S_N2': 1.0, 'union_tok_per_frame': 64.34782608695652, 'store_TB': 0.23984751304347823, 'probe_s_per_frame': 0.07726445198059081, 'x_identical_min': 1.0}

