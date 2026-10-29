# Mask v5 (no SAM at inference): test-split metrics (pre-registered in scripts/eci/v2/mask5_prereg.md)

## ants: route (b) probe, cfg {'t': 0.7, 't_core': 0.8, 'or_n2': False, 'smooth': False}; PASS = False

| arm | token_f1 | token_recall | token_precision | thin_recall | bg_tok_per_frame | iso_bg_comp_per_frame | tok_per_frame |
|---|---|---|---|---|---|---|---|
| N2 | 0.867 | 0.818 | 0.921 | 0.479 | 0.800 | 0.000 | 41.600 |
| new | 0.911 | 0.949 | 0.876 | 0.761 | 2.600 | 0.000 | 50.700 |
| B_s_crop | 0.924 | 0.874 | 0.981 | 0.581 | 0.200 | 0.000 | 41.700 |

test: 10 frames, 117 thin points; store: {'union5_over_t': 1.330392572709147, 'new_inside_S_N2': 0.9985093167701864, 'union_tok_per_frame': 68.21739130434783, 'store_TB': 0.25427077565217393, 'probe_s_per_frame': 0.06699900627136231, 'x_identical_min': 1.0}

