# Mask v5 (no SAM at inference): test-split metrics (pre-registered in scripts/eci/v2/mask5_prereg.md)

## mice: route (b) probe, cfg {'t': 0.6, 't_core': 0.5, 'or_n2': True, 'smooth': True}; PASS = True

| arm | token_f1 | token_recall | token_precision | thin_recall | bg_tok_per_frame | iso_bg_comp_per_frame | tok_per_frame |
|---|---|---|---|---|---|---|---|
| N2 | 0.778 | 0.904 | 0.683 | 0.489 | 66.900 | 0.900 | 256.600 |
| new | 0.849 | 0.923 | 0.786 | 0.574 | 37.300 | 0.000 | 227.600 |
| B_s_crop | 0.887 | 0.916 | 0.860 | 0.511 | 23.100 | 0.100 | 206.500 |

test: 10 frames, 47 thin points; store: {'union5_over_t': 1.2496127504906658, 'new_inside_S_N2': 0.991581984557464, 'union_tok_per_frame': 277.04, 'store_TB': 2.2071001088, 'probe_s_per_frame': 0.13135547637939454, 'x_identical_min': 1.0}

## frogs: route (b) probe, cfg {'t': 0.6, 't_core': 0.5, 'or_n2': True, 'smooth': True}; PASS = False

| arm | token_f1 | token_recall | token_precision | thin_recall | bg_tok_per_frame | iso_bg_comp_per_frame | tok_per_frame |
|---|---|---|---|---|---|---|---|
| N2 | 0.773 | 0.910 | 0.672 | 0.621 | 7.800 | 0.600 | 27.100 |
| new | 0.859 | 0.870 | 0.849 | 0.621 | 2.400 | 0.000 | 20.500 |
| B_s_crop | 0.827 | 0.955 | 0.729 | 0.862 | 6.600 | 0.100 | 26.200 |

test: 10 frames, 29 thin points; store: {'union5_over_t': 1.5675572105801567, 'new_inside_S_N2': 1.0, 'union_tok_per_frame': 33.523809523809526, 'store_TB': 0.18949266285714286, 'probe_s_per_frame': 0.07764315605163574, 'x_identical_min': 1.0}

