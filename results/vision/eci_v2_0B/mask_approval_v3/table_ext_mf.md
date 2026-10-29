# Mask v5 (no SAM at inference): test-split metrics (pre-registered in scripts/eci/v2/mask5_prereg.md)

## mice: route (b) probe, cfg {'t': 0.9, 't_core': 0.8, 'or_n2': False, 'smooth': False}; PASS = True

| arm | token_f1 | token_recall | token_precision | thin_recall | bg_tok_per_frame | iso_bg_comp_per_frame | tok_per_frame |
|---|---|---|---|---|---|---|---|
| N2 | 0.778 | 0.904 | 0.683 | 0.489 | 66.900 | 0.900 | 256.600 |
| new | 0.921 | 0.924 | 0.918 | 0.574 | 10.100 | 0.600 | 195.200 |
| B_s_crop | 0.887 | 0.916 | 0.860 | 0.511 | 23.100 | 0.100 | 206.500 |

test: 10 frames, 47 thin points; store: {'union5_over_t': 1.2760997010208261, 'new_inside_S_N2': 0.9792767782924995, 'union_tok_per_frame': 247.72, 'store_TB': 1.9735158784, 'probe_s_per_frame': 0.1330737590789795, 'x_identical_min': 1.0}

## frogs: route (b) probe, cfg {'t': 0.9, 't_core': 0.8, 'or_n2': False, 'smooth': False}; PASS = True

| arm | token_f1 | token_recall | token_precision | thin_recall | bg_tok_per_frame | iso_bg_comp_per_frame | tok_per_frame |
|---|---|---|---|---|---|---|---|
| N2 | 0.773 | 0.910 | 0.672 | 0.621 | 7.800 | 0.600 | 27.100 |
| new | 0.884 | 0.915 | 0.855 | 0.793 | 2.100 | 0.000 | 21.400 |
| B_s_crop | 0.827 | 0.955 | 0.729 | 0.862 | 6.600 | 0.100 | 26.200 |

test: 10 frames, 29 thin points; store: {'union5_over_t': 1.5205938585085306, 'new_inside_S_N2': 0.9982758620689655, 'union_tok_per_frame': 35.95238095238095, 'store_TB': 0.20322011428571427, 'probe_s_per_frame': 0.0655019760131836, 'x_identical_min': 1.0}

