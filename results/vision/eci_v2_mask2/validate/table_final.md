| set | metric | M0 | N1 | N2 |
|---|---|---|---|---|
| mab | body_tok | 0.518 | 0.563 | 0.591 |
| mab | leak | 0.024 | 0.020 | 0.028 |
| mab | tail_base_tok | 0.389 | 0.560 | 0.589 |
| mab | tail_seg_px | 0.146 | 0.328 | 0.328 |
| mab | tail_seg_tok | 0.107 | 0.242 | 0.311 |
| mab | tail_tip_tok | 0.057 | 0.071 | 0.095 |
| mab | tokens_per_frame | 26.670 | 36.962 | 41.323 |
| cal | body_tok | 0.731 | 0.937 | 0.966 |
| cal | leak | 0.147 | 0.164 | 0.221 |
| cal | pale_missed | 0.719 | 0.386 | 0.300 |
| cal | pale_tok | 0.075 | 0.207 | 0.273 |
| cal | tail_base_tok_int | 0.151 | 0.289 | 0.360 |
| cal | tail_base_tok_res | 0.741 | 0.947 | 0.961 |
| cal | tokens_per_frame | 97.559 | 155.474 | 229.498 |
| crim | c57_tok | 0.995 | 0.995 | 0.995 |
| crim | c57_tok_tol1 | 0.995 | 0.995 | 0.995 |
| crim | leak | 0.175 | 0.150 | 0.178 |
| crim | pale_tok | 0.031 | 0.068 | 0.089 |
| crim | pale_tok_tol1 | 0.151 | 0.328 | 0.365 |
| crim | res_tok | 0.964 | 0.969 | 0.969 |
| crim | res_tok_tol1 | 0.977 | 0.984 | 0.987 |
| crim | tokens_per_frame | 77.602 | 107.878 | 141.766 |

step-1 pass: {'N1': False, 'N2': False, 'M0': False}; carried arm: None

```
{
 "N1": {
  "1a_tail_seg_gain": 0.13465277777777782,
  "1b_pale_gain_calms21": 0.13157894736842107,
  "1b_pale_gain_crim13": 0.03645833333333333,
  "1c_body_delta_mabe22": 0.04533179012345667,
  "1c_body_delta_calms21": 0.20577485380116978,
  "1d_leak": {
   "mab": 0.020149926727539173,
   "cal": 0.16357763484540735,
   "crim": 0.14990947495473747
  }
 },
 "N2": {
  "1a_tail_seg_gain": 0.20350694444444442,
  "1b_pale_gain_calms21": 0.19736842105263153,
  "1b_pale_gain_crim13": 0.05729166666666667,
  "1c_body_delta_mabe22": 0.07357253086419746,
  "1c_body_delta_calms21": 0.23464912280701777,
  "1d_leak": {
   "mab": 0.027577514494580287,
   "cal": 0.2211063439432017,
   "crim": 0.17778022704728313
  }
 }
}
```