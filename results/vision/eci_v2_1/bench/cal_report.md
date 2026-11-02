## CalMS21 (19 test videos; median [range] over seeds 2001-2003; P(on) primary)

| arm | mean AUROC P(on) | mean AUROC raw | mean AUROC inv+mount P(on) | mean nAP P(on) | mean top-1% prec P(on) | AUROC attack P(on) | AUROC investigation P(on) | AUROC mount P(on) | SAE eval FVE |
|---|---|---|---|---|---|---|---|---|---|
| S | 0.936 [0.933, 0.943] | 0.896 [0.890, 0.902] | 0.922 [0.916, 0.928] | 0.663 [0.648, 0.677] | 0.823 [0.811, 0.852] | 0.968 [0.965, 0.972] | 0.862 [0.848, 0.872] | 0.984 [0.982, 0.985] | 0.852 [0.852, 0.853] |
| V | 0.955 [0.947, 0.958] | 0.910 [0.899, 0.918] | 0.941 [0.929, 0.946] | 0.723 [0.644, 0.732] | 0.849 [0.815, 0.859] | 0.982 [0.981, 0.985] | 0.899 [0.885, 0.904] | 0.983 [0.974, 0.987] | 0.816 [0.816, 0.816] |
| W | 0.936 [0.934, 0.958] | 0.886 [0.878, 0.920] | 0.913 [0.909, 0.947] | 0.685 [0.652, 0.723] | 0.843 [0.821, 0.846] | 0.980 [0.978, 0.985] | 0.841 [0.833, 0.907] | 0.986 [0.984, 0.987] | 0.798 [0.797, 0.798] |

| contrast | readout | labels | metric | point | 95% CI (seeds fixed) | 95% CI (seeds resampled) | verdict |
|---|---|---|---|---|---|---|---|
| V - S | pon | all 3 | mean AUROC | +0.0157 | [+0.0113, +0.0230] | [+0.0067, +0.0257] | temporal SUPPORTED (V) |
| V - S | pon | all 3 | mean nAP | +0.0375 | [+0.0099, +0.0704] | [-0.0230, +0.0892] |  |
| V - S | pon | all 3 | mean top1_prec | +0.0125 | [-0.0264, +0.0450] | [-0.0386, +0.0569] |  |
| V - S | pon | inv+mount | mean AUROC | +0.0166 | [+0.0105, +0.0213] | [+0.0040, +0.0282] |  |
| V - S | pon | inv+mount | mean nAP | +0.0152 | [-0.0106, +0.0398] | [-0.0659, +0.0818] |  |
| V - S | pon | inv+mount | mean top1_prec | -0.0282 | [-0.0724, +0.0090] | [-0.0943, +0.0252] |  |
| V - S | raw | all 3 | mean AUROC | +0.0132 | [+0.0088, +0.0172] | [+0.0014, +0.0238] |  |
| V - S | raw | all 3 | mean nAP | +0.0317 | [+0.0171, +0.0424] | [-0.0219, +0.0716] |  |
| V - S | raw | all 3 | mean top1_prec | +0.0146 | [-0.0197, +0.0342] | [-0.0366, +0.0487] |  |
| V - S | raw | inv+mount | mean AUROC | +0.0126 | [+0.0060, +0.0178] | [-0.0053, +0.0299] |  |
| V - S | raw | inv+mount | mean nAP | -0.0026 | [-0.0211, +0.0125] | [-0.0800, +0.0582] |  |
| V - S | raw | inv+mount | mean top1_prec | -0.0218 | [-0.0558, +0.0090] | [-0.0823, +0.0318] |  |
| W - S | pon | all 3 | mean AUROC | +0.0049 | [-0.0012, +0.0120] | [-0.0090, +0.0204] | temporal NOT supported (W) |
| W - S | pon | all 3 | mean nAP | +0.0245 | [+0.0015, +0.0508] | [-0.0177, +0.0698] |  |
| W - S | pon | all 3 | mean top1_prec | +0.0082 | [-0.0234, +0.0400] | [-0.0330, +0.0454] |  |
| W - S | pon | inv+mount | mean AUROC | +0.0010 | [-0.0061, +0.0067] | [-0.0196, +0.0240] |  |
| W - S | pon | inv+mount | mean nAP | +0.0096 | [-0.0140, +0.0440] | [-0.0491, +0.0763] |  |
| W - S | pon | inv+mount | mean top1_prec | -0.0177 | [-0.0556, +0.0266] | [-0.0648, +0.0350] |  |
| W - S | raw | all 3 | mean AUROC | -0.0014 | [-0.0077, +0.0040] | [-0.0220, +0.0231] |  |
| W - S | raw | all 3 | mean nAP | +0.0360 | [+0.0141, +0.0572] | [-0.0022, +0.0664] |  |
| W - S | raw | all 3 | mean top1_prec | +0.0168 | [-0.0243, +0.0447] | [-0.0297, +0.0503] |  |
| W - S | raw | inv+mount | mean AUROC | -0.0102 | [-0.0192, -0.0036] | [-0.0413, +0.0247] |  |
| W - S | raw | inv+mount | mean nAP | +0.0172 | [-0.0134, +0.0523] | [-0.0390, +0.0709] |  |
| W - S | raw | inv+mount | mean top1_prec | -0.0175 | [-0.0619, +0.0306] | [-0.0716, +0.0404] |  |
| W - V | pon | all 3 | mean AUROC | -0.0108 | [-0.0155, -0.0061] | [-0.0246, +0.0037] |  |
| W - V | pon | all 3 | mean nAP | -0.0129 | [-0.0348, +0.0114] | [-0.0748, +0.0489] |  |
| W - V | pon | all 3 | mean top1_prec | -0.0043 | [-0.0362, +0.0330] | [-0.0469, +0.0456] |  |
| W - V | pon | inv+mount | mean AUROC | -0.0155 | [-0.0229, -0.0075] | [-0.0373, +0.0079] |  |
| W - V | pon | inv+mount | mean nAP | -0.0057 | [-0.0367, +0.0299] | [-0.0920, +0.0925] |  |
| W - V | pon | inv+mount | mean top1_prec | +0.0105 | [-0.0384, +0.0658] | [-0.0529, +0.0835] |  |
| W - V | raw | all 3 | mean AUROC | -0.0146 | [-0.0199, -0.0090] | [-0.0363, +0.0089] |  |
| W - V | raw | all 3 | mean nAP | +0.0043 | [-0.0161, +0.0290] | [-0.0485, +0.0623] |  |
| W - V | raw | all 3 | mean top1_prec | +0.0022 | [-0.0254, +0.0323] | [-0.0410, +0.0513] |  |
| W - V | raw | inv+mount | mean AUROC | -0.0228 | [-0.0307, -0.0147] | [-0.0565, +0.0140] |  |
| W - V | raw | inv+mount | mean nAP | +0.0199 | [-0.0124, +0.0547] | [-0.0617, +0.1067] |  |
| W - V | raw | inv+mount | mean top1_prec | +0.0043 | [-0.0367, +0.0474] | [-0.0559, +0.0757] |  |

Attack is positive in only 5 of the 19 test videos (mouse075-079); the inv+mount rows exclude it.
Bootstrap: 2000 replicates over test videos (rng 20261102), neuron picks fixed at the full-data picks.