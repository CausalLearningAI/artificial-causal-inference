"""
MABe22 ECI round 3: encoder throughput benchmark (GPU job, ~5 min; cost-only input to the E3 orbit and E4 precision
decisions pre-registered in eci_round3.sh). DINOv2-base on random 224 frames upscaled to S (336 / 448 / 672), fp32 TF32
off (R's precision) and bf16 autocast (448 only, with the token deviation vs fp32). Output: OUT/r3/bench_<gpu>.json
"""
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import OUT  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device('cuda')
m = AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval()
g = torch.Generator().manual_seed(0)
x0 = torch.randint(0, 256, (64, 3, 224, 224), generator=g).float().to(dev)
res = {'gpu': torch.cuda.get_device_name(0)}
ref = None
for S, bs, prec in ((448, 32, 'fp32'), (336, 32, 'fp32'), (672, 16, 'fp32'), (448, 32, 'bf16'), (448, 32, 'tf32')):
    torch.backends.cuda.matmul.allow_tf32 = prec == 'tf32'
    x = F.interpolate(x0, size=(S, S), mode='bicubic', align_corners=False).round().clamp(0, 255) / 255
    with torch.inference_mode():
        for rep in range(2):
            torch.cuda.synchronize()
            t0 = time.time()
            outs = []
            for a in range(0, 64, bs):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=prec == 'bf16'):
                    h = m(pixel_values=x[a:a + bs]).last_hidden_state[:, 1:].float()
                outs.append(h)
            torch.cuda.synchronize()
            dt = time.time() - t0
        h = torch.cat(outs)
    r = {'img_per_s': 64 / dt}
    if S == 448 and prec == 'fp32':
        ref = h
    elif S == 448:
        r['cos_vs_fp32_mean'] = float(F.cosine_similarity(h, ref, dim=-1).mean())
        r['cos_vs_fp32_min'] = float(F.cosine_similarity(h, ref, dim=-1).min())
        r['rel_err'] = float((h - ref).norm() / ref.norm())
    res[f'{S}_{prec}'] = r
    print(S, prec, r, flush=True)
(OUT / 'r3').mkdir(parents=True, exist_ok=True)
(OUT / 'r3' / f"bench_{res['gpu'].replace(' ', '_')}.json").write_text(json.dumps(res, indent=1))
