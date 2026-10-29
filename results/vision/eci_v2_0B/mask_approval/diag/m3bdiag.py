import sys, json, numpy as np, torch
sys.path.insert(0, '/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2')
sys.path.insert(0, '/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference')
import mask3_val as MV
from src.eci.v2 import mask2 as M2, mask3 as M3
torch.set_num_threads(16)
o, n = MV.series_pair('mice', 'v1', torch.device('cpu'), rule='m3')
r = MV.masks_for(o, n, o.V['wt_ash1l_m_1_S_H'], [6001])
_, co, cn = r[0]
x = co['x']; P = n.pack('wt_ash1l_m_1_S_H', 0)
xf = M2.gauss_blur(x.float(), M2.SIGMA)
df = (xf - M2.gauss_blur(P['bg'][None].float(), M2.SIGMA)).abs().amax(1)[0]
dm = (xf - M2.gauss_blur(P['bg_raw'][None].float(), M2.SIGMA)).abs().amax(1)[0]
g = M2.grey_f(x)[0]
sl = (slice(775, 815), slice(650, 820))
tail = (g[sl] < 175)  # tail pixels darker than bedding inside the box (rough)
out = {'tail_px': int(tail.sum())}
for nm, d in (('d_fixed', df), ('d_median', dm), ('d_min', torch.minimum(df, dm))):
    v = d[sl][tail]
    out[nm] = {'q25_50_75': np.quantile(v.numpy(), [.25, .5, .75]).tolist(), 'gt35': float((v > 35).float().mean())}
gm = M2.grey_f(P['bg_raw'][None])[0][sl][tail]; gq = P['q98'][sl][tail]
out['grey_frame_q50'] = float(g[sl][tail].median()); out['grey_median_bg_q50'] = float(gm.median()); out['q98_q50'] = float(gq.float().median())
out['in_region'] = float(torch.from_numpy(M3.fix_pack(P['bg_raw'].numpy(), P['q98'].numpy(), P['mad'].numpy(), 32.0)['region'])[sl][tail].float().mean())
print(json.dumps(out, indent=1))
