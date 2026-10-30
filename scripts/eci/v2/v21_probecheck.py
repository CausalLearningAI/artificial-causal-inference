import json, sys, numpy as np
from pathlib import Path
R=Path('/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference'); sys.path.insert(0,str(R)); sys.path.insert(0,str(R/'scripts/eci/v2'))
import mask5_val as M5
res=json.load(open(R/'results/vision/eci_v2_0B/mask_approval_v3/results_final.json'))
split=json.load(open(R/'results/vision/eci_v2_0B/mask_approval_v3/split.json'))
for dom in ['ants','mice','frogs']:
    P={k:np.load(f'/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full/{dom}/v2.1_m5/_probe/probe.npz')[k] for k in ('mu','sd','w','b')}; P['b']=float(P['b'])
    cfg=res[dom]['cfg']; rows=[]
    for key in split[dom]['test']:
        d_,ser,rest=key.split('_',2); obs,k=rest.rsplit('_',1)
        z=np.load(f'/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5/{dom}/win_{ser}_{obs}_{k}.npz'); it=int(z['it']); G=int(z['G'])
        keep=M5.postproc(M5.probe_p(P,z['z'][it]),G,cfg['t'],cfg['t_core'],None,cfg['smooth'])
        g=np.load(R/f'results/vision/eci_v2_0B/mask_approval_v2/gold/{key}.npz'); S=G*16
        M5._GC.clear(); rows.append(M5.score(keep,np.unpackbits(g['pix'],-1)[:,:S].astype(bool),g['thin'],G,float(z['h'])))
    M5._GC.clear()
    m=M5.summ(rows); a=res[dom]['test']['new']
    print(dom, {k:(round(m[k],4),round(a[k],4)) for k in ['token_f1','thin_recall','bg_tok_per_frame','tok_per_frame']})
