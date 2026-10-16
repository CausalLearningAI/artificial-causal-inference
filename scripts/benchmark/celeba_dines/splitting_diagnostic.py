"""
Feature-splitting diagnostic for the CelebA TopK SAE (9,216 atoms, 19,962 test images, 40 attributes).
Per attribute: best single atom (F1 of Z>0, AUROC of Z), best OR-pair among top-20 atoms by F1, forward-selected
logistic group (k=1,2,3,5, held-out AUROC on a stratified 70/30 split, seed 0), Jaccard duplicates (>0.5) of the best atom.
Atom ids are 1-indexed (same as best_f1_atoms.csv). Inputs are staged to $LOCAL_DIR first.
Usage: splitting_diagnostic.py OUT_DIR
"""
import os, sys, shutil, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

SRC = Path('/mnt/beegfs/locatgrp/shared/rcadei/dines_celeba')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LOCAL = Path(os.environ.get('LOCAL_DIR', '/tmp/splitdiag')); LOCAL.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for f in ['codes_f32.npy', 'labels.npz']:
    shutil.copy(SRC / f, LOCAL / f)
Z = np.load(LOCAL / 'codes_f32.npy')
lab = np.load(LOCAL / 'labels.npz', allow_pickle=True)
Y = lab['labels'].astype(np.int8); names = [str(n) for n in lab['names']]
N, D = Z.shape
print('loaded', Z.shape, Y.shape, f'{time.time()-t0:.0f}s', flush=True)
B = (Z > 0)
Bf = B.astype(np.float32)
nfire = Bf.sum(0)
print('mean active per image', Bf.sum(1).mean(), flush=True)
TP = Bf.T @ Y.astype(np.float32)            # D x 40

def auroc_cols(X, y):  # AUROC of every column of X (N x m) vs binary y, via ranks (ties averaged)
    r = np.apply_along_axis(rankdata, 0, X)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)

def f1(tp, nf, npos): return 2 * tp / np.maximum(nf + npos, 1e-9)

idx = np.arange(N)
rows = []
for a, name in enumerate(names):
    y = Y[:, a].astype(int); npos = y.sum()
    F1 = f1(TP[:, a], nfire, npos)
    order = np.argsort(-F1); j = int(order[0])
    top = order[:20]
    # AUROC of best-F1 atom, and best AUROC among the top 20
    auc_top = auroc_cols(Z[:, top], y)
    auc_best = float(auc_top[0])
    # OR pairs
    best = (-1, None, None)
    Bt = B[:, top]
    for p, q in itertools.combinations(range(20), 2):
        u = Bt[:, p] | Bt[:, q]
        tp = (u & (y == 1)).sum(); s = 2 * tp / (u.sum() + npos)
        if s > best[0]: best = (s, p, q)
    s2, p, q = best
    both = (Bt[:, p] & Bt[:, q]).sum(); either = (Bt[:, p] | Bt[:, q]).sum()
    # duplicates of best atom
    inter = Bf.T @ Bf[:, j]; jac = inter / (nfire + nfire[j] - inter)
    jac[j] = 0
    dups = [(int(k) + 1, round(float(jac[k]), 3)) for k in np.where(jac > 0.5)[0]]
    # group: top-20 chosen on train split only; forward selection by train AUROC; held-out AUROC
    tr, te = train_test_split(idx, test_size=0.3, random_state=0, stratify=y)
    Ftr = f1(Bf[tr].T @ y[tr].astype(np.float32), nfire_tr := Bf[tr].sum(0), y[tr].sum())
    cand = np.argsort(-Ftr)[:20]
    X = np.log1p(Z[:, cand]); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; X = (X - mu) / sd
    sel = []; held = {}
    for k in range(1, 6):
        bs, bc = -1, None
        for c in range(20):
            if c in sel: continue
            m = LogisticRegression(max_iter=300).fit(X[tr][:, sel + [c]], y[tr])
            s = roc_auc_score(y[tr], m.decision_function(X[tr][:, sel + [c]]))
            if s > bs: bs, bc = s, c
        sel.append(bc)
        m = LogisticRegression(max_iter=300).fit(X[tr][:, sel], y[tr])
        held[k] = roc_auc_score(y[te], m.decision_function(X[te][:, sel]))
    k95 = next(k for k in range(1, 6) if held[k] >= 0.95 * held[5])
    rows.append(dict(attribute=name, prevalence=npos / N, best_atom=j + 1, single_f1=F1[j], single_auroc=auc_best,
        max_auroc_top20=float(auc_top.max()),
        pair_atoms=f'{top[p]+1}+{top[q]+1}', pair_f1=s2, pair_f1_gain=s2 - F1[j],
        pair_cofire_jaccard=both / either, pair_cofire_frac_of_rarer=both / max(1, min(Bt[:, p].sum(), Bt[:, q].sum())),
        group_atoms_k5=' '.join(str(cand[c] + 1) for c in sel),
        auroc_k1=held[1], auroc_k2=held[2], auroc_k3=held[3], auroc_k5=held[5],
        auroc_gain_k5_vs_k1=held[5] - held[1], k_for_95pct_of_k5=k95,
        duplicates_jaccard_gt_0p5=';'.join(f'{k}:{v}' for k, v in dups)))
    print(name, f'{time.time()-t0:.0f}s', flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'splitting_diagnostic.csv', index=False)
ref = pd.read_csv('/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/eci/dines_celeba/best_f1_atoms.csv').set_index('attribute')
chk = df.set_index('attribute')
same = (ref.loc[chk.index, 'best_atom'] == chk['best_atom']).sum()
maxdiff = (ref.loc[chk.index, 'best_f1'] - chk['single_f1']).abs().max()

flag = (df.pair_f1_gain > 0.05) | (df.auroc_gain_k5_vs_k1 > 0.02)
def md(d):
    c = ['attribute', 'prevalence', 'best_atom', 'single_f1', 'single_auroc', 'pair_atoms', 'pair_f1', 'pair_f1_gain',
         'pair_cofire_jaccard', 'auroc_k1', 'auroc_k2', 'auroc_k3', 'auroc_k5', 'k_for_95pct_of_k5', 'duplicates_jaccard_gt_0p5']
    return d[c].round(3).to_markdown(index=False)
bench = df[df.attribute.isin(['Wearing_Hat', 'Eyeglasses', 'Blond_Hair', 'Smiling'])]
lines = ['# SAE feature-splitting diagnostic (CelebA test, 9,216 TopK atoms)', '',
 f'Reproduction check vs best_f1_atoms.csv: best atom identical for {same}/40 attributes; max |F1 diff| = {maxdiff:.2e}.', '',
 f'Attributes where best OR-pair F1 gain > 0.05: {(df.pair_f1_gain>0.05).sum()}/40',
 f'Attributes where group k=5 held-out AUROC gain over k=1 > 0.02: {(df.auroc_gain_k5_vs_k1>0.02).sum()}/40',
 f'Either: {flag.sum()}/40', '',
 f'Median pair F1 gain {df.pair_f1_gain.median():.3f} (mean {df.pair_f1_gain.mean():.3f}, max {df.pair_f1_gain.max():.3f})',
 f'Median AUROC gain k5-k1 {df.auroc_gain_k5_vs_k1.median():.3f}; k2-k1 {(df.auroc_k2-df.auroc_k1).median():.3f}; k3-k1 {(df.auroc_k3-df.auroc_k1).median():.3f}',
 f'Median k reaching 95% of k=5 AUROC: {df.k_for_95pct_of_k5.median()}; counts by k: {df.k_for_95pct_of_k5.value_counts().sort_index().to_dict()}',
 f'Attributes with a duplicate atom (Jaccard>0.5 with best atom): {(df.duplicates_jaccard_gt_0p5.fillna("")!="").sum()}/40', '',
 '## Benchmark attributes', md(bench), '', '## All attributes', md(df), '',
 'Notes: atom ids 1-indexed. Single F1/AUROC on all 19,962 images; AUROC k=1..5 is held-out (30%, stratified, seed 0) from forward-selected logistic regression (log1p codes, top-20 candidates chosen by F1 on the train split). Pair = best F1 of (Z_a>0 OR Z_b>0) over pairs of the top-20 F1 atoms on all images (in-sample, 190 pairs). Cofire Jaccard = both/either.']
(OUT / 'summary.md').write_text('\n'.join(lines))
print('\n'.join(lines[:12]))
shutil.rmtree(LOCAL, ignore_errors=True)
print('done', f'{time.time()-t0:.0f}s')
