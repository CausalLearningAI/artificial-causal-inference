#!/usr/bin/env python3
"""Dump one JSON describing every scored run: its metrics AND its full specification.

The status report's model figure is a VIEW over this file. Each point carries the whole recipe
that produced it -- encoder, whether the encoder saw unlabelled frames, how it was adapted, the
head, the augmentation, the objective -- so hovering a point answers "what IS this run?" without
the reader having to decode a directory name.

Everything here is read from `results/vision/mice/frame/<tag>/config.json` (written by
train_online_aug.py when a run FINISHES) and `val_probs.npz` (its held-out predictions). Nothing
is transcribed: a run with no config.json has not landed and does not appear.

One field cannot be read from the config and is mapped here instead: `init_encoder` records the
SSL checkpoint a run started from, but the first SSL arm
(`res448_k2_frozen_d4photo_sslinit`) predates that field. `ablate_ssl_dapt.sh` names it
explicitly -- SSL_TAG=ssl_dapt, STAGE_B_TAG=res448_k2_frozen_d4photo_sslinit -- so that one
pairing is hard-coded, and flagged as such in the output.

    python scripts/mice_behavior/build_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from event_eval import evaluate                                            # noqa: E402

FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
OUT = FRAME / '_figures'
THS = np.round(np.arange(0.05, 1.0, 0.05), 2)

# Runs whose config predates the `init_encoder` field. Without these the model figure reports
# "no SSL" for arms the section-04 table calls SSL-adapted -- a figure contradicting a table.
#   frozen_d4photo_sslinit  named explicitly in ablate_ssl_dapt.sh (SSL_TAG / STAGE_B_TAG)
#   bit6_d4_sslinit         no launcher records it; the pairing is inferred from the `_sslinit`
#                           suffix and flagged as such in the output rather than asserted
SSL_BACKFILL = {'res448_k2_frozen_d4photo_sslinit': ('ssl_dapt', 'ablate_ssl_dapt.sh'),
                'res448_k2_bit6_d4_sslinit': ('ssl_dapt', 'run name')}

# how many unlabelled frames each SSL checkpoint was adapted on, from its own run log
SSL_CORPUS = {'ssl_dapt': '374k frames, v1+v2, 2 blocks',
              'ssl_s5_b2': '2x corpus, matched compute, 2 blocks',
              'ssl_s10_b6': '374k frames, 6 blocks'}

# Runs that are not candidate models. Mixing them in would compare a 5-pool budget against a
# 20-pool one, or an objective change against a head change, and the AP-vs-rDelta correlation
# reported to the reader would be measured over an incoherent set.
#
# `objective` is its own role for a reason: every vREx and DERM arm was launched from a script
# that does not set CROSS_ATTN_DIM/PATCH_POOL_DIM, so the DERM arms carry the plain 5.03 M head
# while the encoder/head candidates carry the 0.52 M cross-attention one. They are comparable to
# each other and to their own matched controls, not to the candidate pool.
ROLE = {
    'res448_k2_frozen_d4photo_lc5p': 'label budget', 'res448_k2_frozen_d4photo_lc10p': 'label budget',
    'res448_k2_frozen_d4photo_lc15p': 'label budget',
    'res448_k2_frozen_d4photo_px112': 'input ablation',
    'res448_k2_frozen_d4photo_px224': 'input ablation',
    'res448_k2_frozen_d4photo_vrexCond_b1': 'objective', 'res448_k2_frozen_d4photo_vrexCond_b10': 'objective',
    'res448_k2_frozen_d4photo_vrexAnn_b10': 'objective', 'res448_k2_frozen_d4photo_vrexAnn_b100': 'objective',
    'res448_k2_frozen_d4photo_dermPhase': 'objective', 'res448_k2_frozen_d4photo_dermPhase_s1': 'objective',
    'res448_k2_frozen_d4photo_dermCond': 'objective', 'res448_k2_frozen_d4photo_dermAnn': 'objective',
    'res448_k2_frozen_d4photo_ermH5M': 'objective', 'res448_k2_frozen_d4photo_ermH5M_s1': 'objective',
    'xfit_f1': 'deployment fold', 'xfit_f2': 'deployment fold', 'xfit_f3': 'deployment fold',
}


def spec(tag: str, c: dict) -> dict:
    """The recipe, as the reader needs to see it -- not as the config happens to store it."""
    blocks = c.get('unfreeze_blocks') or 0
    mode = c.get('ft_mode') or 'full'
    n_enc = c.get('n_encoder_trainable')
    if blocks == 0:
        ft = 'none (encoder frozen)'
    elif mode == 'bitfit':
        ft = f'BitFit, last {blocks} blocks'
    else:
        ft = f'full, last {blocks} blocks'
    if blocks and n_enc:
        ft += f' — {n_enc:,} params @ lr {c.get("encoder_lr")}'
    elif blocks:
        ft += f' @ lr {c.get("encoder_lr")}'

    ssl = c.get('init_encoder')
    back = SSL_BACKFILL.get(tag)
    ssl_tag = Path(ssl).parts[-2] if ssl else (back[0] if back else None)
    head_p = c.get('n_head_params')
    bits = []
    q = c.get('pool_queries') or 1
    bits.append(f'{q} learned quer{"y" if q == 1 else "ies"} over {c.get("n_patches")} patch tokens')
    if c.get('pool_grid'):
        bits.append(f'{c["pool_grid"]}x{c["pool_grid"]} region grid')
    if c.get('patch_selfattn_dim'):
        bits.append(f'patch self-attention (d={c["patch_selfattn_dim"]})')
    bits.append(f'temporal attention over {2 * (c.get("context_k") or 0) + 1} frames')

    # env_key is the flag's internal name; print what the environments actually ARE
    ENV = {'condition': 'the {n} phase × exposure cells', 'phase': 'the {n} phases',
           'annotator': 'annotator ({n} in the training pools)', 'none': 'none'}
    beta, env = c.get('vrex_beta'), c.get('env_key')
    # the count comes from the run, not from this table: the annotator arms see only the
    # annotators present in their training split, which is fewer than the six in the cohort
    env_nice = ENV.get(env, str(env)).replace('{n}', str(c.get('n_environments', '?')))
    if beta:
        obj = f'vREx, environments = {env_nice}, β = {beta:g}'
    elif c.get('derm'):
        obj = f'DERM, environments = {env_nice}'
    else:
        obj = 'ERM (weighted BCE)'
    # the head is what the DERM/vREx launcher silently changed; surface it next to the objective
    if not c.get('cross_attn_dim') and (c.get('n_head_params') or 0) > 1e6:
        obj += ' — plain head, comparable only to its own matched control'
    aug = {'d4_photo': 'D4 dihedral + brightness / contrast / gamma',
           'd4': 'D4 dihedral only', 'none': 'none'}.get(c.get('augment'), str(c.get('augment')))
    return {
        'encoder': 'DINOv2-base ViT-B/14 (87 M)',
        'input': f'{c.get("input_size")} px → {c.get("n_patches")} tokens/frame'
                 + (f' (pixel detail capped at {c["pixel_source"]} px)' if c.get('pixel_source') else ''),
        'ssl': (f'yes — {SSL_CORPUS.get(ssl_tag, ssl_tag)}'
                + (f' [pairing from {back[1]}, not the config]' if back else ''))
               if ssl_tag else 'no (stock DINOv2)',
        'finetuning': ft,
        'head': (f'{head_p:,} params — ' if head_p else '') + '; '.join(bits),
        'augmentation': aug,
        'objective': obj,
        'train_pools': c.get('n_train_pools') or 20,
        'epochs': c.get('n_epochs'),
        'seed': c.get('seed'),
    }


def main():
    rows = []
    for d in sorted(FRAME.iterdir()):
        if not (d / 'val_probs.npz').exists() or not (d / 'config.json').exists():
            continue
        c = json.load(open(d / 'config.json'))
        ap = c.get('ap_report', {}).get('macro/tol0', {}).get('ap')
        if ap is None:
            continue
        try:
            res = evaluate(d.name, 1, 1, THS)
        except Exception as e:                                  # a run whose npz is truncated
            print(f'  [skip] {d.name}: {e}'); continue
        rows.append({
            'tag': d.name,
            'role': ROLE.get(d.name, 'model candidate'),
            'ap': round(float(ap), 4),
            'f1_nt': round(float(res['nt']['best'].f1), 3),
            'f1_nn': round(float(res['nn']['best'].f1), 3),
            'rd_nt': round(float(res['nt']['r_delta']), 3),
            'rd_nn': round(float(res['nn']['r_delta']), 3),
            'spec': spec(d.name, c),
        })
        print(f'  {d.name:46s} AP {ap:.4f}  rd {res["nt"]["r_delta"]:.3f}/{res["nn"]["r_delta"]:.3f}')
    cand = [r for r in rows if r['role'] == 'model candidate']
    from scipy import stats
    rho = stats.spearmanr([r['ap'] for r in cand],
                          [(r['rd_nt'] + r['rd_nn']) / 2 for r in cand]).statistic
    payload = {'meta': {'n_runs': len(rows), 'n_candidates': len(cand),
                        'spearman_ap_vs_rdelta': round(float(rho), 3),
                        'split': 'standing 4-pool validation split (24 observations), '
                                 'except the xfit_* folds, which are out-of-fold on 8 pools each'},
               'runs': rows}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT / 'models.json', 'w'), indent=1)
    print(f'\nwrote {OUT / "models.json"}  ({len(rows)} runs, {len(cand)} candidates, '
          f'Spearman AP vs mean rΔ = {rho:+.3f})')


if __name__ == '__main__':
    main()
