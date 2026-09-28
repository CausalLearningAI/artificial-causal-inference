"""
Render NES (Neural Effect Search) hypothesis galleries for the mice v1 ECI pipeline.

Reads results/vision/mice/eci/nes/<sae>/{selected_neurons.json, summary.csv} and renders, under
results/vision/mice/eci/nes/<sae>/galleries/:

  neurons/            the standard neuron_gallery.py figure (src/eci/viz.py gallery_for) for all
                       neurons in selected_neurons.json, with a stats table joined in (every
                       analysis where the neuron was selected, round, tau, p, robustness flags
                       from SUMMARY.md, artefact_flag).
  videos/<analysis>/   within-video views (src/eci/viz.py video_view):
                       family A (paired stage transition): for the round-1 neuron, the 3 pools
                       with the largest |paired difference| in the direction of the effect, both
                       videos of the pair (6 views per analysis, 8 analyses);
                       family B (genotype within stage) hits: 2 het + 2 wt videos with the most
                       extreme per-video mean for that stage (4 views per hit, 7 hits).
  stats_wide.csv        one row per neuron (fed to gallery_for as stats_table).
  stats_detail.csv       one row per (neuron, analysis, prefix) hit: round, tau, p, direction,
                       robustness flags (signflip, max-pool, rate, BH, matched, trim30, other
                       prefix), artefact_flag -- the full machine-readable stats table.
  index.html            self-contained index organized by the 14 analyses (family A first).

Does not edit src/eci/*.py or scripts/eci/run_nes.py: only imports and calls them.

Usage:
    python scripts/eci/render_nes_galleries.py --sae matryoshka_btk_1024_k16_ep20_s0
"""
import argparse
import html
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.viz import (STAGE_LABEL, PatchEncoder, default_sae_path, gallery_for,  # noqa: E402
                         load_full_codes, video_view)
from src.eci import contrasts as C  # noqa: E402

sys.path.insert(0, str(REPO / 'scripts' / 'eci'))
import run_nes as RN  # noqa: E402  (only used read-only: PREFIXES, selected_set, ARTEFACTS)

GENO_LABEL = {'het': 'het', 'wt': 'wt'}
SENS = {'signflip': dict(test='signflip'), 'max-pool': dict(pooling='max'), 'rate': dict(outcome_type='rate'),
        'BH': dict(correction='bh'), 'matched': dict(window='matched'), 'trim30': dict(window='trim30')}
PRIM = dict(pooling='mean', outcome_type='mean', test='t', correction='bonferroni', window='full')
ROB_ORDER = list(SENS) + ['other_prefix']


def robustness(tidy, aid, prefix, neuron, family):
    """Y/N/- per sensitivity setting + other-prefix, exactly the logic of run_nes.write_reports
    (reproduced here read-only via RN.selected_set, since that logic is a local closure there)."""
    flags = {}
    for name, ov in SENS.items():
        if (name == 'signflip' and family != 'A') or (name == 'matched' and not aid.endswith(('1to2', '4to5'))):
            flags[name] = '-'
            continue
        flags[name] = 'Y' if neuron in RN.selected_set(tidy, aid, prefix=prefix, **{**PRIM, **ov}) else 'N'
    if prefix == 1024 and neuron >= 128:
        flags['other_prefix'] = '-'
    else:
        other_prefix = [p for p in RN.PREFIXES if p != prefix][0]
        other_sel = RN.selected_set(tidy, aid, prefix=other_prefix, **PRIM)
        flags['other_prefix'] = 'Y' if neuron in other_sel else 'N'
    return flags


def primary_round1(tidy, aid):
    """Round-1 neuron of an analysis under the primary setting (prefix 128; identical tau/p at
    prefix 1024 when the neuron id is shared)."""
    sub = tidy[(tidy.analysis_id == aid) & (tidy['round'] == 1)]
    for k, v in PRIM.items():
        sub = sub[sub[k] == v]
    sub = sub[sub.prefix == 128] if (sub.prefix == 128).any() else sub[sub.prefix == 1024]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return dict(neuron=int(r.neuron), direction=r.direction, tau=float(r.tau), p=float(r.p))


def hit_neurons_of(tidy, aid):
    """All distinct neurons selected (any round, either prefix) in an analysis under primary."""
    sub = tidy[(tidy.analysis_id == aid) & (tidy['round'] > 0)]
    for k, v in PRIM.items():
        sub = sub[sub[k] == v]
    out = []
    for j in sorted(sub.neuron.dropna().astype(int).unique().tolist()):
        r = sub[sub.neuron == j].iloc[0]
        out.append(dict(neuron=j, round=int(r['round']), direction=r.direction, tau=float(r.tau), p=float(r.p),
                        prefix=int(r.prefix)))
    return out


# --------------------------------------------------------------------- stats tables
def build_stats_tables(selected, tidy):
    """stats_detail: one row per (neuron, analysis, prefix) hit, with robustness flags.
    stats_wide: one row per neuron (for gallery_for's stats_table)."""
    detail_rows, wide_rows = [], []
    for j_str, info in selected['neurons'].items():
        j = int(j_str)
        taus, ps, aids, hit_strs = [], [], set(), []
        for h in info['hits']:
            aid, prefix, rnd = h['analysis_id'], h['prefix'], h['round']
            family = aid[0]
            flags = robustness(tidy, aid, prefix, j, family)
            detail_rows.append({'neuron': j, 'analysis_id': aid, 'family': family, 'prefix': prefix, 'round': rnd,
                                'direction': h['direction'], 'tau': h['tau'], 'p': h['p'],
                                **{f'rob_{k}': v for k, v in flags.items()}, 'artefact_flag': info['artefact_flag']})
            taus.append(h['tau']), ps.append(h['p']), aids.add(aid)
            rob = ','.join(f'{k}={v}' for k, v in flags.items())
            hit_strs.append(f"{aid}(p{prefix} r{rnd} {h['direction']} tau={h['tau']:.3g} p={h['p']:.1e})[{rob}]")
        imin = int(np.argmin(ps))
        wide_rows.append({'neuron': j, 'n_analyses': len(aids), 'n_hits': len(info['hits']),
                          'analyses': ';'.join(sorted(aids)), 'families': ','.join(sorted({a[0] for a in aids})),
                          'min_p': ps[imin], 'primary_analysis': sorted(info['hits'], key=lambda h: h['p'])[0]['analysis_id'],
                          'primary_tau': taus[imin], 'artefact_flag': info['artefact_flag'],
                          'hits': ' | '.join(hit_strs)})
    detail = pd.DataFrame(detail_rows)
    wide = pd.DataFrame(wide_rows).set_index('neuron').sort_index()
    return wide, detail


# --------------------------------------------------------------------- within-video view selection
def family_a_views(design, summ, tidy, sae, out_root, pe, device, source):
    """For each of the 8 family-A analyses and its round-1 neuron: 3 pools with the largest
    |paired difference| in the direction of the effect, both videos of the pair."""
    rows = []
    for geno in ('het', 'wt'):
        for tr, (a, b) in C.TRANSITIONS.items():
            aid = f'A_{geno}_{tr}'
            hit = primary_round1(tidy, aid)
            if hit is None:
                print(f'[WARN] {aid}: no round-1 hit found', flush=True)
                continue
            j = hit['neuron']
            pools, Za, Zb = C.paired(summ['mean'], design, geno, a, b, stat='mean', window_a='full', window_b='full')
            diff = Zb[:, j] - Za[:, j]
            sign = 1.0 if hit['direction'] == 'up' else -1.0
            order = np.argsort(-(diff * sign))[:3]
            d = design[design.genotype == geno]
            ia = d[d.stage == a].set_index('pool')
            ib = d[d.stage == b].set_index('pool')
            vdir = out_root / 'videos' / aid
            vdir.mkdir(parents=True, exist_ok=True)
            for rank, idx in enumerate(order, start=1):
                p = pools[idx]
                obs_a, obs_b = ia.loc[p, 'observation_id'], ib.loc[p, 'observation_id']
                extra = {'analysis': aid, 'pool_rank': rank, 'pool': p, 'direction': hit['direction'],
                        'round1_tau': hit['tau'], 'round1_p': hit['p'], 'pair_diff_b_minus_a': float(diff[idx])}
                for stage_role, obs, other in ((f'stage{a}', obs_a, obs_b), (f'stage{b}', obs_b, obs_a)):
                    out_path = vdir / f'video_n{j:04d}_{obs}.png'
                    _, picks = video_view(j, obs, source=source, out_path=out_path, sae_name=sae, key='mean',
                                          n_top=6, min_gap_s=2.0, device=device, patch_encoder=pe,
                                          extra={**extra, 'this_stage': stage_role, 'paired_with': other})
                    rows.append({'analysis_id': aid, 'family': 'A', 'neuron': j, 'pool': p, 'pool_rank': rank,
                                'observation_id': obs, 'stage_role': stage_role, 'paired_with': other,
                                'pair_diff_b_minus_a': float(diff[idx]),
                                'png': str(out_path.relative_to(out_root))})
            print(f'  {aid}: neuron {j} ({hit["direction"]}) -> pools {[pools[i] for i in order]}', flush=True)
    return pd.DataFrame(rows)


def family_b_views(design, summ, tidy, sae, out_root, pe, device, source):
    """For each family-B hit (neuron selected in a B_stage analysis): 2 het + 2 wt videos with
    the most extreme (highest) per-video mean at that stage."""
    rows = []
    for stage in range(1, 7):
        aid = f'B_stage{stage}'
        hits = hit_neurons_of(tidy, aid)
        if not hits:
            continue
        d_stage = design[design.stage == stage]
        for h in hits:
            j = h['neuron']
            pools, Z, T = C.genotype_contrast(summ['mean'], design, stage, stat='mean', window='full')
            vals = Z[:, j]
            d_idx = d_stage.set_index('pool')
            obs_ids = d_idx.loc[pools, 'observation_id'].values
            vdir = out_root / 'videos' / aid
            vdir.mkdir(parents=True, exist_ok=True)
            for geno, mask in (('het', T == 1), ('wt', T == 0)):
                idx = np.argsort(-vals[mask])[:2]
                g_obs, g_vals = obs_ids[mask][idx], vals[mask][idx]
                for rank, (obs, v) in enumerate(zip(g_obs, g_vals), start=1):
                    out_path = vdir / f'video_n{j:04d}_{obs}.png'
                    extra = {'analysis': aid, 'genotype': geno, 'extreme_rank': rank, 'direction': h['direction'],
                            'hit_round': h['round'], 'hit_tau': h['tau'], 'hit_p': h['p'], 'video_mean_shown': float(v)}
                    _, picks = video_view(j, obs, source=source, out_path=out_path, sae_name=sae, key='mean',
                                          n_top=6, min_gap_s=2.0, device=device, patch_encoder=pe, extra=extra)
                    rows.append({'analysis_id': aid, 'family': 'B', 'neuron': j, 'genotype': geno,
                                'extreme_rank': rank, 'observation_id': obs, 'video_mean': float(v),
                                'png': str(out_path.relative_to(out_root))})
            print(f'  {aid}: neuron {j} ({h["direction"]})', flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- index.html
def analysis_header(aid):
    if aid.startswith('A_'):
        _, geno, tr = aid.split('_')
        a, b = C.TRANSITIONS[tr]
        return (f'{aid} &mdash; family A (paired within pool), {geno}: '
                f'stage {a} ({STAGE_LABEL[a]}) &rarr; stage {b} ({STAGE_LABEL[b]})')
    stage = int(aid.replace('B_stage', ''))
    return f'{aid} &mdash; family B (het vs wt), stage {stage} ({STAGE_LABEL[stage]})'


def write_index(out_root, tidy, wide, video_a, video_b, sae):
    def esc(s):
        return html.escape(str(s))

    def thumb_cell(j):
        return (f'<a class="nthumb" href="neurons/neuron_{j:04d}.png">'
                f'<img loading="lazy" src="neurons/thumbs/neuron_{j:04d}.jpg" alt="neuron {j}">'
                f'<div>neuron {j}</div></a>')

    def rob_str(tidy, aid, prefix, j, family):
        f = robustness(tidy, aid, prefix, j, family)
        return ' '.join(f'{k}:{v}' for k, v in f.items())

    sections = []
    aids_order = ([f'A_{g}_{t}' for g in ('het', 'wt') for t in C.TRANSITIONS] + [f'B_stage{s}' for s in range(1, 7)])
    for aid in aids_order:
        family = aid[0]
        hits = hit_neurons_of(tidy, aid)
        if not hits:
            sections.append(f'<section><h2>{analysis_header(aid)}</h2><p class="note">nothing selected.</p></section>')
            continue
        neuron_rows = []
        for h in hits:
            j = h['neuron']
            rob = rob_str(tidy, aid, h['prefix'], j, family)
            art = wide.loc[j, 'artefact_flag'] if j in wide.index else ''
            art_html = f'<span class="art">artefact: {esc(art)}</span>' if art else ''
            neuron_rows.append(
                f'<tr><td>{thumb_cell(j)}</td><td>{h["round"]}</td><td>{esc(h["direction"])}</td>'
                f'<td>{h["tau"]:.3g}</td><td>{h["p"]:.2e}</td><td class="rob">{esc(rob)}</td><td>{art_html}</td></tr>')
        neuron_table = ('<table class="neurons"><tr><th>neuron</th><th>round</th><th>dir</th><th>tau</th>'
                        '<th>p</th><th>robustness (signflip max-pool rate BH matched trim30 other_prefix)</th>'
                        '<th></th></tr>' + ''.join(neuron_rows) + '</table>')

        if family == 'A':
            sub = video_a[video_a.analysis_id == aid]
            links = []
            for pool, g in sub.groupby('pool'):
                g = g.sort_values('stage_role')
                cells = ' vs '.join(f'<a href="{esc(r.png)}">{esc(r.stage_role)}</a>' for r in g.itertuples())
                diff = g['pair_diff_b_minus_a'].iloc[0]
                links.append(f'<li>pool {esc(pool)} (rank {int(g["pool_rank"].iloc[0])}, '
                             f'&Delta;={diff:.3g}): {cells}</li>')
            video_html = f'<p>Within-video views (3 pools, largest paired difference):</p><ul>{"".join(links)}</ul>'
        else:
            sub = video_b[video_b.analysis_id == aid]
            links = []
            for j, gj in sub.groupby('neuron'):
                parts = []
                for geno, gg in gj.groupby('genotype'):
                    gg = gg.sort_values('extreme_rank')
                    parts.append(geno + ': ' + ', '.join(
                        f'<a href="{esc(r.png)}">#{int(r.extreme_rank)} ({r.video_mean:.3g})</a>' for r in gg.itertuples()))
                links.append(f'<li>neuron {j}: {" | ".join(parts)}</li>')
            video_html = f'<p>Within-video views (2 het + 2 wt, most extreme per-video mean):</p><ul>{"".join(links)}</ul>' if links else ''

        sections.append(f'<section><h2>{analysis_header(aid)}</h2>{neuron_table}{video_html}</section>')

    note = (
        'Neural Effect Search (NES) on SAE ' + esc(sae) + ', mice v1, primary setting: codes_mean pooling, '
        'outcome = per-video mean activation, t-test, Bonferroni alpha 0.05, full stage window. '
        'Family A = paired within pool (36 pairs: one het/wt group of pools measured at both stages of a '
        'transition). Family B = het vs wt within a stage (36 het vs 36 wt videos, unpaired). '
        'Robustness flags (Y = also selected under that single change from primary, same prefix; N = not; '
        '"-" = not applicable): signflip (sign-flip test, family A only), max-pool (codes_max pooling), '
        'rate (firing-rate outcome instead of mean), BH (Benjamini-Hochberg instead of Bonferroni), matched '
        '(time-matched window, habituation-to-odor transitions only), trim30 (first 30 s of every video '
        'dropped), other_prefix (also selected at the other Matryoshka prefix, 128 vs 1024). Family A listed '
        'first. See neurons/index.html for the full sortable neuron gallery and stats_detail.csv for the '
        'complete machine-readable stats table.')

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>NES galleries: {esc(sae)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fafaf8; --fg:#1d1d1f; --card:#fff; --line:#ddd; --muted:#666; --accent:#2a78b5; color-scheme: light dark; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --fg:#e6e6e6; --card:#212226; --line:#383a40; --muted:#9a9a9a; --accent:#6cb2eb; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; margin:0; padding:16px 24px 60px; }}
h1 {{ font-size:20px; margin:0 0 6px; }} h2 {{ font-size:16px; margin:0 0 8px; }}
.note {{ color:var(--muted); max-width:80em; margin:0 0 18px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin:0 0 14px; }}
table.neurons {{ border-collapse:collapse; width:100%; margin:6px 0 10px; font-size:12px; }}
table.neurons th, table.neurons td {{ padding:4px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
.rob {{ font-family:ui-monospace, monospace; font-size:11px; white-space:normal; word-break:break-word; max-width:320px; }}
.art {{ color:#b03a2e; font-weight:600; }}
.nthumb {{ display:inline-block; text-decoration:none; color:inherit; text-align:center; }}
.nthumb img {{ width:120px; border-radius:4px; display:block; }}
a {{ color:var(--accent); }}
ul {{ margin:6px 0; padding-left:20px; }} li {{ margin:2px 0; }}
</style></head><body>
<h1>NES hypothesis galleries: {esc(sae)}</h1>
<p class="note">{note}</p>
{''.join(sections)}
</body></html>"""
    (out_root / 'index.html').write_text(page)
    return out_root / 'index.html'


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sae', default='matryoshka_btk_1024_k16_ep20_s0')
    ap.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    ap.add_argument('--data-dir', default=str(REPO / 'data'))
    ap.add_argument('--nes-dir', default=None, help='default: results/vision/mice/eci/nes/<sae>')
    ap.add_argument('--n-match', type=int, default=4500)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    t_start = time.time()

    ds = Path(args.dataset_dir)
    nes_dir = Path(args.nes_dir) if args.nes_dir else REPO / 'results/vision/mice/eci/nes' / args.sae
    out_root = nes_dir / 'galleries'
    out_root.mkdir(parents=True, exist_ok=True)

    import json
    selected = json.loads((nes_dir / 'selected_neurons.json').read_text())
    tidy = pd.read_csv(nes_dir / 'summary.csv')
    neurons = sorted(int(k) for k in selected['neurons'])
    print(f'{len(neurons)} selected neurons: {neurons}', flush=True)

    print('loading codes + design...', flush=True)
    source = load_full_codes(args.sae, ds, args.data_dir)
    design = C.load_design(ds / 'mice/v1/annotations.csv', Path(args.data_dir) / 'mice/v1/experiment.csv')
    codes_dir = ds / 'mice/v1/eci/codes' / args.sae
    summ = {'mean': C.cached_summaries(codes_dir / 'codes_mean.npy', design,
                                       nes_dir / '_cache' / 'video_summaries_mean.npz', args.n_match)}

    print('stats tables...', flush=True)
    wide, detail = build_stats_tables(selected, tidy)
    wide.to_csv(out_root / 'stats_wide.csv')
    detail.to_csv(out_root / 'stats_detail.csv', index=False)

    (out_root / 'videos').mkdir(parents=True, exist_ok=True)
    pe = PatchEncoder(default_sae_path(args.sae, ds), device=args.device)

    print('1/3 neuron gallery (task 1)...', flush=True)
    t0 = time.time()
    gallery_for(neurons, stats_table=wide, source=source, sae_name=args.sae, out_dir=out_root / 'neurons',
               key='mean', n_tiles=12, max_per_video=1, device=args.device, sort_by='min_p', ascending=True,
               title=f'NES selected neurons: {args.sae} (primary setting)',
               note='25 neurons selected by NES (any analysis, primary setting: codes_mean, per-video mean, '
                    't-test, Bonferroni, full window). "hits" lists every (analysis, prefix, round) where the '
                    'neuron was selected with tau, p and robustness flags (see index.html one level up for the '
                    'legend). Thumbnails: top-3 frames with per-patch heatmap.',
               dataset_dir=ds, patch_encoder=pe)
    t_neurons = time.time() - t0
    print(f'   {t_neurons:.0f}s', flush=True)

    print('2/3 within-video views, family A (task 2)...', flush=True)
    t0 = time.time()
    video_a = family_a_views(design, summ, tidy, args.sae, out_root, pe, args.device, source)
    video_a.to_csv(out_root / 'videos' / 'family_a_views.csv', index=False)
    t_a = time.time() - t0
    print(f'   {len(video_a)} views, {t_a:.0f}s', flush=True)

    print('2/3 within-video views, family B (task 2)...', flush=True)
    t0 = time.time()
    video_b = family_b_views(design, summ, tidy, args.sae, out_root, pe, args.device, source)
    video_b.to_csv(out_root / 'videos' / 'family_b_views.csv', index=False)
    t_b = time.time() - t0
    print(f'   {len(video_b)} views, {t_b:.0f}s', flush=True)

    print('3/3 index.html (task 3)...', flush=True)
    write_index(out_root, tidy, wide, video_a, video_b, args.sae)

    total = time.time() - t_start
    print(f'done in {total:.0f}s -> {out_root}  '
          f'(neurons {t_neurons:.0f}s, family-A views {t_a:.0f}s, family-B views {t_b:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
