"""Rename frame-classifier run directories to names that state what the run was.

The old names came from three eras and encode nothing reliable: 'res448' (which resolution?
which context? frozen?), 'd4_decay40_res448', '448_best_s1', 'ft_b4'. Every one of them needs
config.json opened to interpret. The scheme here derives the name FROM the config, so the
name and the run can never disagree:

    res<input>_k<context>[s<stride>]_<encoder>[_sa<dim>][_q<K>]_<augment>[_<discriminator>...]

    res448_k2_ft2_d4photo        448px, context_k 2, 2 encoder blocks unfrozen, D4+photometric
    res448_k2_frozen_d4          448px, context_k 2, frozen encoder, D4 only
    res448_k2_bit2_d4photo       ... 2 blocks unfrozen but BITFIT (biases/norm gains only)
    res448_k2_frozen_sa128_d4photo   ... frozen, plus a 128-dim patch self-attention layer
    res448_k2_frozen_q4_d4photo      ... frozen, 4 patch-pooling queries instead of 1
    res448_k2s2_frozen_d4photo   ... with stride 2
    res224_k2_frozen_d4_decay20  ... disambiguated by LR schedule (see below)

The redundant 'patchgrid256_dinov2_' prefix is dropped -- every surviving run is a 256-dim
patch grid over DINOv2, so it carried no information.

DISCRIMINATORS. Several runs differ only on an axis the base name does not encode -- the
d4_decay12/20/40/60 series differ solely in LR schedule length. Rather than widen the scheme
for everyone, a colliding group gets the minimum extra suffixes needed to separate it, drawn
from lr_decay_epochs, optimizer, motion, neg_ratio, seed. A name is therefore as short as it
can be while still being unique.

IDEMPOTENT AND RE-RUNNABLE. Names are a pure function of config.json, so a run already
correctly named is skipped. Re-run this after in-flight jobs land to fold them into the scheme
(they write under whatever --tag they were launched with).

    python scripts/mice_behavior/rename_runs.py            # dry run, prints the plan
    python scripts/mice_behavior/rename_runs.py --apply    # do it

Directories are resolved by name in several scripts, so --apply also reports any code
reference that still points at an old name; fix those before relying on the result.
"""
import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

FRAME_DIR = Path('results/vision/mice/frame')
PREFIX = 'patchgrid256_dinov2_'
# not runs: aggregate outputs, search state, and the k-fold harnesses
SKIP = {'ensemble_night', 'ensemble_smoke', 'search', 'search_online_aug',
        'patchgrid256_dinov2_kfold', 'patchgrid256_dinov3_kfold'}
# tokens already implied by the base name, so they carry nothing as a discriminator
REDUNDANT = {'d4', 'd4photo', 'photo', 'noaug', 'online', 'aug', 'best', 'patchgrid256', 'dinov2'}


def base_name(c: dict) -> str:
    res = c.get('input_size') or '?'
    k = c.get('context_k', '?')
    stride = c.get('stride') or 1
    ctx = f'k{k}' + (f's{stride}' if stride != 1 else '')
    nb = c.get('unfreeze_blocks') or 0
    # 'bit<N>' vs 'ft<N>': same N blocks unfrozen, but bitfit trains only their biases/norm
    # gains. Those are different experiments with the same block count, so the mode has to be
    # in the name and not a tie-break suffix.
    enc = ('frozen' if not nb
           else f'bit{nb}' if c.get('ft_mode') == 'bitfit' else f'ft{nb}')
    # head architecture is a first-class axis of the 2026-08-14 ablation (is the HEAD the
    # bottleneck, or the encoder?), so it sits beside the encoder slot rather than being
    # demoted to a collision tie-break. Absent for the default head, which keeps every
    # pre-existing run's name byte-identical.
    head = []
    if c.get('patch_selfattn_dim'):
        head.append(f"sa{c['patch_selfattn_dim']}")
    if (c.get('pool_queries') or 1) != 1:
        head.append(f"q{c['pool_queries']}")
    aug = {'d4_photo': 'd4photo', 'd4': 'd4', 'none': 'noaug'}.get(c.get('augment'), c.get('augment') or 'augNA')
    return '_'.join(['res' + str(res), ctx, enc] + head + [aug])


def discriminators(c: dict) -> dict[str, str]:
    """Candidate suffixes keyed by axis, in priority order. Applied only on a base-name
    collision, and then only for axes that actually VARY within the colliding group -- an axis
    every member shares cannot separate anything, so including it just lengthens every name.
    A run missing the field contributes '' , which is itself a distinguishing value.

    encoder LR leads for fine-tuned runs: when two fine-tuned runs share a base name, the LR is
    overwhelmingly the axis they differ on. Frozen runs never emit it.
    """
    out = {}
    if c.get('encoder_lr') and (c.get('unfreeze_blocks') or 0):
        out['elr'] = f"elr{c['encoder_lr']:g}".replace('-', '')
    if c.get('lr_decay_epochs') is not None:
        out['decay'] = f"decay{c['lr_decay_epochs']}"
    if c.get('seed') is not None:
        out['seed'] = f"seed{c['seed']}"
    if c.get('optimizer'):
        out['opt'] = c['optimizer']
    if c.get('use_motion'):
        out['motion'] = 'motion'
    if c.get('photo_strength') not in (None, 1.0) and c.get('augment') == 'd4_photo':
        out['ps'] = f"ps{c['photo_strength']:g}"
    if c.get('dropout') is not None:
        out['do'] = f"do{c['dropout']:g}"
    if c.get('neg_ratio') not in (None, 1):
        out['neg'] = f"neg{c['neg_ratio']}"
    return out


ORDER = ['elr', 'decay', 'seed', 'opt', 'motion', 'ps', 'do', 'neg']


def separate(base: str, members: list) -> list[str] | None:
    """Shortest prefix of the VARYING discriminators that gives every member a unique name."""
    ds = [discriminators(c) for _, c in members]
    varying = [k for k in ORDER if len({d.get(k, '') for d in ds}) > 1]
    for depth in range(1, len(varying) + 1):
        cand = [base + ''.join('_' + d[k] for k in varying[:depth] if d.get(k)) for d in ds]
        if len(set(cand)) == len(cand):
            return cand
    return None


def legacy_suffix(old: str, base: str) -> str:
    """What the OLD directory name said that the config does not record.

    The d4_decay12/20/40/60 series is the case that forces this: those configs predate
    lr_decay_epochs being persisted, so all four are byte-identical on every field the scheme
    can see, and the only place the schedule length survives is the directory name. Dropping
    that would silently merge four distinct runs, so the leftover tokens are carried over.
    """
    toks = [t for t in re.split(r'[_]+', old.replace(PREFIX, ''))
            if t and t.lower() not in REDUNDANT and t.lower() not in base.lower()]
    return '_'.join(toks)


def plan() -> list[tuple[Path, str]]:
    runs = []
    for d in sorted(FRAME_DIR.iterdir()):
        if not d.is_dir() or d.name in SKIP or 'smoke' in d.name:
            continue                       # smoke runs are throwaway; naming them invites reuse
        cf = d / 'config.json'
        if not cf.exists():
            continue                       # still running / crashed: no config to name it from
        runs.append((d, json.load(open(cf))))

    groups = defaultdict(list)
    for d, c in runs:
        groups[base_name(c)].append((d, c))

    out = []
    for base, members in groups.items():
        if len(members) == 1:
            out.append((members[0][0], base))
            continue
        # widen with as few config discriminators as it takes to separate this group
        names = separate(base, members)
        # configs alone can't separate them (older schema recorded almost nothing) -- fall back
        # to what the old directory name said, which is where the difference actually lives
        if names is None:
            names = [f'{base}_{legacy_suffix(d.name, base)}'.rstrip('_') for d, _ in members]
        for (d, _), n in zip(members, names):
            out.append((d, n))
    # last-resort uniqueness: never clobber, even if it costs a suffix
    seen, final = set(), []
    for d, n in sorted(out, key=lambda x: x[1]):
        if n in seen:
            n = f'{n}_{legacy_suffix(d.name, n) or "alt"}'
        seen.add(n)
        final.append((d, n))
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='perform the renames (default: dry run)')
    args = p.parse_args()

    moves = [(d, n) for d, n in plan() if d.name != n]
    if not moves:
        print('nothing to rename -- every run already matches the scheme'); return

    width = max(len(d.name) for d, _ in moves)
    for d, n in moves:
        print(f'  {d.name:<{width}}  ->  {n}')
    print(f'\n{len(moves)} directory(ies)')

    if not args.apply:
        print('\ndry run -- pass --apply to perform these renames'); return

    for d, n in moves:
        target = FRAME_DIR / n
        if target.exists():
            print(f'  SKIP {d.name}: {n} already exists'); continue
        d.rename(target)
    print(f'renamed {len(moves)}')

    print('\nchecking code for references to the old names...')
    stale = 0
    for d, _ in moves:
        hits = subprocess.run(['grep', '-rln', '--include=*.py', '--include=*.sh', d.name,
                               'scripts/', 'src/'], capture_output=True, text=True).stdout.split()
        for h in hits:
            if '__pycache__' in h:
                continue
            print(f'  STALE  {d.name}  in  {h}'); stale += 1
    print('no stale references' if not stale else f'{stale} reference(s) need updating')


if __name__ == '__main__':
    main()
