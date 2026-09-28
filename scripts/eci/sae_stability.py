"""
Stability of the ECI SAE dictionary across two training seeds: for each decoder atom
of seed A (within each Matryoshka prefix), the max cosine similarity to seed B's atoms
(same prefix, and full dictionary). Written to <sae A dir>/stability_vs_s{B}.json.

Usage:
    python scripts/eci/sae_stability.py                       # s0 vs s1, default name
    python scripts/eci/sae_stability.py --a <dirA> --b <dirB>
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.sae import decoder_stability, load_sae  # noqa: E402


def main():
    root = REPO / 'dataset/mice/v1/eci/sae'
    p = argparse.ArgumentParser()
    p.add_argument('--a', default=str(root / 'matryoshka_btk_1024_k16_ep20_s0'))
    p.add_argument('--b', default=str(root / 'matryoshka_btk_1024_k16_ep20_s1'))
    args = p.parse_args()
    sae_a, _, _ = load_sae(Path(args.a) / 'sae.pt')
    sae_b, _, ck_b = load_sae(Path(args.b) / 'sae.pt')
    res = decoder_stability(sae_a.W_dec.data, sae_b.W_dec.data, sae_a.prefixes)
    for m, r in res.items():
        print(f'  m={m:>5}  ' + '  '.join(f'{k} {v:.3f}' for k, v in r.items()))
    out = Path(args.a) / f"stability_vs_s{ck_b['args']['seed']}.json"
    out.write_text(json.dumps({'a': args.a, 'b': args.b, 'prefixes': res}, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
