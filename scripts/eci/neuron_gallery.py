"""
Neuron-interpretation galleries (src/eci/viz.py) for the ECI SAE on mice v1.

Output: results/vision/mice/eci/galleries/<sae name>[/<tag>]/
    neuron_XXXX.png   top-12 frames with per-patch heatmaps, least-activated 12, stage x genotype panel
    thumbs/           index thumbnails
    index.html        all neurons, sortable (default: firing rate)
    neurons.csv       per-neuron stats (+ the columns of --stats if given)
    gallery.json      settings, heatmap-vs-stored-code check, timings

Usage:
    python scripts/eci/neuron_gallery.py --prefix 128                         # first 128 neurons
    python scripts/eci/neuron_gallery.py --neurons 3,17,250 --key max
    python scripts/eci/neuron_gallery.py --source sample --prefix 128 --first 16 --tag sample_test
    python scripts/eci/neuron_gallery.py --stats nes.csv   # neurons = rows of nes.csv (column 'neuron')
    python scripts/eci/neuron_gallery.py --neurons 40 --videos wt_kdm6b_f_1_S_O,het_ash1l_f_1_F_H
        # within-video views (time course + top-6 frames of that video) -> <out>/videos/
    python scripts/eci/neuron_gallery.py --sae <20-epoch sae> --sae-path .../sae.pt --codes-dir .../codes/<name>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.viz import gallery_for, load_full_codes, load_sample_codes, video_views  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--sae', default='matryoshka_btk_1024_k16_ep20_s0')
    p.add_argument('--sae-path', default=None, help='default: dataset/mice/v1/eci/sae/<sae>/sae.pt')
    p.add_argument('--codes-dir', default=None, help='default: dataset/mice/v1/eci/codes/<sae>')
    p.add_argument('--videos', default=None, help='comma-separated observation ids: within-video views for '
                   'every (neuron, video) pair instead of the gallery')
    p.add_argument('--n-video-top', type=int, default=6)
    p.add_argument('--source', choices=['auto', 'full', 'sample'], default='auto',
                   help='full = all frames (needs merged codes), sample = training-sample frames')
    p.add_argument('--neurons', default=None, help='comma-separated neuron ids')
    p.add_argument('--prefix', type=int, default=128, help='use neurons [0, prefix) (Matryoshka prefix)')
    p.add_argument('--first', type=int, default=None, help='only the first N neurons of the selection')
    p.add_argument('--stats', default=None, help='CSV with a "neuron" column (e.g. NES output) -> gallery of those')
    p.add_argument('--key', choices=['mean', 'max'], default='mean', help='rank frames by codes_mean or codes_max')
    p.add_argument('--n-tiles', type=int, default=12)
    p.add_argument('--max-per-video', type=int, default=1)
    p.add_argument('--min-gap-s', type=float, default=2.0, help='spacing between frames of a video (if > 1 per video)')
    p.add_argument('--sort-by', default='firing_rate')
    p.add_argument('--ascending', action='store_true')
    p.add_argument('--tag', default=None, help='subdirectory of the output directory')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    ds = Path(args.dataset_dir)
    stats_table = None
    if args.stats:
        stats_table = pd.read_csv(args.stats).set_index('neuron')
        neurons = stats_table.index.values.astype(int)
    elif args.neurons:
        neurons = np.array([int(x) for x in args.neurons.split(',')])
    else:
        neurons = np.arange(args.prefix)
    if args.first:
        neurons = neurons[:args.first]

    codes_dir = Path(args.codes_dir) if args.codes_dir else ds / 'mice/v1/eci/codes' / args.sae
    if args.source == 'full' or args.source == 'auto' and (codes_dir / 'DONE').exists():
        source = load_full_codes(args.sae, ds, REPO / 'data', codes_dir=codes_dir)
    else:
        print('[INFO] using training-sample codes (full codes not merged or --source sample)', flush=True)
        source = load_sample_codes(args.sae, ds, device=args.device)
    out_dir = Path(args.out_dir) if args.out_dir else REPO / 'results/vision/mice/eci/galleries' / args.sae
    if args.tag:
        out_dir = out_dir / args.tag
    if args.videos:
        pairs = [(int(j), v) for j in neurons for v in args.videos.split(',')]
        video_views(pairs, source, out_dir / 'videos', args.sae, args.sae_path, key=args.key,
                    n_top=args.n_video_top, min_gap_s=args.min_gap_s, device=args.device, stats_table=stats_table)
        return
    gallery_for(neurons, stats_table, source=source, sae_name=args.sae, out_dir=out_dir, key=args.key,
                n_tiles=args.n_tiles, max_per_video=args.max_per_video, min_gap_s=args.min_gap_s,
                device=args.device, seed=args.seed, sort_by=args.sort_by, ascending=args.ascending,
                dataset_dir=ds, sae_path=args.sae_path)


if __name__ == '__main__':
    main()
