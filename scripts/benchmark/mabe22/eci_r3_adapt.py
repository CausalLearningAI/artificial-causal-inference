"""
MABe22 ECI round 3, experiment E6: self-supervised ADAPTATION of DINOv2-base on unlabelled MABe22 mouse userTrain frames.

Rule (scripts/benchmark/mabe22/eci_round3.sh, E6): adapt DINOv2-base with the repo's scripts/mice_behavior/ssl_dapt.py
objective on unlabelled userTrain frames only, fixed budget ~2 GPU-hours, then run recipe R on the adapted encoder.
This script produces ONLY the adapted encoder; encoding + scoring happen elsewhere.

Objective (imported from ssl_dapt.py: block_mask, make_targets, patch_features; the training step is a line-for-line
copy of ssl_dapt's inner loop): data2vec-2.0-style masked patch-feature prediction. Teacher = EMA of the student, sees
the clean image; student sees ~50% of the 32x32 patch grid replaced by the (trainable) mask token, in rectangles of
>= 16 patches, and a 2-layer MLP predictor regresses the teacher's targets at the masked positions. Target = mean of
the top 4 blocks' patch tokens, each layer-normed, then layer-normed again. Loss smooth-L1 (beta 1). Trainable: the
last 2 transformer blocks (layer-wise LR decay 0.65), the final layernorm and the mask token; everything else frozen.
AdamW wd 0.05, encoder LR 3e-5 (predictor 10x), linear warmup over 1/12 of the steps (ssl_dapt: 1 of 12 epochs) then
cosine to 0.01x, EMA momentum 0.996 -> 1 on a cosine, grad-clip 3.0, fp16 autocast + GradScaler, batch 64, 448 px.

Data: ONLY dataset/benchmark/mabe22/videos/mouse_userTrain/<clip>.avi (784 clips x 1,801 frames). No mouse_submission
clip and no label file is ever opened. Each clip is copied to the job's local staging dir and decoded with
eci_common.decode_clip (ffmpeg611 binary), every --stride-th frame kept, all in RAM as uint8 224x224x3. --holdout-clips
clips (seeded) are NOT trained on: they give the drift probe (as ssl_dapt's never-trained-on val probe) and the
frames of the --verify check. Input path = eci_encode.Enc.__call__: torch bicubic 224 -> 448, .round().clamp(0,255),
/255, ImageNet mean/std. Augmentation: one random D4 element per frame (exact pixel permutation, applied to the uint8
224 frame before upsampling). ssl_dapt's photometric path is not used: it lives in its PIL/JPEG-cache dataset class
and has no array-based code path.

Kept network: the STUDENT (as ssl_dapt, which saves the student's weights as the encoder Stage B consumes). The
predictor is discarded (part of the objective, not of the representation). Output (results/vision/benchmark_mabe22/
eci/r3/adapt/): encoder.pt (full state_dict for AutoModel.from_pretrained('facebook/dinov2-base')), train.json.

    PY=eci_r3_adapt EXTRA="--smoke" sbatch -p gpu --gres=gpu:L40S:1 ... scripts/benchmark/mabe22/eci.sh
    PY=eci_r3_adapt EXTRA="--steps N" sbatch -p gpu --gres=gpu:L40S:1 --time=03:00:00 --mem=64G ... eci.sh
    PY=eci_r3_adapt EXTRA="--verify" sbatch -p gpu --gres=gpu:L40S:1 --time=00:20:00 ... eci.sh
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mice_behavior'))
from eci_common import OUT, REPO, VID, decode_clip, log, stage_dir  # noqa: E402

SET = 'mouse_userTrain'
MODEL_ID = 'facebook/dinov2-base'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
N_VIDEO_FRAMES = 1801


def clips_all():
    c = sorted(p.stem for p in (VID / SET).glob('*.avi'))
    assert len(c) == 784, len(c)
    return c


def decode_many(clips, frame_idx, stage, workers):
    """decode each clip (local copy + ffmpeg) and keep frame_idx -> (len(clips)*len(frame_idx), 224, 224, 3) uint8."""
    k = len(frame_idx)
    out = np.empty((len(clips) * k, 224, 224, 3), np.uint8)
    n_frames = {}

    def one(i):
        a = decode_clip(SET, clips[i], stage)
        n_frames[clips[i]] = int(len(a))
        if len(a) != N_VIDEO_FRAMES:
            raise RuntimeError(f'{clips[i]}: {len(a)} frames, expected {N_VIDEO_FRAMES}')
        out[i * k:(i + 1) * k] = a[frame_idx]

    with ThreadPoolExecutor(workers) as ex:
        for j, _ in enumerate(ex.map(one, range(len(clips)))):
            if (j + 1) % 100 == 0:
                log(f'  decoded {j + 1}/{len(clips)} clips')
    return out, n_frames


class Prep:
    """uint8 (B, 224, 224, 3) numpy -> normalised (B, 3, size, size) float GPU tensor, exactly eci_encode.Enc.__call__."""

    def __init__(self, torch, dev, size):
        self.torch, self.dev, self.size = torch, dev, size
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, :, None, None]

    def __call__(self, u8, d4=None):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float()
        if d4 is not None:  # exact D4 element per frame: bit 2 = horizontal flip, bits 0-1 = number of 90-degree turns
            x = torch.stack([torch.rot90(xi.flip(-1) if g & 4 else xi, int(g & 3), dims=(-2, -1))
                             for xi, g in zip(x, d4)])
        if self.size != x.shape[-1]:
            x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode='bicubic',
                                                align_corners=False).round().clamp(0, 255)
        return (x / 255 - self.mean) / self.std


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def train(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel
    from ssl_dapt import block_mask, make_targets, patch_features  # the ssl_dapt objective, imported not copied

    t_job = time.time()
    dev = torch.device('cuda')
    gpu = torch.cuda.get_device_name(0)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = stage_dir()

    # ---- data: userTrain only ------------------------------------------------------------------------------------
    clips = clips_all()
    hold = sorted(rng.choice(clips, size=args.holdout_clips, replace=False).tolist())
    train_clips = [c for c in clips if c not in set(hold)]
    if args.n_clips:
        train_clips = train_clips[:args.n_clips]
    fidx = np.arange(0, N_VIDEO_FRAMES, args.stride)
    # probe frames (held-out clips only, never trained on): evenly spread, offset so they are not on the train grid
    pidx = np.linspace(7, N_VIDEO_FRAMES - 8, args.probe_per_clip).round().astype(int)
    log(f'{gpu} | {len(train_clips)} train clips x {len(fidx)} frames (stride {args.stride}) = '
        f'{len(train_clips) * len(fidx):,} frames | holdout clips {hold}')
    t0 = time.time()
    frames, nf = decode_many(train_clips, fidx, stage, args.decode_workers)
    probe_u8, _ = decode_many(hold, pidx, stage, args.decode_workers)
    t_decode = time.time() - t0
    log(f'decoded {len(frames):,} train + {len(probe_u8)} probe frames in {t_decode / 60:.1f} min '
        f'({frames.nbytes / 1024 ** 3:.1f} GiB in RAM)')

    # ---- student / teacher (ssl_dapt.main, same parameter groups) --------------------------------------------------
    student = AutoModel.from_pretrained(MODEL_ID).to(dev)
    student.requires_grad_(False)
    blocks = student.encoder.layer
    n = min(args.unfreeze_blocks, len(blocks))
    groups = []
    for depth, blk in enumerate(reversed(blocks[-n:])):
        blk.requires_grad_(True)
        groups.append({'params': list(blk.parameters()), 'lr': args.encoder_lr * (args.layerwise_decay ** depth)})
    student.layernorm.requires_grad_(True)
    groups.append({'params': list(student.layernorm.parameters()), 'lr': args.encoder_lr})
    student.embeddings.mask_token.requires_grad_(True)
    groups.append({'params': [student.embeddings.mask_token], 'lr': args.encoder_lr})
    teacher = deepcopy(student).eval()
    teacher.requires_grad_(False)
    train_names = [k for k, p in student.named_parameters() if p.requires_grad]
    init = {k: p.detach().float().cpu().clone() for k, p in student.named_parameters() if p.requires_grad}
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    hid = student.config.hidden_size
    predictor = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, hid)).to(dev)
    groups.append({'params': list(predictor.parameters()), 'lr': args.encoder_lr * 10})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    total_steps = args.steps
    warmup_steps = int(round(total_steps * args.warmup_frac))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.01 + 0.5 * (1 - 0.01) * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    grid = args.input_size // 14
    prep = Prep(torch, dev, args.input_size)
    log(f'student: last {n}/{len(blocks)} blocks + final layernorm + mask token trainable, {trainable / 1e6:.2f}M '
        f'params ({100 * trainable / total_params:.2f}%) | grid {grid}x{grid} mask {args.mask_ratio:.0%} '
        f'blocks>={args.mask_min_patches} | top {args.target_layers} layers | steps {total_steps} warmup '
        f'{warmup_steps} batch {args.batch_size} lr {args.encoder_lr:g} wd {args.weight_decay:g} '
        f'ema {args.ema_momentum} aug {args.augment}')

    probe_x = prep(probe_u8)
    stock = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    stock.requires_grad_(False)
    ref_feat = F.normalize(patch_features(stock, probe_x), dim=-1)
    del stock
    torch.cuda.empty_cache()

    def drift_now(model):
        model.eval()
        cur = patch_features(model, probe_x)
        return float(1 - (F.normalize(cur, dim=-1) * ref_feat).sum(-1).mean())

    # ---- training loop (ssl_dapt's inner loop, step-budgeted) -----------------------------------------------------
    history, drifts = [], [{'step': 0, 'drift_student': drift_now(student)}]
    win = {'loss': 0.0, 'st_cos': 0.0, 'target_std': 0.0, 'pred_std': 0.0, 'n': 0}
    order, pos, epoch = rng.permutation(len(frames)), 0, 0
    gstep, t_train, t_meas, s_meas = 0, 0.0, None, None
    torch.cuda.synchronize()
    t_start = time.time()
    while gstep < total_steps:
        if pos + args.batch_size > len(order):
            order, pos, epoch = rng.permutation(len(frames)), 0, epoch + 1
        bi = np.sort(order[pos:pos + args.batch_size])
        pos += args.batch_size
        d4 = rng.integers(0, 8, size=len(bi)) if args.augment == 'd4' else None
        student.train()
        x = prep(frames[bi], d4)
        mask = block_mask(len(x), grid, args.mask_ratio, args.mask_min_patches, rng).to(dev)
        with torch.amp.autocast('cuda'):
            tgt = make_targets(teacher, x, args.target_layers)
            sh = student(pixel_values=x, bool_masked_pos=mask).last_hidden_state[:, 1:]
            pred = predictor(sh)
        pred, tgt = pred.float()[mask], tgt.float()[mask]
        loss = F.smooth_l1_loss(pred, tgt, beta=1.0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params'] if p.requires_grad], 3.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        gstep += 1
        mom = 1 - (1 - args.ema_momentum) * 0.5 * (1 + math.cos(math.pi * gstep / total_steps))
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), student.parameters()):
                tp.mul_(mom).add_(sp.detach(), alpha=1 - mom)
            for tb, sb in zip(teacher.buffers(), student.buffers()):
                tb.copy_(sb)
            win['loss'] += loss.item()
            win['st_cos'] += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
            win['target_std'] += tgt.std(0).mean().item()
            win['pred_std'] += pred.std(0).mean().item()
            win['n'] += 1
        if not math.isfinite(loss.item()):
            raise SystemExit(f'non-finite loss at step {gstep}')
        if gstep == args.meas_from:
            torch.cuda.synchronize()
            t_meas, s_meas = time.time(), gstep
        if gstep % args.log_every == 0 or gstep == total_steps:
            rec = {k: win[k] / win['n'] for k in ('loss', 'st_cos', 'target_std', 'pred_std')}
            rec.update(step=gstep, epoch=epoch, lr=opt.param_groups[0]['lr'], ema_momentum=mom,
                       elapsed_s=time.time() - t_start)
            history.append(rec)
            win = {k: 0.0 if k != 'n' else 0 for k in win}
            log(f'step {gstep:>6}/{total_steps} ep {epoch} loss {rec["loss"]:.4f} st_cos {rec["st_cos"]:.4f} '
                f'target_std {rec["target_std"]:.4f} pred_std {rec["pred_std"]:.4f} lr {rec["lr"]:.2e} '
                f'mom {mom:.5f} ({rec["elapsed_s"] / 60:.1f} min)')
            if rec['target_std'] < args.collapse_std:
                raise SystemExit(f'COLLAPSE: target_std {rec["target_std"]:.4f} < {args.collapse_std} at step '
                                 f'{gstep}; aborting without writing a checkpoint')
        if gstep % args.drift_every == 0 or gstep == total_steps:
            drifts.append({'step': gstep, 'drift_student': drift_now(student), 'drift_teacher': drift_now(teacher)})
            log(f'  drift (1 - cos to stock, {len(probe_x)} held-out frames) student '
                f'{drifts[-1]["drift_student"]:.4f} teacher {drifts[-1]["drift_teacher"]:.4f}')
    torch.cuda.synchronize()
    t_train = time.time() - t_start
    sps = (gstep - s_meas) / (time.time() - t_meas) if t_meas is not None and gstep > s_meas else None
    log(f'trained {gstep} steps ({gstep * args.batch_size / len(frames):.2f} epochs of {len(frames):,} frames) in '
        f'{t_train / 60:.1f} min; steady-state {sps if sps is None else round(sps, 4)} steps/s '
        f'(from step {args.meas_from}, includes drift probes)')

    # ---- parameter drift from init -------------------------------------------------------------------------------
    def pdrift(model):
        sd = dict(model.named_parameters())
        by = {}
        for k in train_names:
            g = 'mask_token' if 'mask_token' in k else ('layernorm' if k.startswith('layernorm') else
                                                         'block_' + k.split('.')[2])
            d, z = by.setdefault(g, [0.0, 0.0]), init[k]
            d[0] += float((sd[k].detach().float().cpu() - z).pow(2).sum())
            d[1] += float(z.pow(2).sum())
        return {g: {'l2_delta': math.sqrt(a), 'l2_init': math.sqrt(b), 'rel': math.sqrt(a / b) if b > 0 else None}
                for g, (a, b) in by.items()}

    par = {'student': pdrift(student), 'teacher': pdrift(teacher)}
    for who, d in par.items():
        log(f'param drift {who}: ' + ', '.join(f'{g} rel {v["rel"]:.4f}' if v['rel'] is not None else
                                               f'{g} |d| {v["l2_delta"]:.4f}' for g, v in d.items()))

    # ---- save (student, full state_dict) and check it loads -------------------------------------------------------
    sd = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
    torch.save(sd, out_dir / 'encoder.pt')
    chk = AutoModel.from_pretrained(MODEL_ID)
    chk.load_state_dict(torch.load(out_dir / 'encoder.pt', map_location='cpu'))
    log(f'saved {out_dir / "encoder.pt"} ({len(sd)} tensors; reload into AutoModel strict OK)')
    res = {
        'experiment': 'MABe22 ECI round 3, E6 (adapted encoder)', 'script': 'scripts/benchmark/mabe22/eci_r3_adapt.py',
        'git_head_at_run': git_head(), 'slurm_job': os.environ.get('SLURM_JOB_ID'), 'gpu': gpu,
        'saved_network': 'student (as ssl_dapt.py); predictor discarded; teacher not saved',
        'objective_source': 'block_mask, make_targets, patch_features imported from scripts/mice_behavior/ssl_dapt.py;'
                            ' training step copied from ssl_dapt.main',
        'recipe': vars(args) | {'warmup_steps': warmup_steps, 'grid': grid, 'trainable_params': int(trainable),
                                'trainable_tensors': train_names, 'predictor_lr': args.encoder_lr * 10,
                                'amp': 'fp16 autocast + GradScaler', 'grad_clip': 3.0,
                                'input': 'torch bicubic 224->448, round, clamp, /255, ImageNet mean/std',
                                'photometric_aug': False},
        'data': {'set': SET, 'n_clips_total': 784, 'n_train_clips': len(train_clips), 'holdout_clips': hold,
                 'train_frame_idx': f'0..1800 step {args.stride} ({len(fidx)} per clip)',
                 'n_train_frames': int(len(frames)), 'probe_frame_idx': pidx.tolist(),
                 'n_probe_frames': int(len(probe_u8)), 'video_frames_per_clip': sorted(set(nf.values())),
                 'labels_used': False, 'submission_clips_used': False},
        'steps': gstep, 'batch_size': args.batch_size, 'epochs': gstep * args.batch_size / len(frames),
        'time_s': {'decode': t_decode, 'train': t_train, 'job_total': time.time() - t_job},
        'steady_steps_per_s': sps, 'loss_first': history[0]['loss'] if history else None,
        'loss_last': history[-1]['loss'] if history else None, 'history': history, 'drift': drifts,
        'param_drift': par,
    }
    (out_dir / 'train.json').write_text(json.dumps(res, indent=1))
    log(f'wrote {out_dir / "train.json"}')
    return out_dir


def verify(args):
    """stock vs adapted last-layer patch tokens on 8 held-out userTrain frames, fp32, eci_encode's input path."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device('cuda')
    out_dir = Path(args.out)
    tj = json.loads((out_dir / 'train.json').read_text())
    hold = tj['data']['holdout_clips']
    u8, _ = decode_many(hold, np.array([307, 1207]), stage_dir(), 4)  # 2 frames per held-out clip, off the train grid
    u8 = u8[:8]
    prep = Prep(torch, dev, args.input_size)
    x = prep(u8)
    stock = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    ad = AutoModel.from_pretrained(MODEL_ID)
    ad.load_state_dict(torch.load(out_dir / 'encoder.pt', map_location='cpu'))
    ad = ad.to(dev).eval()
    with torch.inference_mode():
        hs = stock(pixel_values=x).last_hidden_state[:, 1:].float()
        ha = ad(pixel_values=x).last_hidden_state[:, 1:].float()
    assert torch.isfinite(ha).all()
    cos = F.cosine_similarity(hs, ha, dim=-1)  # (8, 1024)
    sd_s = hs.reshape(-1, hs.shape[-1]).std(0)
    sd_a = ha.reshape(-1, ha.shape[-1]).std(0)
    res = {'checkpoint': str(out_dir / 'encoder.pt'), 'n_frames': int(len(x)), 'frames': f'{hold} x [307, 1207]',
           'precision': 'fp32, tf32 off', 'cos_mean': float(cos.mean()), 'cos_min': float(cos.min()),
           'cos_max': float(cos.max()), 'cos_per_frame': cos.mean(1).tolist(),
           'std_per_dim_mean_stock': float(sd_s.mean()), 'std_per_dim_mean_adapted': float(sd_a.mean()),
           'std_ratio_adapted_over_stock': float(sd_a.mean() / sd_s.mean()),
           'frac_dims_std_below_0.1x_stock': float((sd_a < 0.1 * sd_s).float().mean()),
           'token_norm_mean_stock': float(hs.norm(dim=-1).mean()), 'token_norm_mean_adapted': float(ha.norm(dim=-1).mean()),
           'slurm_job': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(0)}
    (out_dir / 'verify.json').write_text(json.dumps(res, indent=1))
    log('verify: ' + json.dumps({k: v for k, v in res.items() if k != 'cos_per_frame'}))
    if not res['cos_mean'] < 1 - 1e-6:
        raise SystemExit('verify FAILED: adapted tokens identical to stock')
    if res['std_ratio_adapted_over_stock'] < 0.3:
        raise SystemExit('verify FAILED: adapted tokens look collapsed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=None, help='fixed optimizer-step budget (from the smoke throughput)')
    p.add_argument('--stride', type=int, default=15, help='keep video frames 0, 15, ..., 1800 (2 fps, 121 per clip)')
    p.add_argument('--holdout-clips', type=int, default=4, help='userTrain clips kept out of training (probe/verify)')
    p.add_argument('--probe-per-clip', type=int, default=64)
    p.add_argument('--n-clips', type=int, default=0, help='limit train clips (smoke only)')
    p.add_argument('--input-size', type=int, default=448)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--unfreeze-blocks', type=int, default=2)
    p.add_argument('--encoder-lr', type=float, default=3e-5)
    p.add_argument('--layerwise-decay', type=float, default=0.65)
    p.add_argument('--mask-ratio', type=float, default=0.5)
    p.add_argument('--mask-min-patches', type=int, default=16)
    p.add_argument('--target-layers', type=int, default=4)
    p.add_argument('--ema-momentum', type=float, default=0.996)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--warmup-frac', type=float, default=1 / 12, help='ssl_dapt: 1 warmup epoch of 12')
    p.add_argument('--augment', default='d4', choices=['none', 'd4'])
    p.add_argument('--collapse-std', type=float, default=0.05)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--drift-every', type=int, default=1000)
    p.add_argument('--meas-from', type=int, default=50, help='throughput is timed from this step on')
    p.add_argument('--decode-workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=str(OUT / 'r3/adapt'))
    p.add_argument('--smoke', action='store_true', help='24 clips, 300 steps, own output dir, then --verify on it')
    p.add_argument('--verify', action='store_true', help='only the stock-vs-adapted check on --out/encoder.pt')
    args = p.parse_args()
    if args.smoke:
        args.n_clips, args.steps, args.log_every, args.drift_every = 24, 300, 50, 150
        args.out = str(OUT / 'r3/adapt_smoke')
    if args.verify:
        return verify(args)
    if not args.steps:
        raise SystemExit('--steps is required (fix it from the smoke-run throughput)')
    train(args)
    if args.smoke:
        verify(args)


if __name__ == '__main__':
    main()
