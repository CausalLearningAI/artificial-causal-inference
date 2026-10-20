"""
ECI v2: the frozen representation pipeline of the do-pilot ECI atlas (one SAE per domain).

    frames  -> video.py      decode the 5 fps grid once from the native source (ffmpeg 6.1.1, rgb24, no JPEG)
            -> pipeline.py   canonical frame (mice rot90 odour corner top right; frogs hflip mutant side right) or an
                             animal-following crop (tadpoles 96 px -> 384 px); mask + encoder per series
            -> mask.py       TIGHT mask: token kept iff >= 25% of its 16 x 16 patch are animal pixels (pixel cue per
                             domain: dark vs the video's background; frogs: darker than the frog-free background, ROI)
            -> encode.py     DINOv3-L/16 block 20 (bf16, SDPA, pinned revision), native resolution, never upsampled
            -> context.py    optional spatial context [z_t, mean of the other kept tokens in the w x w window]
            -> sae.py        Matryoshka BatchTopK SAE, k 16, theta* on calibration videos, deployment seed by FVE
            -> readout.py    frame value = max over kept tokens; 0.6 s centred mean (the only temporal step); fnorm
                             firing rule; per-video time spent / bout rate / latency
            -> score.py      behaviour alignment (cross-fitted R~, AUROC) - labels only here
            -> heatmap.py    per-token activation and exact contribution maps of a neuron

Frozen spec: configs/eci_v2/representation.json (decided by scripts/eci/v2/freeze.py, results/vision/eci_v2/freeze/).
Domain configs: configs/eci_v2/{mice,ants,frogs}.json. Entry point: scripts/eci/v2/run.py (+ nes.sh for NES / NESS).
"""
