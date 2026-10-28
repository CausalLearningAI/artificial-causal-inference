# ECI concept atlas, v2.0: publish plan (STAGED, NOT PUBLISHED)

Staged site: `results/vision/eci_atlas_v20/site/` (built from code commit 92c3d8c; the assemble and build commands are below).
Nothing here has been published. Publishing needs the user's approval first.

## Targets

1. **Atlas**: https://claude.ai/artifact/JfJnP4CqjYWoprEq9dM8Qa, the same URL updated in place (never a new URL).
   Publish from a **claude-work** session: from the personal account, the content scan refuses about half of the clip packs ("cpio archive").
2. **Pipeline page**: https://claude.ai/artifact/JjLjfBdfkx9SuzrD61QWqf, updated in the same step. Finalise its pipeline figure for v2.0 first,
   then the page. Memory note (2026-10-23): this page is owned by the personal account, and a publish from the work account was refused
   ("public artifact from outside your organization"). So it may have to go out from a claude-personal session, using the same files.
   Its v2.0 figure and text are not part of this build. They still have to be prepared (scripts/eci/fig_pipeline_v2.py, results/vision/eci_pipeline_fig/).

Order: atlas first, then the pipeline page (figure first), then the git snapshot of atlas/.

## Totals

- Files in the new version: **304** (plus nothing else). Total size: **149.4 MB** (142.5 MiB). The limit is 256 MB per version.
- Largest file: 9.5 MB (the limit is 15 MB per file). index.html: 562 KB.
- Clip packs: **32** files, 134.2 MB. 29 are new v2.0 packs (prefix v20, 110.5 MB); 3 are online-model packs already in the live version (23.6 MB).
- Clip URLs in the asset store that the page still references (online model, uploaded earlier): 20. Nothing new needs to go to the asset store:
  the v2.0 packs ship as version files, as the frozen v2 packs did.
- Paths to remove relative to live version 64 (frozen v2, R4 and old data files; listed in publish_plan.json): **246**.
- Not re-sent: 3 unchanged live packs (antsfg_1s_4ed4326dec.mp4, antsfg_1s_d6db1848d8.mp4, antsfg_frame_bf19bd7b9b.mp4). A publish keeps the files it leaves out. Upload: **125.8 MB** in 301 files.
- Publish calls: **5** (at most 255 files and 64 MB per call; the removals as null in call 1; index.html alone in the last call).

| group | files | MB |
|---|---:|---:|
| clip packs (.webm / .mp4) | 32 | 134.2 |
| data (assets/data/*.json) | 121 | 10.8 |
| images (arena backgrounds, v0 strips) | 150 | 3.9 |
| index.html | 1 | 0.6 |

## Contents

- Models: v0 (baseline), v1 (online, the published data, linked from results/vision/eci_atlas_v37/build/online_*), and **v2.0** (replaces the frozen v2).
  The v2.0 models are v20_mice_N2f5_s950, v20_ants_N2f5_s951 (one SAE for v2 / v3 / v4) and v20_frogs_N2f5_s952 (juveniles and tadpoles). Methods: NES and NESS.
- v2.0 Outcome selector: **expected fraction of time on (HMM P(on))** (`<name>_hmm`, mean only, listed first) and **continuous** (`<name>_smooth`;
  bout rate, mean and latency). Both have their own clips. The primary readout is still an open decision with the user.
  The domain default stays v1 · NES. Note: switching from v1 to v2.0 lands on the continuous outcome, because the page keeps the current Outcome where it can.
- Heat maps: real per-token maps of the v2.0 PASS B token codes (scripts/eci/v2/run.py heat, V2_BEE eci_codes/v2_0B), drawn in one pink colour
  with the max-pooled patch outlined (src/eci/viz.py `_overlay_rgb(style="alpha", mark=True)`).
- Alignment panel: v2.0 measured by scripts/eci/atlas_align.py (results/vision/eci_atlas_v20/align/, job 1249933; continuous and P(on) at prefixes 128 / 256).
- Not on the page: the HMM Viterbi set `_hmmv` (no build). NESS for ants v3 t=2 vs t=8 (no no-day NESS run exists; NES has it from pairs_noday).

## Clip packs to upload (version files)

| file | MB |
|---|---:|
| assets/antsfg_1s_4ed4326dec.mp4 | 7.89 |
| assets/antsfg_1s_d6db1848d8.mp4 | 7.87 |
| assets/antsfg_frame_bf19bd7b9b.mp4 | 7.88 |
| assets/v20frogsh_1s_e219e848e1.webm | 2.22 |
| assets/v20frogsh_frame_bea71f89a7.webm | 2.03 |
| assets/v20frogss_1s_0931c68985.webm | 3.75 |
| assets/v20frogss_frame_d765af50b0.webm | 3.48 |
| assets/v20miceh_1s_8087987e92.webm | 5.02 |
| assets/v20miceh_frame_8b6f061de3.webm | 4.19 |
| assets/v20mices_1s_8f8139d865.webm | 9.51 |
| assets/v20mices_frame_b0dd30b55d.webm | 7.88 |
| assets/v20tadh_1s_eeceaa17df.webm | 0.86 |
| assets/v20tadh_frame_5093422d77.webm | 0.67 |
| assets/v20tads_1s_b494d3b6b5.webm | 1.53 |
| assets/v20tads_frame_de8a1e792c.webm | 1.26 |
| assets/v20v2ph_1s_a4c7e1255d.webm | 1.84 |
| assets/v20v2ph_frame_4435eb90f2.webm | 1.54 |
| assets/v20v2ps_1s_befb7c886e.webm | 4.27 |
| assets/v20v2ps_frame_c6a9b37e68.webm | 3.77 |
| assets/v20v3ph_1s_327e6db522.webm | 4.46 |
| assets/v20v3ph_1s_8402f512cf.webm | 0.13 |
| assets/v20v3ph_frame_518819f988.webm | 3.95 |
| assets/v20v3ps_1s_fda2d6aabd.webm | 5.76 |
| assets/v20v3ps_frame_df99b44d6c.webm | 5.15 |
| assets/v20v4ph_1s_6530de9c39.webm | 2.05 |
| assets/v20v4ph_1s_b002e4e5d8.webm | 4.22 |
| assets/v20v4ph_frame_8fbbbca3fb.webm | 0.33 |
| assets/v20v4ph_frame_d379521405.webm | 5.11 |
| assets/v20v4ps_1s_e7328fa8ff.webm | 7.39 |
| assets/v20v4ps_1s_eb74e8d988.webm | 5.95 |
| assets/v20v4ps_frame_5919432ea7.webm | 3.25 |
| assets/v20v4ps_frame_cbd116ddc1.webm | 8.97 |

## Publish steps (claude-work session, after approval)

1. `Artifact read` https://claude.ai/artifact/JfJnP4CqjYWoprEq9dM8Qa and `list scope files`. Check that the live version is still 64.
   If it is not, recompute the removals: every live path that is not in the staged site gets null.
2. Publish the 5 batches of `results/vision/eci_atlas_v20/publish_plan.json` in order, with `url` = the atlas,
   `root` = results/vision/eci_atlas_v20/site, and `files` = the batch map (null = remove). Send index.html last, in its own call.
3. Open the page. Check v2.0 on each domain: the Outcome toggle, a clip with pink heat and an outline, and the Alignment panel.
4. Pipeline page: finalise the v2.0 figure, then publish https://claude.ai/artifact/JjLjfBdfkx9SuzrD61QWqf (from the account that owns it).
5. Git snapshot: copy `site/index.html` to `atlas/index.html`. Write `atlas/manifest.json` from `site/manifest.json` with artifact_version set to
   the new version id, the build command and the code commit. Keep `atlas/asset_map.json` (no new asset uploads). Then commit and push
   ("feat(atlas): publish version N ... v2.0").

## Build commands (CPU Slurm, -p defaultp)

```
ATLAS_ALIGN_V2_ROOT=results/vision/eci_v2_0B ATLAS_ALIGN_OUT=results/vision/eci_atlas_v20/align sbatch --export=ALL scripts/eci/atlas_align.sh
for k in v2_mice v2_ants_v2p v2_ants_v3p v2_ants_v4p v2_frogs v2_frogs_tad; do for v in hmm smooth; do
  ATLAS_ALIGN=results/vision/eci_atlas_v20/align ATLAS_OUT=results/vision/eci_atlas_v20 ATLAS_V2_ROOT=results/vision/eci_v2_0B ATLAS_HEAT_STYLE=alpha \
  ARGS="--dom $k --model new --variant $v --steps prep,data,v2heat,arena,render,dump [--pack-mb 7 for hmm]" sbatch --export=ALL scripts/eci/build_atlas_v37.sh
done; done
ATLAS_ALIGN=results/vision/eci_atlas_v20/align ATLAS_OUT=results/vision/eci_atlas_v20 ATLAS_V2_ROOT=results/vision/eci_v2_0B ARGS="--steps assemble" sbatch --export=ALL scripts/eci/build_atlas_v37.sh
sbatch (srun) python scripts/eci/atlas_v37_pagetest.py --site results/vision/eci_atlas_v20/site --shots results/vision/eci_atlas_v20/pagetest
```

## Page test

- Builder check (check_site, at assemble, job 1251412): 121 data files and 182 local media references, **0 missing**. 5 unused files removed. Largest file 9.5 MB. Total version 149.4 MB in 304 files.
- A separate re-check on the staged files: every data JSON parses, and every reference exists (0 missing). The page carries the labels v2.0 and 'expected fraction of time on (HMM P(on))'.
- Headless browser test, scripts/eci/atlas_v37_pagetest.py (Chromium, job 1253078, 31 min; report and screenshots in pagetest/): it walks every domain x model x outcome x method x aggregation x comparison.
  Result: **0 JS errors**. 1267 report lines. 274 search views listed, 144 of them on v2.0. All 28 v2.0 views with at least one discovery open a neuron detail.
  Clip packs: 137 checked, 136 loaded and looping. The one not loaded is an asset-store /_blob/ URL of the online model, which cannot resolve on a local server (expected).
  Candidate lists shown next to discoveries: 0. The Outcome toggle works on all 3 domains. P(on) is greyed for v0 / v1 ('not computed with this model / method'), as intended.
  Test fix: the test now resets to the domain default before switching model (atlas_v37_pagetest.py). Before the fix it timed out, because v0 / v1 are correctly disabled under the v2.0-only P(on) outcome.
- Heat style checked by eye on decoded clip frames (ants v2 and mice v20miceh_frame): pink single-colour heat with the outline on the max-pooled patch.
