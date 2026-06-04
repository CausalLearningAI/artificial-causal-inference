# Mice Data Structure

Autism spectrum disorder (ASD) mouse model experiments. Quadruplets of mice from three ASD-associated genetic lines are recorded during odor-exposure sessions. The **treatment** is genotype (wildtype vs. heterozygous knockout), and **outcomes** are social behaviors labeled by annotators (nose-nose, nose-tail sniffing interactions).

---

## Directory Layout

```
data/mice/
├── source/                        # Raw videos from acquisition (665 .mp4 files)
├── v1/                            # Standardized version with annotations
│   ├── experiment.csv             # Metadata table (432 observations)
│   ├── observations/
│   │   ├── full/                  # Standardized videos (426 .mp4)
│   │   └── metadata.json          # Aggregate video stats
│   └── annotations/               # Per-observation behavior labels (144 .csv)
└── v2/                            # Standardized version, no annotations yet
    ├── experiment.csv             # Metadata table (216 observations)
    └── observations/
        ├── full/                  # Standardized videos (216 .mp4)
        └── metadata.json          # Aggregate video stats
```

---

## Source Videos (`source/`)

- **665 raw .mp4 files** at 30 FPS with variable resolution
- Span recording rounds `rd11` through `rd63` (July 2024 – January 2026)
- Total size: ~957 GB
- Filename convention: `{YYYY-MM-DD}_{HH-MM-SS}_BHVScreen_{pool}_{odor_type}_{phase}.mp4`

### Filename Fields

| Field        | Examples                        | Description                                     |
|--------------|---------------------------------|-------------------------------------------------|
| `pool`       | `rd11_1`, `rd12`, `rd36_2`      | Recording day + cage position                   |
| `odor_type`  | `SocialOdor`, `FearOdor`, `SocialOdor2` | Odor stimulus type                     |
| `phase`      | `Habit`, `Test`, `Post`         | Session phase (H / O / P)                       |

---

## Experiment Metadata (`experiment.csv`)

One row per observation. Present in both `v1/` and `v2/`.

| Column             | Description                                              |
|--------------------|----------------------------------------------------------|
| `pool`             | Recording group ID (e.g. `rd13`, `rd36_2`)              |
| `line`             | Genetic line: `ash1l`, `kdm6b`, `kmt5b`                  |
| `sex`              | `m` (male) or `f` (female)                               |
| `genotype`         | **Treatment**: `wt` (wildtype) or `het` (heterozygous)  |
| `phase`            | `H` (Habituation), `O` (Odor/Test), `P` (Post)          |
| `odor`             | `S` (Social) or `F` (Fear)                               |
| `seed`             | Within-pool cage index (1–3)                             |
| `date` / `time`    | Recording date and start time                            |
| `annotator`        | Annotator ID (empty if unannotated)                      |
| `observation_file` | Source video filename                                    |
| `annotation_file`  | Matching BORIS annotation CSV (empty if unannotated)     |
| `valid`            | `1` if observation passes QC                             |
| `observation_id`   | Unique key: `{genotype}_{line}_{sex}_{seed}_{odor}_{phase}` |
| `start_frame`      | Start frame index (usually 0)                            |
| `end_frame`        | End frame index (54000 for Habit, 27000 for Test/Post at 30 FPS) |

---

## Standardized Videos (`observations/full/`)

Produced by `src/data/standardize.py` from source videos.

| Property       | Value              |
|----------------|--------------------|
| Format         | .mp4 (H.264)       |
| FPS            | 5                  |
| Resolution     | 512 × 512          |
| Channels       | YUV                |
| Duration       | 15–30 min          |

### `metadata.json`

Aggregate stats over both `source` and `full` sets (count, FPS, resolution, duration range, total size).

---

## Annotation Files (`v1/annotations/`)

One CSV per annotated observation (144 files), exported from [BORIS](https://www.boris.unito.it/) (Behavioral Observation Research Interactive Software). Filenames mirror source video names.

### Columns

| Column                  | Description                                     |
|-------------------------|-------------------------------------------------|
| `Behavior`              | Behavior label (`nose-nose`, `nose-tail`)        |
| `Behavioral category`   | Category (`Smell`)                              |
| `Behavior type`         | `STATE` (interval event)                        |
| `Start (s)` / `Stop (s)` | Time boundaries in seconds                    |
| `Image index start/stop`| Frame boundaries (at source 30 FPS)            |
| `behavior_type`         | Short code: `nn`, `np`, `nt` (see below)        |
| `agent1(active)`        | ID of the initiating mouse (1–4)                |
| `agent2`                | ID of the receiving mouse (1–4)                 |

### Behavior Codes

| Code | Full name        | Symmetrical | Description                           |
|------|------------------|-------------|---------------------------------------|
| `nn` | nose-nose        | Yes         | Both mice sniff each other's nose     |
| `np` | nose-nose (pass) | No          | One mouse sniffs the other's nose     |
| `nt` | nose-tail        | No          | One mouse sniffs the other's tail     |

These map to outcome dimensions `[nn, np, nt]` in the dataset config.

---

## Experiment Design

Each recording pool (`rdXX`) captures one quadruplet of mice from the same genetic line (one litter). The session follows **6 sequential stages** across two hormonal odor conditions:

| Stage | `phase` | `odor` | Duration | Frames (30 FPS) | Description                          |
|-------|---------|--------|----------|-----------------|--------------------------------------|
| 1     | H       | S      | 30 min   | 54 000          | Habituation before social odor        |
| 2     | O       | S      | 15 min   | 27 000          | Social hormonal odor exposure         |
| 3     | P       | S      | 15 min   | 27 000          | Post social odor                      |
| 4     | H       | F      | 30 min   | 54 000          | Habituation before fear odor          |
| 5     | O       | F      | 15 min   | 27 000          | Fear hormonal odor exposure           |
| 6     | P       | F      | 15 min   | 27 000          | Post fear odor                        |

The two odor stimuli are **social** (`S`) and **fear** (`F`) hormonal cues. Each pool yields exactly 6 observation files.

---

## Versions

| Version | Cage composition       | Pools | Observations | Annotations   | Date range          |
|---------|------------------------|-------|--------------|---------------|---------------------|
| `v1`    | 3 WT + 1 HET (labeled) | 72    | 432          | 144 (partial) | Jul 2024 – Jan 2026 |
| `v2`    | 3 WT + 1 HET (mixed)   | 36    | 216          | none          | Apr 2025 – Jan 2026 |

**v1** is the labeled dataset for PPCI training. Each cage contains littermates from one of the three genetic lines; genotype (WT vs. HET) is recorded per observation. Genotypes are balanced (216 wt, 216 het). Genetic lines are balanced (144 each: ash1l, kdm6b, kmt5b).

**v2** cages also contain 3 wildtype and 1 heterozygous mouse, but genotype is not tracked per-mouse at the metadata level (all rows labeled `mixed`). The HET mouse is physically identifiable by a shaved patch in the middle of the back. Intended for unsupervised / ECI analysis once per-individual tracking is available.
