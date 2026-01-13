# Dataset: Ant Triplet Grooming Experiment

This dataset contains videos and expert annotations from a controlled behavioral experiment on triplets of worker ants: one focal ant (randomly assigned to a treatment) and two nestmate ants. The goal is to estimate the causal effect of treatment on nestmate ants behavior, e.g., grooming directed toward the focal ant.

Each replicate consists of one focal ant and two nestmates that are distinguished by pen marking: one yellow and one blue.

---

## Experimental design

### Treatments and target estimand
- **Treatments:** `T ∈ {0, 1, 2}` (applied to the focal ant)
- **Goal:** estimate treatment effects on nestmate ants behaviors, e.g., grooming

### Replicates (videos)
- **Batches:** 5 *(a–e)*
- **Positions per batch:** 9 *(1–9)*
- **Planned observations:** 45 *(5 × 9)*
- **Excluded observations:** 1  
  - Batch: c  
  - Position: 9  
  - Reason: ant escaped
- **Analyzable observations:** 44

### Recording duration
- **Attempted recording per replicate:** 90 min  
- **Valid recording used per replicate:** 30 min

---

## Behavioral annotation

- **Annotated behavior:** grooming directed toward the focal ant:
  - **yellow → focal** grooming (binary, per frame)
  - **blue → focal** grooming (binary, per frame)
- **Temporal granularity:** up to two behavior changes per second

> Notes: All annotations were produced by a single domain expert, tracking up to two behavior changes per second.

---

## Biological metadata

| Field | Value |
|---|---|
| Ant season | Summer |
| Days | 145 |
| Species | Lneg |
| Population | Seva |
| Collection year | 2019 |
| Colony | 5 |

---

## Recording acquisition hardware

| Component | Specification |
|---|---|
| Camera | FLIR BFS-U3-120S4C-CS USB 3.1 Blackfly® S (Color) |
| Lens | Tamron M111FM25, 25mm fixed focal, f/1.8, C-mount |
| Bit depth | 8 |

---

## Final dataset (what you get)

- **44 videos total**
- **Treatment counts:** 14 / 15 / 15 for treatments `T = 0 / 1 / 2`
- **Duration:** 30 minutes per video
- **Annotations per video:** one behavior (*grooming*) annotated for both nestmates
  - yellow-to-focal grooming (binary, per frame)
  - blue-to-focal grooming (binary, per frame)

