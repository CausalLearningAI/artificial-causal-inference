# `ants/v1`
### Investigating Social Immunity in ant triplets experiment

This dataset contains videos and expert annotations from a controlled behavioral experiment on triplets of worker ants: one focal ant (randomly assigned to a treatment) and two nestmate ants. The goal is to estimate the causal effect of treatment on nestmate ants behavior, e.g., grooming directed toward the focal ant.Each replicate consists of one focal ant and two nestmates that are distinguished by pen marking: one yellow and one blue. 

---

## Summary

- **Randomized Controlled Trial** 
- **Observations**: 44 videos total (30min each)
  - 15 with treatment `T = 0`
  - 14 with treatment `T = 1`
  - 15 with treatment `T = 2`
- **Annotations (per video):** one behavior (*grooming*) annotated per frame for both nestmates
  - yellow-to-focal grooming (binary)
  - blue-to-focal grooming (binary)
- **Objective:** estimate treatment effect on nestmate ants behaviors, e.g., grooming
- **Date of recording:** October 5th, 2023

---

## Experimental design

### Replicates 
- **Batches:** 5 *(a–e)*
- **Positions per batch:** 9 *(1–9)*
- **Planned observations:** 45 *(5 × 9)*
- **Excluded observations:** 1  (batch c, position 9; ant escaped)
- **Analyzable observations:** 44
- **Treatments**: 3 *(0, 1, 2)* (randomly assigned, balanced)

### Behavioral annotation
- **Annotated behavior:** grooming directed toward the focal ant:
  - yellow → focal grooming (binary, per frame)
  - blue → focal grooming (binary, per frame)

> Notes: *All annotations were produced by a single domain expert, tracking up to two behavior changes per second.*

### Recording 
- **Attempted recording per replicate:** 90min  
- **Valid recording used per replicate:** 30min
- **Recording acquisition hardware:**
    | Component | Specification |
    |---|---|
    | Camera | FLIR BFS-U3-120S4C-CS USB 3.1 Blackfly® S (Color) |
    | Lens | Tamron M111FM25, 25mm fixed focal, f/1.8, C-mount |
    | Bit depth | 8 |

### Biological metadata (ants)

| Field | Value |
|---|---|
| Ant season | Summer |
| Days | 145 |
| Species | Lneg |
| Population | Seva |
| Collection year | 2019 |
| Colony | 5 |

---

## Citation

```bibtex
@inproceedings{cadei2025prediction,
  title={Prediction-Powered Causal Inferences},
  author={Cadei, Riccardo and Demirel, Ilker and De Bartolomeis, Piersilvio and Lindorfer, Lukas and Cremer, Sylvia and Schmid, Cordelia and Locatello, Francesco},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

