# `ants/v2`
### Investigating Social Immunity in ant triplets experiment

This dataset contains videos and expert annotations from a controlled behavioral experiment on triplets of worker ants: one focal ant (randomly assigned to a treatment) and two nestmate ants. The goal is to estimate the causal effect of treatment on nestmate ants behavior, e.g., grooming directed toward the focal ant. Each replicate consists of one focal ant and two nestmates that are distinguished by pen marking: one yellow and one blue.

---

## Summary

- **Randomized Controlled Trial**
- **Observations:** 44 videos total (**10min valid each**)
  - 20 with treatment `T = 1`
  - 24 with treatment `T = 2`
- **Annotations (per video):** one behavior (*grooming*) annotated per frame for both nestmates
  - yellow-to-focal grooming (binary)
  - blue-to-focal grooming (binary)
- **Objective:** estimate treatment effect on nestmate ants behaviors, e.g., grooming
- **Date of recording:** April 18–19, 2024

---

## Experimental design

### Replicates
- **Batches:** 5 *(a–e)*
- **Positions per batch:** 9 *(1–9)*
- **Planned observations:** 45 *(5 × 9)*
- **Excluded observations:** 1 *(batch **b**, position **3**; invalid)*
- **Analyzable observations:** 44
- **Treatments:** 2 *(1, 2)* (randomly assigned)

### Behavioral annotation
- **Annotated behavior:** grooming directed toward the focal ant:
  - yellow → focal grooming (binary, per frame)
  - blue → focal grooming (binary, per frame)

> Notes: *The annotations were produced by 3 domain experts, tracking behavior changes frame by frame.*

### Recording
- **Attempted recording per replicate:** 20min  
- **Valid recording used per replicate:** 10min
- **Recording acquisition hardware (NEW):**
    | Component | Specification |
    |---|---|
    | Camera | FLIR Blackfly S BFS-U3-120S4C |
    | Lens | Edmund Optics 25mm HP series lens, 1inch (P000001501232-24068) |

### Biological metadata (ants)

| Field | Value |
|---|---|
| Ant season | Unknown |
| Days | Unknown |
| Species | Lneg |
| Population | Jena |
| Collection year | 2022 |
| Colony | Unknown |
