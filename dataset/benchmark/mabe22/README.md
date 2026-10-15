# MABe22 (external benchmark, NOT one of our lab experiments)

External public benchmark: Sun et al., "MABe22: A Multi-Species Multi-Task Benchmark for Learned Representations of Behavior", ICML 2023, arXiv 2207.10553.
Used only as a blind development benchmark for the unsupervised video representation. Keep separate from dataset/{ants,mice,frogs}.

- Source: CaltechDATA record https://data.caltech.edu/records/rdsa8-rde65 (published 2023-05-31); project page https://sites.google.com/view/computational-behavior/our-datasets/mabe2022-dataset
- License: the record's API metadata has no `rights` field; the record page says "Dataset Open" and links a CaltechDATA Terms page (not read). The paper/arXiv page does not state a data license; request is to cite arXiv 2207.10553. Treat as: citation required, license unconfirmed.
- Downloaded: 2026-10-15 (Slurm job 828077). Every file's size and md5 matched the record.
- raw/: label/keypoint/frame-map .npy files (mouse, beetle; fly labels + frame map only) and record_files.json. The video zips were deleted after extraction was verified.
- videos/{mouse_userTrain,mouse_submission,beetle_userTrain,beetle_submission}/<clip_id>.avi : 224x224 mpeg4 AVI, 30 fps (the only resolution released; no 800x800 originals found).
- videos/crc_failed/: 4 clips whose zip CRC-32 does not match their bytes (upstream defect); excluded from the sets above.
- Inventory: results/vision/benchmark_mabe22/inventory/inventory.md ; scripts: scripts/benchmark/mabe22/
