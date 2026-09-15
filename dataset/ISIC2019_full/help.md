# ISIC 2019 — full local dataset (git-ignored)

This folder holds the downloaded ISIC 2019 data, organized by class:

```text
dataset/isic2019_full/
  AK/   BCC/  BKL/  DF/   MEL/  NV/   SCC/  VASC/  # *.jpg images (25,331)
  ISIC_2019_Training_GroundTruth.csv  # 25,331 rows + header
  ISIC_2019_Training_Metadata.csv
```

Everything in this folder except this README is git-ignored (see
`.gitignore`), so the multi-GB download never gets committed.

Note: the 24 prompt records in `prompt.yaml` point at the small tracked
sample under `dataset/ISIC2019/` — that folder is untouched by this change.
