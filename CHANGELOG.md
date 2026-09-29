# Changelog

## [1.0.0] — unreleased

First release (the repository was an empty SageMaker/Streamlit scaffold before).

- One predictor API (`set_image` once, `predict(Prompt)` many times) over four backends: U-Net,
  from-scratch mini-SAM, pretrained SlimSAM (optional `[sam]` extra) and a mock.
- Local CPU training on Oxford-IIIT Pet with a cached 128px dataset, joint augmentation, copy-paste
  compositing for promptability, resumable checkpoints.
- Deterministic click-based evaluation (IoU at 1..5 clicks, NoC@85, box IoU, Boundary IoU).
- CLI: prepare-data, train, eval, predict, click, gallery, fetch-weights, export.
