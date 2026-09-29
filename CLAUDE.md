# boundaryBrush

Promptable segmentation: a U-Net, a from-scratch mini-SAM and pretrained SlimSAM behind one
predictor API, trained locally on Oxford-IIIT Pet. It is also the segmentation engine for
`../010-illusionFrame` (its `boundarybrush` mask kind).

## Commands

- `make install` (uv, all extras) · `make test` (`make test-ci` skips `slow`) · `make lint`
- `make data` (download Pets + build `data/cache/pets_{trainval,test}_128.pt`)
- `make train-unet EPOCHS=30` / `make train-minisam` (resumable: `--resume` continues `runs/<model>/last.pt`)
- `uv run boundarybrush eval unet|minisam|slimsam [--limit N]` -> `results/eval_<backend>.json`
- `uv run boundarybrush predict IMG -b minisam --click x,y[:0] --box x0,y0,x1,y1 -o out/`
- `uv run boundarybrush click IMG -b slimsam` (matplotlib; `--mpl-backend TkAgg` if focus misbehaves)
- `uv run boundarybrush gallery` · `boundarybrush export runs/*/best.pt --dest release --manifest ...`

## Public API (keep stable — illusionFrame imports it)

`from boundarybrush import Prompt, MaskResult, Backend, Segmenter, get_segmenter`;
`seg.set_image(rgb_uint8)` embeds once, `seg.predict(Prompt(points, labels, box)) -> MaskResult(mask, score)`.
Coordinates are continuous pixel (x, y); pixel (r, c) is clicked at (c + 0.5, r + 0.5).

## Module map

- `domain/` `Backend` StrEnum; `Prompt` (validated), `MaskResult`
- `ops.py` zero-padded max-pool morphology, `deepest_point`, `component_at`, the one shared resize
  (`resize_rgb` / `resize_labels`), `to_model_input`, `logits_to_mask`
- `metrics.py` per-image IoU, Boundary IoU (d = 2% of the diagonal), NoC@85
- `data.py` `OxfordPetSource`, tensor cache (trimaps, not masks), seeded split, joint augmentation,
  `paste` compositing, `PetsDataset`, `PromptedPetsDataset` (K prompt sets per image)
- `prompts.py` token types (PAD/POS/NEG/BOX_TL/BOX_BR), training sampler, `encode_prompt`
- `models/unet.py`, `models/minisam.py` (stride-8 encoder + skip, shared Fourier PE, two-way decoder),
  `models/losses.py` (weighted BCE + Dice; mini-SAM min-loss + IoU head on every head)
- `train.py` AdamW + warmup-cosine, JSONL metrics, best/last checkpoints, resume;
  `checkpoint.py` self-describing checkpoints (`weights_only=True` loads)
- `providers/` `Segmenter` ABC, `unet_provider` (clicks pick connected components), `minisam_provider`,
  `slimsam_provider` (transformers; embeds once, scales coords itself), `mock`, `registry.get_segmenter`
- `evaluate.py` click protocol (deepest point, then larger error region) + box + automatic IoU
- `viz.py` overlay, gallery, click app · `weights.py` fetch + SHA-256 manifest · `cli.py`

## Constraints

This machine: 2014 Intel MBP, CPU only — torch 2.2.2, numpy < 2, `BOUNDARYBRUSH_DEVICE=cpu`
(never auto: MPS may claim the Iris Pro). Tests pin torch to one thread (tests/conftest.py).
Oxford-IIIT Pet images are CC BY-SA 4.0 with owner copyright — credit any image shown.
Deliberate deviations from the workspace conventions: dataclass config (no pydantic-settings),
local JSONL metrics (no MLflow), no Dockerfile or deploy workflows (library, not a service).
