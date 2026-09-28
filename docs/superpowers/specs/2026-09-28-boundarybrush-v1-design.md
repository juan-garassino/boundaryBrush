# Part A — boundaryBrush v1.0 — promptable segmentation (U-Net + mini-SAM + SlimSAM)

## Context

`005-products/004-creative-tools/002-boundaryBrush` is an empty skeleton (every file 0 bytes: SageMaker/
Streamlit/CloudFormation stubs, not a git repo). Juan wants it at a portfolio-grade v1.0 as a **working
prototype**: "like a SAM model, and also a U-Net on an easy dataset", trained **locally**, with **minimal
dependencies**. It is also the segmentation engine for the next project, **illusionFrame** (camera →
segment → generative mutation → projector loop, e.g. projection-mapping onto one of Juan's paintings), so
it must expose a stable predictor-style API.

Decisions already made with Juan: portfolio-grade (no hosting), local CPU training, Oxford-IIIT Pet,
**three backends behind one interface** (U-Net, from-scratch mini-SAM, pretrained SlimSAM), cloud
scaffold deleted. dataPalette (→ PyPI), illusionFrame, dataDiffusion follow later, each with its own plan.

Hardware reality: 2014 MBP, i7-4870HQ 4c/8t, 16 GB, no torch GPU → **CPU only, torch 2.2.2 cap,
numpy<2**. Training runs are hours (overnight), not minutes; IoU targets are expectations to measure
and report honestly, not pass/fail gates.

## What v1.0 delivers

```
boundarybrush prepare-data                       # download Pets, cache 128px uint8 tensors
boundarybrush train unet|minisam [--epochs --resume --limit]
boundarybrush eval unet|minisam|slimsam [--limit]  # IoU, Boundary IoU, IoU@1click, IoU@box, IoU-vs-clicks, NoC@85
boundarybrush predict IMG --backend B --click 120,80 [--click 30,40:0] [--box x0,y0,x1,y1] -o out/
boundarybrush click IMG --backend B              # matplotlib: left=+, right/`n`=−, r reset, s save
boundarybrush gallery                            # README comparison figure
boundarybrush fetch-weights                      # release assets → ~/.cache/boundarybrush, SHA-256 checked
```

Public API for illusionFrame: `get_segmenter(Backend, weights=None)`, `Segmenter.set_image(rgb uint8 HxWx3)`
(caches embedding), `Segmenter.predict(Prompt) -> MaskResult(mask: bool HxW original size, score: float)`.

## Layout (template: `005-products/024-dino`)

```
002-boundaryBrush/
  src/boundarybrush/
    __init__.py  __main__.py  cli.py  config.py        # config = dataclass + env vars (BOUNDARYBRUSH_*)
    domain/enums.py   Backend(StrEnum): unet, minisam, slimsam, mock
    domain/models.py  Prompt, MaskResult (dataclasses, validated in __post_init__)
    ops.py            zero-padded erode/dilate, deepest_point, component_at (max-pool flood fill),
                      shared resize, logits_to_mask (upsample logits, then threshold)
    metrics.py        per-image IoU, Boundary IoU (d = round(0.02·diag)), NoC@85 (max 5)
    data.py           injectable source → cached trimap tensors, seeded 10% val split,
                      joint v2 aug (tv_tensors.Mask), target/weights derived after aug, copy-paste compositing
    prompts.py        training sampler + tensor encoding (pad label −1)
    models/unet.py    4 levels, base 16, 2 convs/block, logits out
    models/minisam.py SAM-style, see below
    train.py checkpoint.py   AdamW+cosine, metrics.jsonl, best/last (atomic), --resume, --limit
    evaluate.py       interactive protocol written once against the Segmenter ABC
    providers/base.py registry.py mock.py unet_provider.py minisam_provider.py slimsam_provider.py
    viz.py            overlay, gallery, click app
    weights.py        fetch + SHA-256 manifest (weights_only=True loads)
  tests/  conftest.py + one test_<unit>.py per module (offline, < ~60 s)
  results/ (eval JSON + figures, committed)   docs/images/
  pyproject.toml  uv.lock  .python-version(3.12)  Makefile  .github/workflows/ci.yml
  README.md  CLAUDE.md  CHANGELOG.md  LICENSE(MIT)  .gitignore (data/, runs/)
```

## Key design decisions

**Data (Oxford-IIIT Pet, 128×128, squashed — documented).** Cache the *trimap* (1 pet / 2 bg / 3 border)
as uint8; derive after augmentation: target = trimap≠2, loss weight 0.5 on the border band (it is the
"undefined" band — upweighting it would fatten masks). Same resize function in data prep and every
provider's `set_image`. `prepare-data` accepts an already-populated dir (Oxford host is flaky).

**U-Net.** Plain 4-level, base 16 (~1.9M params), BCEWithLogits(reduction="none") × weights + per-sample
Dice. Provider: click → connected component at 128px (max-pool flood fill, logits outside set to −inf, then
upsample); negative click removes its component; box intersects; no prompt → full mask; positive click on
background → empty mask, score 0. Score = mean fg probability in the selected region.

**Mini-SAM** — port `009-mini-networks/src/mini_networks/models/sam/model.py`, with these fixes:
- 3-channel input, **stride-8** residual CNN encoder → 16×16×128, + 1–2 self-attention layers over the 256
  image tokens (global context), + one stride-2 skip (64², ~16 ch) fused after the decoder (SAM-HQ style).
- **One shared frozen Fourier PE** (`PositionEmbeddingRandom`) for grid cells and click coords, re-added
  to q/k in every attention layer; token self-attention added to `TwoWayBlock`.
- Explicit pad type + `key_padding_mask` (009 turns padding into positive clicks at (0,0)); coordinates
  are **(x, y)**, pixel-centred `(idx+0.5)/W`, each axis normalized by its own size.
- `n_masks=1` + IoU head (config option kept; Pets has no part/whole ambiguity so 3 heads would collapse).
  Loss = weighted BCE + Dice + MSE(IoU head, detached actual IoU).
- Training prompts cover the eval distribution: 1–5 clicks any ±-mix, ~half the negatives from a dilated
  ring round the object, jittered box, box+clicks p≈0.15; K=2–4 prompt sets per image sharing one encoder
  pass.
- **Copy-paste compositing** (p=0.5): paste a scaled pet from another sample; the click chooses the
  target (target adjusted for occlusion). This is what makes it actually promptable on a one-pet dataset;
  reported via a "wrong-prompt IoU" diagnostic.

**SlimSAM** (`nielsr/slimsam-77-uniform`, 38.9 MB, optional extra `[sam]`, `transformers>=4.49,<5`).
Model/processor injected (lazy import; tests use stubs). `set_image` runs the processor once, caches
`get_image_embeddings()` + `original_sizes` + `reshaped_input_sizes` under `inference_mode`; `predict`
scales (x, y) itself (longest side → 1024), shapes `input_points (1,1,N,2)`, `input_labels (1,1,N)`,
`input_boxes (1,1,4)`; `multimask_output=True` only for exactly one click and no box (argmax
`iou_scores`); `post_process_masks` → original size. `use_safetensors=True` (torch<2.6 refuses .bin).

**Eval protocol** (same 128px test tensors for all backends; SlimSAM therefore sees upscaled input,
a lower bound — documented). Click 1 = deepest fg point; next click = argmax erosion depth over FN vs FP
maps (FN → +, FP → −), image border counted as boundary (zero padding), deterministic ties. Metrics:
mean per-image IoU, Boundary IoU, IoU@1click, IoU@box (tight GT box), IoU-vs-clicks 1..5, NoC@85 (max 5).
SlimSAM default `--limit 500`.

**Expected results** (report actual numbers): U-Net IoU ≈ 0.80; mini-SAM ≈ 0.75 @1 click, ≈ 0.82 @box;
SlimSAM = reference.

**Training on this CPU.** `torch.set_num_threads(4)`, `num_workers=0` (cached tensors), device default
**cpu** (don't copy dino's auto → MPS), runs wrapped in `caffeinate -i`, sequential not parallel. Rough
cost: U-Net 4–8 min/epoch × 30–50 epochs; mini-SAM overnight. Benchmark one real epoch before launching.

## Dependencies & tooling

```toml
requires-python = ">=3.11,<3.13"
dependencies = ["torch==2.2.2", "torchvision==0.17.2", "numpy>=1.24,<2", "pillow>=10", "matplotlib>=3.8"]
[project.optional-dependencies] sam = ["transformers>=4.49,<5"]; dev = ["pytest>=7.2", "ruff"]
[project.scripts] boundarybrush = "boundarybrush.cli:main"
[tool.uv] environments = ["sys_platform == 'darwin'", "sys_platform == 'linux' and platform_machine == 'x86_64'"]
[[tool.uv.index]] name = "pytorch-cpu", url = "https://download.pytorch.org/whl/cpu", explicit = true
[tool.uv.sources] torch/torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
[tool.pytest.ini_options] pythonpath = ["src"], testpaths = ["tests"]
```

- Makefile: awk `help` (from `025-ltm/Makefile`), `install`, `test`, `test-ci`, `lint`, `data`,
  `train-unet`, `train-minisam`, `eval`, `demo`, `clean`.
- CI (`.github/workflows/ci.yml`, from 024-dino + 025-ltm): setup-uv@v3 (cache on), `uv sync --extra dev`,
  `! grep -q nvidia uv.lock`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, pytest, ruff; smoke-import job
  with `--extra sam` importing every module + `transformers.SamModel`.

**Deviations from root CLAUDE.md (deliberate, per "remove dependencies"):** dataclass config instead of
pydantic-settings; local `metrics.jsonl` instead of the shared MLflow tracker; no Dockerfile / run / deploy
targets (library, not a service); no deploy/rollback workflows.

## Reuse

- `009-mini-networks/src/mini_networks/models/sam/{model,trainer}.py`: MiniSAM structure, loss shape, and
  the "wrong-prompt IoU" diagnostic. **Don't copy:** the zero-padding bug, `(1 - types)` label mapping,
  winner-only IoU supervision, (y, x) coordinates, no pixel-centring, 1-channel/hard-coded grid,
  `interior_click`.
- `009-mini-networks/.../segmentation/unet.py`: block shape only (drop in-forward sigmoid, BCELoss,
  batch-flattened dice, Dropout2d).
- `024-dino`: `cli.py` build_parser + lazy imports, `train/checkpoint.py` atomic save, `train/loop.py`
  MetricsWriter, `tests/test_train_smoke.py` pattern. Not: split torch pins, auto→MPS,
  pydantic-settings, `keep_every` pruning.
- `025-ltm`: Makefile help, CI offline flags.

## Implementation phases (TDD; each phase ends with a commit on `feat/v1`)

0. **Scaffold + risk spikes.** Verify all old files are 0 bytes, then delete them; `git init` (scaffold +
   this plan copied to `docs/superpowers/specs/2026-09-28-boundarybrush-v1-design.md` on `main`); branch
   `feat/v1`. pyproject/Makefile/CI/config. Spikes: `uv lock` has no nvidia; `--extra sam` imports
   SamModel on torch 2.2.2; time one SlimSAM embedding. **Start the Pets download in the background.**
1. **Pure logic.** `domain/`, `ops.py`, `metrics.py`. Tests first: Prompt validation, erosion on a
   border-touching square / ring / two blobs, IoU/Boundary IoU on hand-made masks.
2. **Data + prompts.** Synthetic 4-sample source in tests: cache shapes/dtypes, split determinism,
   binary targets after aug, band weights, compositing; sampler invariants (+ on fg, − on bg, box
   contains object, padding). Then run the real `prepare-data`.
3. **Models.** Shapes; adding a pad prompt leaves output unchanged; moving a click changes it; tiny
   32px overfit (IoU > 0.9, < 5 s).
4. **Training.** Synthetic 2-epoch smoke: best/last written, finite loss, reload, resume. Benchmark
   one real epoch each → set epochs → **launch U-Net training in background**.
5. **Providers.** Stub nets / stub SamModel+SamProcessor recording `self.calls`: component selection,
   negative removal, box, original-size output; coord mapping; SlimSAM embeds once across 3 predicts,
   coord scaling, multimask rule. After U-Net finishes → **launch mini-SAM overnight**.
6. **Eval.** Oracle segmenter → IoU 1, NoC 1; empty segmenter → NoC max; deterministic click sequence.
   Run U-Net eval; **SlimSAM `--limit 500` in background**.
7. **CLI + viz + weights.** Parser round-trip; `predict --backend mock` writes PNGs; `fetch-weights` via
   `file://` URL passes on matching SHA, fails on mismatch. Makefile targets.
8. **Results + docs.** Mini-SAM eval, gallery, IoU-vs-clicks plot, `results/*.json`, README (results
   table, figures, architecture, limitations: 128px masks look soft at projector resolution; Pets images
   CC BY-SA — credit any bundled sample or use Juan's own photo), CLAUDE.md, CHANGELOG, version 1.0.0.
9. **Release — ask Juan first (outward-facing):** create public GitHub repo `juan-garassino/boundaryBrush`,
   push with the personal SSH key (`id_ed25519_personal`), merge `feat/v1` → `main`, tag `v1.0.0`,
   GitHub release with `unet.pt`, `minisam.pt`, SHA manifest, eval JSON.

## Verification

- `make test` green (offline, < ~60 s); CI green on GitHub (after step 9).
- `uv lock` contains no `nvidia`; `uv run --extra sam python -c "from transformers import SamModel"` works.
- `boundarybrush eval unet|minisam|slimsam` writes `results/*.json`; numbers go into the README table.
- `boundarybrush predict docs/images/sample.jpg --backend minisam --click …` and `--backend slimsam` produce
  sensible overlays; `boundarybrush click` works interactively (fallback `MPLBACKEND=TkAgg`).
- Fresh clone: `uv sync && boundarybrush fetch-weights && boundarybrush predict …` works end-to-end.

## Docs to update

- `002-boundaryBrush/README.md`, `CLAUDE.md`, `CHANGELOG.md`: new (module list per workspace convention).
- `005-products/DOCS.md` § 004-CreativeTools: BoundaryBrush line → "promptable segmentation: U-Net +
  from-scratch mini-SAM + SlimSAM behind one predictor API (v1.0)".
- Flag only (separate repo, don't edit): `003-carreer-navigator/docs/portfolio_catalog.md:141` lists
  boundaryBrush under "Skip (empty skeleton)".

---
