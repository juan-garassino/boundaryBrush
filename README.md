# boundaryBrush

Promptable segmentation: a U-Net, a from-scratch mini-SAM and pretrained SlimSAM behind one predictor API.

> **Status: early scaffold (v0.1.0).** The package skeleton, config, CI and a full design spec are in place.
> The models, training, evaluation and CLI subcommands described below are **planned, not implemented yet**,
> and there are no results to report.

## What and why

boundaryBrush aims to be a small, CPU-trainable segmentation toolkit: click (or box) a pet in an image and get a
mask. Three interchangeable backends are planned so they can be compared on the same protocol: a plain U-Net, a
SAM-style mini model written from scratch, and the pretrained SlimSAM (`nielsr/slimsam-77-uniform`). It is
also intended as the segmentation engine for a later camera-to-projector project, hence the stable
`Segmenter` interface in the spec.

The full design (data pipeline, model architectures, eval protocol, phases) is in
[docs/superpowers/specs/2026-09-28-boundarybrush-v1-design.md](docs/superpowers/specs/2026-09-28-boundarybrush-v1-design.md).

## What exists today

- Installable package `boundarybrush` (hatchling, `src/` layout) with an argparse CLI shell (`--version`, `-v`;
  no subcommands registered yet).
- `config.py`: frozen dataclass of settings read from `BOUNDARYBRUSH_*` environment variables
  (data/runs/weights dirs, device, threads, image size, SAM model id, release URL).
- Two unit tests for the settings defaults and env overrides ([tests/test_config.py](tests/test_config.py)).
- Reproducible CPU-only setup: `uv.lock` pinned to torch 2.2.2 / torchvision 0.17.2 (CPU wheels on Linux);
  CI checks the lockfile contains no `nvidia` packages.
- CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)): ruff, tests, and a smoke import including
  `transformers.SamModel`, with Hugging Face offline flags set.

## Planned (from the design spec)

- Backends `unet`, `minisam`, `slimsam` behind `get_segmenter(...)`, `Segmenter.set_image(...)`,
  `Segmenter.predict(Prompt) -> MaskResult`.
- Oxford-IIIT Pet at 128x128, trained locally on CPU.
- Interactive evaluation: IoU, Boundary IoU, IoU at 1 click / box, IoU vs. number of clicks, NoC@85.
- CLI: `prepare-data`, `train`, `eval`, `predict`, `click`, `gallery`, `fetch-weights`.

The spec lists expected IoU ranges, but these are targets to be measured, not results.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12 (`.python-version` is 3.12). Supported
platforms per `pyproject.toml`: macOS and Linux x86_64.

```bash
make install        # uv sync --extra all (adds transformers for SlimSAM, pytest, ruff)
make test           # uv run pytest -q
make lint           # ruff check + format check
uv run boundarybrush --version
```

Not verified here: in the sandbox used to write this README the torch wheel could not be downloaded
(network restricted), so `uv sync` and `pytest` were not run end to end. The `Makefile` targets for data,
training, evaluation and the demo (`make data`, `train-unet`, `train-minisam`, `eval`, `demo`) call
subcommands that do not exist yet.

## Project structure

```
src/boundarybrush/
  cli.py          argparse entry point (no subcommands yet)
  config.py       BOUNDARYBRUSH_* settings dataclass
  domain/ models/ providers/   empty packages reserved for the planned design
tests/test_config.py
docs/superpowers/specs/       v1.0 design spec
Makefile  pyproject.toml  uv.lock  .github/workflows/ci.yml
```

## Tech stack

Python 3.12, PyTorch 2.2.2 (CPU), torchvision, NumPy 1.x, Pillow, Matplotlib; optional `transformers` for
SlimSAM; uv, hatchling, pytest, ruff, GitHub Actions.

## License

[MIT](LICENSE)
