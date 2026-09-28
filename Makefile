.PHONY: help install test test-ci lint data train-unet train-minisam eval demo clean

EPOCHS ?= 40

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install with all extras (uv)
	uv sync --extra all

test: ## Run the test suite
	uv run pytest -q

test-ci: ## Test gate for CI (deselect known failures here, never skip silently)
	uv run pytest -q -m "not slow"

lint: ## Ruff lint + format check
	uv run ruff check . && uv run ruff format --check .

data: ## Download Oxford-IIIT Pet and build the 128px cache
	uv run boundarybrush prepare-data

train-unet: ## Train the U-Net (make train-unet EPOCHS=40)
	caffeinate -i uv run boundarybrush train unet --epochs $(EPOCHS)

train-minisam: ## Train the mini-SAM (make train-minisam EPOCHS=40)
	caffeinate -i uv run boundarybrush train minisam --epochs $(EPOCHS)

eval: ## Evaluate a backend: make eval BACKEND=unet
	@if [ -z "$(BACKEND)" ]; then echo "Usage: make eval BACKEND=unet|minisam|slimsam"; exit 1; fi
	uv run boundarybrush eval $(BACKEND)

demo: ## Interactive click-to-segment on the sample image
	uv run boundarybrush click docs/images/sample.jpg --backend $(or $(BACKEND),minisam)

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
