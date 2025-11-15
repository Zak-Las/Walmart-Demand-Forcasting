## Minimal Makefile (lean) ----------------------------------------
# Purpose: fast local reproducibility (env, data prep, train, experiment, backtest, QA)
# Explicitly excludes heavy deployment / infra steps to keep portfolio scope tight.

.PHONY: help env update download prepare train experiment backtest test lint format clean

PYTHON?=python
RAW_DIR?=data/raw/m5
PROCESSED_DIR?=data/processed/m5_panel
PANEL?=data/processed/m5_panel_subset.parquet
EPOCHS?=10
MAX_ITEMS?=50
INPUT_LENGTH?=112
FORECAST_LENGTH?=30

help:
	@echo "Available targets (minimal project):" && echo && \
	printf "  %-12s %s\n" \
	"env" "Create conda env (idempotent)" \
	"update" "Update/prune existing env" \
	"download" "Download raw M5 dataset (Kaggle)" \
	"prepare" "Prepare long panel parquet partitions" \
	"train" "Train N-BEATS script run" \
	"experiment" "Execute notebook (includes feature experiment)" \
	"backtest" "Mini rolling-origin backtest vs seasonal naive" \
	"test" "Run pytest suite" \
	"lint" "Ruff static checks" \
	"format" "Black code format" \
	"clean" "Remove caches & transient artifacts";

env:
	conda env create -f environment.yml || echo "(env likely exists; use 'make update' to sync)"

update:
	conda env update -f environment.yml --prune

download:
	@command -v kaggle >/dev/null 2>&1 || { echo 'kaggle CLI missing. Inside env: pip install kaggle'; exit 1; }
	$(PYTHON) -m src.data.download_m5 --output $(RAW_DIR)

prepare:
	$(PYTHON) -m src.data.prepare_m5 --raw-dir $(RAW_DIR) --out-dir $(PROCESSED_DIR)

train:
	$(PYTHON) scripts/train_nbeats.py --panel $(PANEL) --epochs $(EPOCHS) --input-length $(INPUT_LENGTH) --forecast-length $(FORECAST_LENGTH) --max-items $(MAX_ITEMS)

experiment:
	@command -v jupyter >/dev/null 2>&1 || { echo 'jupyter not found (pip install jupyter)'; exit 1; }
	jupyter nbconvert --to notebook --execute notebooks/nbeats_training.ipynb --output artifacts/notebook_exec.ipynb

backtest:
	$(PYTHON) -m src.evaluation.backtest --panel $(PANEL) --checkpoint artifacts/models/nbeats_0.1.0.ckpt --horizon $(FORECAST_LENGTH) --stride 7 --windows 6 --max-items $(MAX_ITEMS) --input-length $(INPUT_LENGTH)

test:
	pytest -q

lint:
	ruff check src tests

format:
	black src tests

clean:
	rm -rf **/__pycache__ .pytest_cache .ruff_cache artifacts/notebook_exec.ipynb
	@echo "Cleaned caches & transient notebook execution artifact."

# End -----------------------------------------------------------------------
