## Minimal Makefile (lean) ----------------------------------------
# Purpose: fast local reproducibility (env, notebooks, QA)
# This repo is notebook-first; scripts/modules are introduced during refactoring.

.PHONY: help env update nb_local nb_global \
	run_local run_global \
	run_local_lgbm run_local_nbeatsx run_global_lgbm run_global_nbeatsx \
	show_local_lgbm show_local_nbeatsx show_global_lgbm show_global_nbeatsx \
	smoke test lint format clean download_m5 verify_m5 repair_m5

PYTHON?=python

help:
	@echo "Available targets (minimal project):" && echo && \
	printf "  %-12s %s\n" \
	"env" "Create conda env (idempotent)" \
	"update" "Update/prune existing env" \
	"nb_local" "Execute Act 1 notebook (local dynamics)" \
	"nb_global" "Execute Act 2 notebook (global scale)" \
	"run_local" "Terminal-only Act 1 repro (LightGBM + N-BEATSx)" \
	"run_global" "Terminal-only Act 2 repro (LightGBM + N-BEATSx)" \
	"run_local_lgbm" "Terminal-only local LightGBM repro" \
	"run_local_nbeatsx" "Terminal-only local N-BEATSx repro" \
	"run_global_lgbm" "Terminal-only global LightGBM repro" \
	"run_global_nbeatsx" "Terminal-only global N-BEATSx repro" \
	"show_local_lgbm" "Print metrics/timing for last local LightGBM run" \
	"show_local_nbeatsx" "Print metrics/timing for last local N-BEATSx run" \
	"show_global_lgbm" "Print metrics/timing for last global LightGBM run" \
	"show_global_nbeatsx" "Print metrics/timing for last global N-BEATSx run" \
	"smoke" "Fast terminal-only smoke run (reduced compute)" \
	"test" "Run pytest suite" \
	"download_m5" "Download M5 CSVs into data/ via Kaggle" \
	"verify_m5" "Verify required M5 CSVs exist and are real CSVs" \
	"repair_m5" "Repair data/: unzip *.zip and fix ZIP-as-CSV" \
	"lint" "Ruff static checks" \
	"format" "Black code format" \
	"clean" "Remove caches & transient artifacts";

env:
	conda env create -f environment.yml || echo "(env likely exists; use 'make update' to sync)"

update:
	conda env update -f environment.yml --prune

nb_local:
	@command -v jupyter >/dev/null 2>&1 || { echo 'jupyter not found (pip install jupyter)'; exit 1; }
	jupyter nbconvert --to notebook --execute Notebooks/01_local_dynamics.ipynb --output /tmp/01_local_dynamics.executed.ipynb

nb_global:
	@command -v jupyter >/dev/null 2>&1 || { echo 'jupyter not found (pip install jupyter)'; exit 1; }
	jupyter nbconvert --to notebook --execute Notebooks/02_global_scale.ipynb --output /tmp/02_global_scale.executed.ipynb

run_local:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/local.toml local-lgbm
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/local.toml local-nbeatsx

run_local_lgbm:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/local.toml local-lgbm

run_local_nbeatsx:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/local.toml local-nbeatsx

run_global:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/global.toml global-lgbm
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/global.toml global-nbeatsx

run_global_lgbm:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/global.toml global-lgbm

run_global_nbeatsx:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/global.toml global-nbeatsx

show_local_lgbm:
	@f=$$(ls -1t Artifacts/runs/*/metrics_local_lgbm.json 2>/dev/null | head -n 1); \
	if [ -z "$$f" ]; then echo "No local LightGBM metrics found under Artifacts/runs/*/metrics_local_lgbm.json"; exit 1; fi; \
	d=$$(dirname "$$f"); \
	echo "Run dir: $$d"; \
	echo "Metrics: $$f"; \
	cat "$$f"; \
	if [ -f "$$d/timing.json" ]; then echo "---"; echo "Timing: $$d/timing.json"; cat "$$d/timing.json"; fi; \
	if [ -f "$$d/run_meta.json" ]; then echo "---"; echo "Run meta: $$d/run_meta.json"; cat "$$d/run_meta.json"; fi

show_local_nbeatsx:
	@f=$$(ls -1t Artifacts/runs/*/metrics_local_nbeatsx.json 2>/dev/null | head -n 1); \
	if [ -z "$$f" ]; then echo "No local N-BEATSx metrics found under Artifacts/runs/*/metrics_local_nbeatsx.json"; exit 1; fi; \
	d=$$(dirname "$$f"); \
	echo "Run dir: $$d"; \
	echo "Metrics: $$f"; \
	cat "$$f"; \
	if [ -f "$$d/timing.json" ]; then echo "---"; echo "Timing: $$d/timing.json"; cat "$$d/timing.json"; fi; \
	if [ -f "$$d/run_meta.json" ]; then echo "---"; echo "Run meta: $$d/run_meta.json"; cat "$$d/run_meta.json"; fi

show_global_lgbm:
	@f=$$(ls -1t Artifacts/runs/*/metrics_global_lgbm.json 2>/dev/null | head -n 1); \
	if [ -z "$$f" ]; then echo "No global LightGBM metrics found under Artifacts/runs/*/metrics_global_lgbm.json"; exit 1; fi; \
	d=$$(dirname "$$f"); \
	echo "Run dir: $$d"; \
	echo "Metrics: $$f"; \
	cat "$$f"; \
	if [ -f "$$d/timing.json" ]; then echo "---"; echo "Timing: $$d/timing.json"; cat "$$d/timing.json"; fi; \
	if [ -f "$$d/run_meta.json" ]; then echo "---"; echo "Run meta: $$d/run_meta.json"; cat "$$d/run_meta.json"; fi

show_global_nbeatsx:
	@f=$$(ls -1t Artifacts/runs/*/metrics_global_nbeatsx.json 2>/dev/null | head -n 1); \
	if [ -z "$$f" ]; then echo "No global N-BEATSx metrics found under Artifacts/runs/*/metrics_global_nbeatsx.json"; exit 1; fi; \
	d=$$(dirname "$$f"); \
	echo "Run dir: $$d"; \
	echo "Metrics: $$f"; \
	cat "$$f"; \
	if [ -f "$$d/timing.json" ]; then echo "---"; echo "Timing: $$d/timing.json"; cat "$$d/timing.json"; fi; \
	if [ -f "$$d/run_meta.json" ]; then echo "---"; echo "Run meta: $$d/run_meta.json"; cat "$$d/run_meta.json"; fi

smoke:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/smoke.toml --run-name smoke_local_lgbm local-lgbm
	$(PYTHON) -m walmart_demand_forecasting.cli --config configs/smoke.toml --run-name smoke_local_nbeatsx local-nbeatsx

test:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pytest -q

download_m5:
	$(PYTHON) -m walmart_demand_forecasting.datasets.download_m5 --output data/

verify_m5:
	$(PYTHON) -m walmart_demand_forecasting.datasets.download_m5 --output data/ --verify

repair_m5:
	$(PYTHON) -m walmart_demand_forecasting.datasets.download_m5 --output data/ --repair

lint:
	ruff check src tests

format:
	black src tests

clean:
	rm -rf **/__pycache__ .pytest_cache .ruff_cache artifacts/notebook_exec.ipynb
	@echo "Cleaned caches & transient notebook execution artifact."

# End -----------------------------------------------------------------------
