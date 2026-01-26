## Minimal Makefile (lean) ----------------------------------------
# Purpose: fast local reproducibility (env, notebooks, QA)
# This repo is notebook-first; scripts/modules are introduced during refactoring.

.PHONY: help env update nb_local nb_global test lint format clean

PYTHON?=python

help:
	@echo "Available targets (minimal project):" && echo && \
	printf "  %-12s %s\n" \
	"env" "Create conda env (idempotent)" \
	"update" "Update/prune existing env" \
	"nb_local" "Execute Act 1 notebook (local dynamics)" \
	"nb_global" "Execute Act 2 notebook (global scale)" \
	"test" "Run pytest suite" \
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

test:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests

format:
	black src tests

clean:
	rm -rf **/__pycache__ .pytest_cache .ruff_cache artifacts/notebook_exec.ipynb
	@echo "Cleaned caches & transient notebook execution artifact."

# End -----------------------------------------------------------------------
