# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.0] - 2025-11-15
### Added
- `CHANGELOG.md` following Keep a Changelog format.

### Notes
This is the repo initialization prior to establishing a reliable baseline, deep learning challenging model.

## [Unreleased]
### Added
- Notebook-aligned CLI reproductions for all four models (local/global × LightGBM/N-BEATSx).
- Consistent evaluation contract: horizon=28, RMSE everywhere; item-level WRMSSE computed for global runs only.
- Per-run artifact logging under `Artifacts/runs/<run_name_or_timestamp>/` including `metrics_*.json`, `timing.json`, and `run_meta.json`.
- Makefile targets for terminal reproducibility:
	- `run_local_lgbm`, `run_local_nbeatsx`, `run_global_lgbm`, `run_global_nbeatsx`
	- `show_local_lgbm`, `show_local_nbeatsx`, `show_global_lgbm`, `show_global_nbeatsx`

### Changed
- Project README now includes recruiter-friendly reproduction commands (CLI + Make) and artifact inspection guidance.


[0.0.0]: https://github.com/Zak-Las/Walmart-Demand-Forcasting # repo initialization