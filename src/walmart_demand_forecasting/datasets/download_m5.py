"""Download and stage the M5 Forecasting dataset.

This script downloads the M5 competition data from Kaggle using the Kaggle API.
Requirements:
    1. Install kaggle: `pip install kaggle` inside your conda environment.
    2. Obtain API token from Kaggle account settings and save as `~/.kaggle/kaggle.json` with permissions 600.
    3. Accept competition rules on Kaggle site (M5 Forecasting Accuracy).

Usage:
    # Default: stage required CSVs into the repo-root data/ folder
    python -m walmart_demand_forecasting.datasets.download_m5

    # Custom output directory
    python -m walmart_demand_forecasting.datasets.download_m5 --output data/

    # Download full archive (fallback)
    python -m walmart_demand_forecasting.datasets.download_m5 --output data/ --full

    # Verify presence only
    python -m walmart_demand_forecasting.datasets.download_m5 --output data/ --verify

Creates (default):
 data/
     calendar.csv
     sales_train_evaluation.csv
     sell_prices.csv
     sales_train_validation.csv  (optional)
     sample_submission.csv  (optional)

If Instacart is later added, place its raw files under data/raw/instacart/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

REQUIRED_FILES = [
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
]
OPTIONAL_FILES = ["sales_train_validation.csv", "sample_submission.csv"]
COMPETITION = "m5-forecasting-accuracy"


ZIP_MAGIC_PREFIXES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def _is_zip_bytes_prefix(path: Path) -> bool:
    try:
        head = path.open("rb").read(4)
    except OSError:
        return False
    return any(head.startswith(prefix) for prefix in ZIP_MAGIC_PREFIXES)


def _is_zip_file(path: Path) -> bool:
    # zipfile.is_zipfile does its own probing; the prefix check is a fast-path.
    return _is_zip_bytes_prefix(path) or zipfile.is_zipfile(path)


def _unique_backup_path(path: Path, suffix: str = ".zip") -> Path:
    candidate = path.with_name(path.name + suffix)
    if not candidate.exists():
        return candidate
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.name}{suffix}.{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate backup filename for {path}")


def _materialize_zipped_csv(path: Path, *, keep_backup: bool = True) -> bool:
    """If `path` is a ZIP archive (even if named .csv), replace it with the embedded CSV.

    Returns True if a repair was performed.
    """

    if not path.exists() or not _is_zip_file(path):
        return False

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        csv_members = [n for n in names if n.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"{path} is a ZIP but contains no .csv members: {names[:5]}")

        # Prefer an exact basename match (Kaggle typically zips the same filename).
        preferred = path.name
        member = preferred if preferred in csv_members else (csv_members[0] if len(csv_members) == 1 else None)
        if member is None:
            raise RuntimeError(
                f"{path} contains multiple CSV members; can't choose safely: {csv_members[:10]}"
            )

        tmp_path = path.with_name(path.name + ".tmp")
        with zf.open(member, "r") as src, tmp_path.open("wb") as dst:
            dst.write(src.read())

    if keep_backup:
        backup = _unique_backup_path(path, suffix=".zip")
        path.replace(backup)
    else:
        path.unlink(missing_ok=True)

    tmp_path.replace(path)
    return True


def repair_m5_dir(output_dir: Path, required_files: Iterable[str] = REQUIRED_FILES) -> None:
    """Ensure required files under output_dir are real CSVs, not ZIP archives."""

    # First, unzip any normal *.zip downloads.
    for z in output_dir.glob("*.zip"):
        print("Unzipping", z.name)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(output_dir)
        z.unlink(missing_ok=True)

    # Then, detect/fix the common failure mode: a ZIP file saved with a .csv name.
    for fname in required_files:
        candidate = output_dir / fname
        if not candidate.exists():
            continue
        if _materialize_zipped_csv(candidate, keep_backup=False):
            print(f"Repaired ZIP-disguised CSV: {fname}")


def default_output_dir() -> Path:
    """Default to the repo-root `data/` folder.

    Works whether invoked from repo root or elsewhere.
    """

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "data"
    return Path("data")


def have_kaggle() -> bool:
    """Return True if kaggle CLI is available."""
    from shutil import which
    return which("kaggle") is not None


def ensure_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    cred = kaggle_dir / "kaggle.json"
    if not cred.exists():
        raise SystemExit(
            "kaggle.json not found at ~/.kaggle/kaggle.json.\n"
            "Download your API token from https://www.kaggle.com/<username>/account and place it there, then: chmod 600 ~/.kaggle/kaggle.json\n"
            "Also ensure you have accepted the competition rules on Kaggle."
        )
    # Fix permissions if needed (ignore errors on non-POSIX FS)
    try:
        os.chmod(cred, 0o600)
    except Exception:
        pass


def download_m5(output_dir: Path, force: bool = False, full: bool = False) -> None:
    if not have_kaggle():
        raise SystemExit("kaggle CLI not found. Install with 'pip install kaggle' inside your env.")
    ensure_credentials()

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in output_dir.glob("*.csv")}

    if not force:
        missing = [f for f in REQUIRED_FILES if f not in existing]
        if not missing:
            # Files may exist but be invalid (e.g., ZIP archives named *.csv). Repair/verify and return.
            repair_m5_dir(output_dir)
            final_missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
            if not final_missing:
                print("All required core M5 files already present. Use --full to fetch optional evaluation file.")
                return
        else:
            print(f"Missing required files: {missing}")

    if full:
        print("Performing full archive download (may be large)...")
        cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(output_dir)]
        subprocess.run(cmd, check=True)
    else:
        print("Downloading individual required files...")
        for fname in REQUIRED_FILES:
            if fname in existing and not force:
                continue
            cmd = [
                "kaggle", "competitions", "download", "-c", COMPETITION, "-f", fname, "-p", str(output_dir)
            ]
            print("Downloading", fname)
            subprocess.run(cmd, check=True)

        # Optionally attempt evaluation file
        # for fname in OPTIONAL_FILES:
        #     if fname not in existing or force:
        #         cmd = [
        #             "kaggle", "competitions", "download", "-c", COMPETITION, "-f", fname, "-p", str(output_dir)
        #         ]
        #         print("Attempting optional file", fname)
        #         subprocess.run(cmd, check=False)

    # Unzip/repair downloaded archives (including ZIPs saved with a .csv name)
    repair_m5_dir(output_dir)

    # Final presence check
    final_missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
    if final_missing:
        print(
            "WARNING: Some required files are still missing: "
            + ", ".join(final_missing)
            + "\nTry rerunning with --full or manually verifying your Kaggle acceptance."
        )
    else:
        print("M5 dataset ready at", output_dir)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download M5 forecasting dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir(),
        help="Output directory for dataset (default: repo-root data/)",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if files exist")
    parser.add_argument("--full", action="store_true", help="Download full competition archive instead of individual files")
    parser.add_argument("--verify", action="store_true", help="Only verify presence of required files")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair existing files in --output (unzip *.zip and fix ZIPs saved as *.csv) without downloading",
    )
    return parser.parse_args(argv)


def verify_m5(output_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
    if missing:
        print("Missing:", missing)
        raise SystemExit(1)

    zipped = [f for f in REQUIRED_FILES if _is_zip_file(output_dir / f)]
    if zipped:
        print(
            "Invalid (ZIP archive saved as .csv): "
            + ", ".join(zipped)
            + "\nFix: run `make download_m5` (or run this module with --repair)."
        )
        raise SystemExit(1)

    print("All required files present and look like CSV text.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.repair:
        args.output.mkdir(parents=True, exist_ok=True)
        repair_m5_dir(args.output)
        return
    if args.verify:
        verify_m5(args.output)
        return
    download_m5(args.output, force=args.force, full=args.full)


if __name__ == "__main__":  # pragma: no cover
    main()
