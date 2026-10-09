"""Sequentially run multiple independent SFT configurations on one GPU.

python -u -m reasoning_opt.training.batch --config configs/sft_batch.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def config_paths(path: Path) -> list[Path]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    paths = cfg.get("configs")
    if not isinstance(paths, list) or not paths or not all(isinstance(x, str) for x in paths):
        raise ValueError("Batch config needs a nonempty 'configs' list")
    root = Path.cwd()
    files = [(root / x).resolve() for x in paths]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f)
    names = [json.loads(f.read_text(encoding="utf-8"))["experiment_name"] for f in files]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate experiment names in batch config: {names}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_batch.json"))
    parser.add_argument("--preflight-only", action="store_true", help="Validate all configurations without training")
    args = parser.parse_args()
    config_files = config_paths(args.config)
    print(f"Running {len(config_files)} SFT configurations sequentially", flush=True)
    for i, path in enumerate(config_files, 1):
        print(f"\n{'='*60}\nSFT {i}/{len(config_files)}: {path}\n{'='*60}", flush=True)
        command = [sys.executable, "-u", "-m", "reasoning_opt.training.sft", "--config", str(path)]
        if args.preflight_only:
            command.append("--preflight-only")
        subprocess.run(command, check=True)
    print("\nAll requested SFT runs completed. Adapters are stored on Google Drive.", flush=True)


if __name__ == "__main__":
    main()
