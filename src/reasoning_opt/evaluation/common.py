"""Shared experiment discovery, atomic JSON writes, and guardrails."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_config(filename: str | Path) -> dict:
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def save_json(filename: str | Path, data: dict) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def fingerprint(filename: str | Path) -> str:
    h = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_experiments(cfg: dict) -> list[tuple[str, Path | None]]:
    """Return (run id, adapter path), base first. Only FINISHED SFT runs are eligible.

    experiment_names: [] means all completed runs. Otherwise, exact run directory
    names (including timestamps), not ambiguous experiment prefixes.
    """
    root = Path(cfg["experiments_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Experiments directory not found: {root}")
    wanted = cfg.get("experiment_names", [])
    if not isinstance(wanted, list) or not all(isinstance(x, str) for x in wanted):
        raise ValueError("experiment_names must be a list of complete run directory names")
    folders = sorted([root / name for name in wanted] if wanted else root.iterdir(), key=lambda x: x.name)
    found = [("base", None)]
    missing = []
    for folder in folders:
        if not folder.is_dir():
            missing.append(str(folder))
            continue
        adapter = folder / "adapter"
        if (folder / "summary.json").is_file() and (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file():
            found.append((folder.name, adapter.resolve()))
        elif wanted:
            missing.append(f"{folder}: not a completed training run with saved adapter")
    if missing:
        raise FileNotFoundError("Requested experiment(s) missing or incomplete: " + "; ".join(missing))
    if len(found) == 1:
        raise ValueError(f"No finished SFT experiments found under {root}")
    return found


def experiment_source_fingerprint(name: str, adapter: Path | None) -> dict:
    return {"run_id": name, "adapter_config_sha256": fingerprint(adapter / "adapter_config.json") if adapter else None,
            "adapter_sha256": fingerprint(adapter / "adapter_model.safetensors") if adapter else None}
