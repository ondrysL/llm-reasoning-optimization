"""Prepare an immutable BottleCap snapshot + GSM8K original-solution SFT data.

Usage:
  python -m reasoning_opt.data.prepare --upstream external/ThinkingCap-Test \
      --output-dir artifacts/datasets

No teacher LLM or generated rationales are used. The SFT dataset contains
reference solutions supplied with the *training split* of the public GSM8K dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

MATH_INSTRUCTION = "Put the final answer on the last line, like: Final answer: 42"


def jsonl_read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def q_from_bottlecap(row: dict) -> str:
    prompt = row["prompt"]
    if len(prompt) != 1 or prompt[0]["role"] != "user":
        raise ValueError("Unexpected BottleCap prompt layout")
    suffix = "\n\n" + MATH_INSTRUCTION
    content = prompt[0]["content"]
    if not content.endswith(suffix):
        raise ValueError("Unexpected GSM8K question suffix in BottleCap prompt")
    return content[: -len(suffix)]


def solution_from_gsm8k(answer_text: str) -> tuple[str, str]:
    if "####" not in answer_text:
        raise ValueError("GSM8K solution without answer marker ####")
    steps, final = answer_text.rsplit("####", 1)
    # Original GSM8K has <<...>> calculator annotations, not readable reasoning.
    steps = re.sub(r"<<[^<>]*>>", "", steps).strip()
    final = final.strip().replace(",", "")
    if not steps or not final:
        raise ValueError("Empty GSM8K reasoning or answer")
    return steps, final


def build_sft_split(bottlecap_rows: list[dict], reference_rows: list[dict], *, split: str) -> list[dict]:
    """Join only official BottleCap GSM8K rows to original GSM8K train solutions."""
    reference = {}
    for r in reference_rows:
        question = r["question"]
        solution = solution_from_gsm8k(r["answer"])
        if question in reference and reference[question] != solution:
            raise ValueError(f"Conflicting reference answers for question: {question[:80]}")
        reference[question] = solution

    output = []
    seen = set()
    for r in bottlecap_rows:
        if r["source"] != "gsm8k":
            continue
        question = q_from_bottlecap(r)
        if question not in reference:
            raise ValueError(f"BottleCap GSM8K question missing from original source: {question[:80]}")
        reasoning, answer = reference[question]
        if answer != str(r["answer"]).strip():
            raise ValueError(f"Answer mismatch for GSM8K question: {question[:80]}")
        item_id = hashlib.sha256(question.encode("utf-8")).hexdigest()[:24]
        if item_id in seen:
            raise ValueError(f"Duplicate GSM8K question within {split}: {item_id}")
        seen.add(item_id)
        output.append({
            "id": item_id,
            "source": "gsm8k",
            "split": split,
            "prompt": r["prompt"],
            "answer": answer,
            "task": "math",
            "reasoning": reasoning,
            "reasoning_origin": "openai/gsm8k train reference; original annotations removed",
        })
    return output


def run(args: argparse.Namespace) -> None:
    upstream = Path(args.upstream).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    prepare_script = upstream / "utils" / "train" / "prepare_data.py"
    if not prepare_script.exists():
        raise FileNotFoundError(f"Missing BottleCap source. First run: bash scripts/bootstrap_bottlecap.sh ({prepare_script})")

    source_train = upstream / "data" / "train.jsonl"
    source_val = upstream / "data" / "val.jsonl"
    if not args.reuse_source or not source_train.exists() or not source_val.exists():
        print("Preparing all four datasets with the unmodified upstream script...", flush=True)
        subprocess.run([sys.executable, str(prepare_script)], cwd=upstream, check=True)

    bottlecap_train = jsonl_read(source_train)
    bottlecap_val = jsonl_read(source_val)

    # The upstream source's train/validation assignment is authoritative.
    print("Loading original GSM8K reference solutions (train split only)...", flush=True)
    from datasets import load_dataset
    gsm = list(load_dataset("openai/gsm8k", "main", split="train"))
    sft_train = build_sft_split(bottlecap_train, gsm, split="train")
    sft_val = build_sft_split(bottlecap_val, gsm, split="val")
    overlap = {r["id"] for r in sft_train} & {r["id"] for r in sft_val}
    if overlap:
        raise RuntimeError(f"Data leakage: {len(overlap)} repeated GSM8K questions across train/validation")
    if not sft_train or not sft_val:
        raise RuntimeError("No GSM8K examples found in at least one split")

    snapshot = output / "bottlecap"
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_train, snapshot / "train.jsonl")
    shutil.copy2(source_val, snapshot / "val.jsonl")
    sft_dir = output / "gsm8k-original"
    jsonl_write(sft_dir / "train.jsonl", sft_train)
    jsonl_write(sft_dir / "val.jsonl", sft_val)

    try:
        pinned_commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pinned_commit = "unknown"
    manifest = {
        "format_version": 1,
        "upstream_repository": "BottleCapAI/ThinkingCap-Test",
        "upstream_commit": pinned_commit,
        "source_dataset": "openai/gsm8k (main, train split)",
        "source_code_sha256": sha256(prepare_script),
        "split_policy": "Use BottleCapAI prepare_data.py splits verbatim; never re-split after adding reasoning.",
        "reasoning_policy": "Original GSM8K train solutions only; no teacher LLM or concise rewriting.",
        "counts": {
            "bottlecap_train": len(bottlecap_train),
            "bottlecap_val": len(bottlecap_val),
            "bottlecap_train_by_source": dict(Counter(r["source"] for r in bottlecap_train)),
            "bottlecap_val_by_source": dict(Counter(r["source"] for r in bottlecap_val)),
            "sft_train": len(sft_train),
            "sft_val": len(sft_val),
        },
        "files_sha256": {},
    }
    for name in ["bottlecap/train.jsonl", "bottlecap/val.jsonl", "gsm8k-original/train.jsonl", "gsm8k-original/val.jsonl"]:
        manifest["files_sha256"][name] = sha256(output / name)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}, indent=2), flush=True)
    print("Finished: data prepared; NO model has been trained.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", default="external/ThinkingCap-Test")
    p.add_argument("--output-dir", default="artifacts/datasets")
    p.add_argument("--reuse-source", action="store_true", help="Reuse existing source train.jsonl and val.jsonl if both exist")
    run(p.parse_args())


if __name__ == "__main__":
    main()
