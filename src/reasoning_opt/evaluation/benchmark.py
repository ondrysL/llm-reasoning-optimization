"""Batch wrapper around the *unmodified scoring protocol* of BottleCapAI.

python -u -m reasoning_opt.evaluation.benchmark --config configs/evaluation.json

First run smoke to test the pipeline; after choosing a model using our held-out
validation set, run the official benchmark once, without tuning against it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .common import (discover_experiments, experiment_source_fingerprint,
                     fingerprint, load_config, save_json)

VALID_BENCHMARKS = {"svamp", "bfcl", "mbpp", "ifeval"}
PATCH_ANCHOR = "enable_lora=rank is not None, max_lora_rank=rank or 16)"


def make_hardware_wrapper(upstream: Path, backend: str, dtype: str) -> Path:
    """Copy original evaluate.py with ONLY model-engine hardware arguments changed.

    The prompts, gold answers, graders, sampling settings and scoring code remain
    unchanged. We never modify the upstream evaluate.py itself.
    """
    if backend != "TRITON_ATTN":
        raise ValueError("Colab T4 preset supports attention_backend=TRITON_ATTN")
    if dtype != "half":
        raise ValueError("Colab T4 preset supports dtype=half")
    origin = upstream / "evaluate.py"
    if not origin.is_file():
        raise FileNotFoundError(f"BottleCap checkout missing: {origin}")
    content = origin.read_text(encoding="utf-8")
    if content.count(PATCH_ANCHOR) != 1:
        raise RuntimeError("BottleCap evaluator LLM signature changed. Inspect before patching!")
    new = content.replace(PATCH_ANCHOR,
        'enable_lora=rank is not None, max_lora_rank=rank or 16,\n'
        '                dtype="half", attention_backend="TRITON_ATTN")')
    target = upstream / "evaluate_t4.py"
    if not target.exists() or target.read_text(encoding="utf-8") != new:
        target.write_text(new, encoding="utf-8")
    return target


def stream_command(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}. Log: {log}")


def ensure_suite(upstream: Path, root: Path) -> None:
    manifest = root / "manifest.json"
    if manifest.is_file() and (root / "questions.jsonl").is_file() and (root / "smoke.jsonl").is_file():
        return
    print("Downloading pinned BottleCap evaluation questions (first time only)...", flush=True)
    stream_command([sys.executable, "-u", "utils/eval/benchmarks.py", "--questions", str(root)],
                   cwd=upstream, log=root.parent / "benchmark_prepare.log")


def summarize(folder: Path, name: str) -> list[dict]:
    data = load_config(folder / "results.json")
    return [{"model": name, "dataset": dataset,
             "accuracy": round(v["accuracy"], 6),
             "mean_output_tokens": round(v["mean_output_tokens"], 2),
             "mean_reasoning_tokens": round(v["mean_reasoning_tokens"], 2),
             "parse_failure_rate": round(v["parse_failure_rate"], 6),
             "prompts": v["prompts"], "samples": data["samples"]}
            for dataset, v in data["results"].items()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/evaluation.json"))
    args = p.parse_args()
    cfg = load_config(args.config)
    settings = cfg["benchmark"]
    mode = settings["mode"]
    if mode not in ("smoke", "benchmark"):
        raise ValueError("mode must be 'smoke' or 'benchmark'")
    benchmarks = settings["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks or not set(benchmarks) <= VALID_BENCHMARKS:
        raise ValueError("Choose valid benchmark names among svamp, bfcl, mbpp, ifeval")
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError("Duplicate benchmarks")
    samples = int(settings["samples"])
    if not 1 <= samples <= 8:
        raise ValueError("samples must be 1..8")
    upstream = Path(cfg["upstream_dir"]).resolve()
    root = Path(cfg["evaluations_root"])
    runs = discover_experiments(cfg)
    runner_script = make_hardware_wrapper(
        upstream, settings["attention_backend"], settings["dtype"])
    question_root = root / "_questions"
    ensure_suite(upstream, question_root)
    suite_manifest = question_root / "manifest.json"
    if not suite_manifest.is_file():
        raise RuntimeError("BottleCap evaluation question manifest missing after preparation")

    upstream_commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    profile = {"mode": mode, "benchmarks": sorted(benchmarks), "samples": samples,
               "model": cfg["model_id"], "evaluator_sha256": fingerprint(runner_script),
               "suite_manifest_sha256": fingerprint(suite_manifest),
               "upstream_commit": upstream_commit, "backend": settings["attention_backend"],
               "dtype": settings["dtype"], "seed": 9001, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}}
    key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:12]
    folder = root / "bottlecap" / f"{mode}-{key}"
    print(f"\nBottleCap {mode.upper()} | {benchmarks}, samples={samples}", flush=True)
    print(f"Experiments: {len(runs) - 1} adapters, plus original Qwen baseline", flush=True)
    all_rows: list[dict] = []
    baseline = folder / "base"
    for i, (name, adapter) in enumerate(runs, 1):
        dest = folder / ("base" if adapter is None else f"models/{name}")
        marker = dest / "run_metadata.json"
        results = dest / "results.json"
        current = {"profile": profile, "source": experiment_source_fingerprint(name, adapter)}
        if marker.is_file() and results.is_file():
            if load_config(marker) != current:
                raise RuntimeError(f"Existing evaluation has different fingerprint: {dest}")
            print(f"[{i}/{len(runs)}] Reusing stored BottleCap score for {name}", flush=True)
        else:
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f"Incomplete run at {dest}; inspect files before retrying")
            print(f"[{i}/{len(runs)}] Evaluating {name}...", flush=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(runner_script), "--questions", str(question_root),
                       "--out", str(dest), "--benchmarks", ",".join(benchmarks),
                       "--samples", str(samples)]
            if mode == "smoke":
                command.append("--smoke")
            if adapter:
                command += ["--adapter", str(adapter)]
            stream_command(command, cwd=upstream,
                           log=folder / "logs" / f"{name}.log")
            save_json(marker, current)
        if mode == "smoke":
            # BottleCap deliberately suppresses smoke scores: 5 questions per
            # benchmark are only a pipeline check, not a model comparison.
            all_rows.append({"model": name, "smoke_completed": True})
        else:
            all_rows.extend(summarize(dest, name))
        save_json(folder / "summary.json", {"status": "smoke - no scores" if mode == "smoke" else "benchmark",
                     "profile": profile, "results": all_rows})
        if mode == "benchmark" and adapter is not None:
            comp = folder / "comparisons" / f"{name}.json"
            comp.parent.mkdir(parents=True, exist_ok=True)
            stream_command([sys.executable, "-u", "compare.py", str(baseline), str(dest),
                            "--out", str(comp)], cwd=upstream,
                           log=folder / "logs" / f"compare-{name}.log")
    if mode == "benchmark" and all_rows:
        csv_file = folder / "summary.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
        print(f"\nBatch evaluation finished. Results table: {csv_file}", flush=True)
        official = sorted(benchmarks) == sorted(VALID_BENCHMARKS) and samples == 8
        print("FULL official recipe" if official else "PRELIMINARY subset - not a full official evaluation", flush=True)
    else:
        print("\nSmoke successful. No official scores reported; use mode=benchmark for scoring.", flush=True)


if __name__ == "__main__":
    main()
