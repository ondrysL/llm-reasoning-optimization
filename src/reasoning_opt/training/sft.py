"""Configurable Qwen3 LoRA SFT on the existing GSM8K original-reasoning split.

Colab:
    python -u -m reasoning_opt.training.sft --config configs/sft.json --preflight-only
    python -u -m reasoning_opt.training.sft --config configs/sft.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("reasoning_opt.sft")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}. Run Data Preparation first.")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def make_target(row: dict, tokenizer) -> str:
    """Qwen3 must learn to emit the opening <think> as its *first* new token."""
    reasoning = row["reasoning"].strip()
    answer = str(row["answer"]).strip()
    if not reasoning or not answer:
        raise ValueError(f"Empty target for record {row.get('id')}")
    if "<think>" in reasoning or "</think>" in reasoning:
        raise ValueError(f"Reasoning unexpectedly contains chat template tokens: {row.get('id')}")
    # Use the actual Qwen3 chat turn terminator; the tokenizer's EOS usually is
    # <|im_end|>, but making this explicit avoids cross-model surprises.
    end = "<|im_end|>"
    if tokenizer.convert_tokens_to_ids(end) == tokenizer.unk_token_id:
        raise ValueError("Expected Qwen3 token <|im_end|> is missing")
    return f"<think>\n{reasoning}\n</think>\n\nFinal answer: {answer}{end}"


def tokenize_row(row: dict, tokenizer, max_seq_length: int) -> tuple[dict | None, str | None]:
    """Return teacher-forced input with prompt masked in labels."""
    prompt = tokenizer.apply_chat_template(
        row["prompt"], tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    if prompt.rstrip().endswith("<think>"):
        raise ValueError("Qwen chat template already inserts <think>. Fix boundary before SFT!")
    target = make_target(row, tokenizer)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (not target_ids or target_ids[0] != open_id or
            target_ids.count(open_id) != 1 or target_ids.count(close_id) != 1 or
            target_ids.index(close_id) <= 0 or target_ids[-1] != end_id):
        raise ValueError(f"Malformed SFT target for record {row.get('id')}: {target[:150]!r}")
    if open_id in prompt_ids or close_id in prompt_ids:
        raise ValueError("Unexpected thinking markers inside the prompt")
    full = prompt_ids + target_ids
    if len(full) > max_seq_length:
        return None, "too_long"
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }, None


def build_split(rows: list[dict], tokenizer, *, limit: int, max_length: int,
                seed: int, shuffle: bool) -> tuple[list[dict], Counter]:
    if limit < 1:
        raise ValueError("train_examples/validation_examples must be positive")
    rows = list(rows)
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit > len(rows):
        raise ValueError(f"Requested {limit} examples, but only {len(rows)} are available")
    stats: Counter = Counter()
    samples: list[dict] = []
    for row in rows:
        if row.get("source") != "gsm8k":
            raise ValueError("SFT trainer expects only GSM8K reasoning records")
        encoded, reason = tokenize_row(row, tokenizer, max_length)
        stats["seen"] += 1
        if reason:
            stats[reason] += 1
            continue
        assert encoded is not None
        samples.append(encoded)
        if len(samples) >= limit:
            break
    stats["kept"] = len(samples)
    if len(samples) < limit:
        LOG.warning("Only %d/%d usable examples (max_seq_length=%d)",
                    len(samples), limit, max_length)
    if not samples:
        raise ValueError("No SFT examples fit into max_seq_length")
    return samples, stats


def validate_overlap(train_rows: list[dict], val_rows: list[dict]) -> None:
    train_ids = {r["id"] for r in train_rows}
    val_ids = {r["id"] for r in val_rows}
    if not train_ids.isdisjoint(val_ids):
        raise ValueError("Dataset leakage: train/validation share GSM8K question IDs")


def collate_examples(features: list[dict], pad_token_id: int):
    import torch
    longest = max(len(row["input_ids"]) for row in features)
    # 8-way padding generally plays nicely with fp16 kernels.
    longest = ((longest + 7) // 8) * 8
    def padded(name, pad):
        return torch.tensor(
            [row[name] + [pad] * (longest - len(row[name])) for row in features],
            dtype=torch.long,
        )
    return {
        "input_ids": padded("input_ids", pad_token_id),
        "attention_mask": padded("attention_mask", 0),
        "labels": padded("labels", -100),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def configured_run(config: dict, preflight_only: bool) -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer

    model_id = config["model_id"]
    data = config["data"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    path = Path(data["root_dir"]).expanduser()
    source = data.get("source", "gsm8k-original")
    train_path = path / source / "train.jsonl"
    val_path = path / source / "val.jsonl"
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    validate_overlap(train_rows, val_rows)

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_length = int(data["max_seq_length"])

    train_samples, train_stat = build_split(
        train_rows, tokenizer, limit=int(data["train_examples"]),
        max_length=max_length, seed=seed, shuffle=True,
    )
    val_samples, val_stat = build_split(
        val_rows, tokenizer, limit=int(data["validation_examples"]),
        max_length=max_length, seed=seed, shuffle=False,
    )

    lengths = sorted(len(x["input_ids"]) for x in train_samples)
    lengths_mean = sum(lengths) / len(lengths)
    lengths_p95 = lengths[int((len(lengths) - 1) * 0.95)]
    LOG.info("Train rows=%s, validation rows=%s", dict(train_stat), dict(val_stat))
    LOG.info("Train sequence length: mean=%.1f tokens; p95=%d; max=%d; cap=%d",
             lengths_mean, lengths_p95, lengths[-1], max_length)
    first = train_samples[0]
    first_target_ids = [tid for tid, label in zip(first["input_ids"], first["labels"]) if label != -100]
    preview = tokenizer.decode(first_target_ids, skip_special_tokens=False)
    LOG.info("SFT target preview (first 350 chars):\n%s", preview[:350])
    if not preview.startswith("<think>") or "</think>" not in preview or "Final answer:" not in preview:
        raise AssertionError("SFT preflight failed: target format is incorrect")

    if preflight_only:
        LOG.info("PREFLIGHT PASSED. No model loaded, no training performed.")
        return

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. In Colab choose Runtime > Change runtime type > T4 GPU")
    set_seed(seed)
    LOG.info("CUDA device: %s; available VRAM: %.1f GiB",
             torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / (1024**3))

    out_base = Path(config["output"]["root_dir"]).expanduser()
    run_name = str(config["experiment_name"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("experiment_name should contain only letters, digits, dots, underscore or dashes")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = out_base / f"{run_name}-{run_id}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {run_dir}")
    run_dir.mkdir(parents=True)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(file_handler)
    config_snapshot = json.loads(json.dumps(config))
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False) + "\n")
    fingerprint = {
        "git_commit": repository_commit(),
        "dataset_manifest_sha256": sha256(path / "manifest.json") if (path / "manifest.json").is_file() else None,
        "train_jsonl_sha256": sha256(train_path),
        "validation_jsonl_sha256": sha256(val_path),
        "train_stats": dict(train_stat),
        "validation_stats": dict(val_stat),
    }
    (run_dir / "reproducibility.json").write_text(json.dumps(fingerprint, indent=2) + "\n")
    LOG.info("Experiment output on Drive: %s", run_dir)

    model = None
    start = time.monotonic()
    try:
        LOG.info("Loading base model %s in float16...", model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=False,
        )
        base_model.config.use_cache = False
        lora = LoraConfig(
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=lora_cfg["target_modules"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora)
        model.print_trainable_parameters()

        class LiveProgress(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = dict(logs or {})
                if not logs:
                    return
                record = {"step": state.global_step, "max_steps": state.max_steps,
                          "elapsed_seconds": round(time.monotonic() - start, 1), **logs}
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
                items = " ".join(f"{k}={v:.5g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                 for k, v in logs.items() if k not in {"epoch", "total_flos"})
                gpu = torch.cuda.memory_allocated(0) / (1024**3)
                LOG.info("STEP %d/%d | %s | GPU memory %.2f GiB | elapsed %.0fs",
                         state.global_step, state.max_steps, items, gpu, record["elapsed_seconds"])

            def on_save(self, args, state, control, **kwargs):
                LOG.info("Checkpoint saved at step %d to %s", state.global_step, args.output_dir)

        args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            num_train_epochs=float(train_cfg.get("epochs", 1)),
            max_steps=int(train_cfg.get("max_steps", -1)),
            per_device_train_batch_size=int(train_cfg["batch_size"]),
            per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", 1)),
            gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
            learning_rate=float(train_cfg["learning_rate"]),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            optim=train_cfg.get("optimizer", "adamw_torch"),
            fp16=True, bf16=False,
            gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps", eval_steps=int(train_cfg["eval_steps"]),
            save_strategy="steps", save_steps=int(train_cfg["save_steps"]),
            save_total_limit=int(train_cfg.get("save_total_limit", 2)),
            logging_strategy="steps", logging_steps=int(train_cfg["logging_steps"]),
            logging_first_step=True,
            logging_dir=str(run_dir / "tensorboard"),
            report_to=["tensorboard"],
            disable_tqdm=True,
            prediction_loss_only=True,
            dataloader_num_workers=0,
            save_safetensors=True,
            remove_unused_columns=False,
            seed=seed, data_seed=seed,
        )
        dataset_train = Dataset.from_list(train_samples)
        dataset_val = Dataset.from_list(val_samples)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            data_collator=lambda rows: collate_examples(rows, tokenizer.pad_token_id),
            processing_class=tokenizer,
            callbacks=[LiveProgress()],
        )
        LOG.info("Starting SFT: epochs=%s, batch=%s, grad_acc=%s, effective_batch=%s, lr=%s",
                 train_cfg.get("epochs", 1), train_cfg["batch_size"],
                 train_cfg["gradient_accumulation_steps"],
                 int(train_cfg["batch_size"]) * int(train_cfg["gradient_accumulation_steps"]),
                 train_cfg["learning_rate"])
        trained = trainer.train()
        LOG.info("Final validation evaluation...")
        evaluation = trainer.evaluate()
        adapter_dir = run_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        summary = {
            "run_dir": str(run_dir),
            "adapter_dir": str(adapter_dir),
            "train_examples": len(train_samples),
            "validation_examples": len(val_samples),
            "train_metrics": trained.metrics,
            "evaluation_metrics": evaluation,
            "total_seconds": time.monotonic() - start,
            "note": "Validation metrics are teacher-forced loss, NOT SVAMP answer accuracy.",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        LOG.info("FINISHED. Final validation loss: %.4f", evaluation.get("eval_loss", float("nan")))
        LOG.info("Saved LoRA adapter: %s", adapter_dir)
        # A tiny qualitative smoke check. It is not an accuracy benchmark.
        sample_count = int(train_cfg.get("sanity_generations", 2))
        if sample_count > 0:
            try:
                model.eval()
                model.config.use_cache = True
                generated = []
                with torch.inference_mode():
                    for row in val_rows[:sample_count]:
                        ids = tokenizer.apply_chat_template(
                            row["prompt"], tokenize=True, add_generation_prompt=True,
                            enable_thinking=True, return_tensors="pt",
                        ).to(next(model.parameters()).device)
                        response = model.generate(
                            input_ids=ids,
                            max_new_tokens=384,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        )[0, ids.shape[-1]:]
                        response_text = tokenizer.decode(response, skip_special_tokens=False)
                        valid = response_text.startswith("<think>") and "</think>" in response_text and "Final answer:" in response_text
                        generated.append({"question": row["prompt"][0]["content"],
                                          "expected_answer": row["answer"],
                                          "response": response_text,
                                          "format_looks_valid": valid})
                        LOG.info("Generation sanity check: format_looks_valid=%s, response excerpt=%r",
                                 valid, response_text[:280])
                with (run_dir / "generation_samples.jsonl").open("w", encoding="utf-8") as f:
                    for row in generated:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                LOG.exception("Optional generation sanity check failed; adapter is already saved")
        LOG.info("Saved metrics and TensorBoard to: %s", run_dir)
    except Exception:
        LOG.exception("Training failed; available logs/checkpoints remain under: %s", run_dir)
        raise
    finally:
        LOG.removeHandler(file_handler)
        file_handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.json"))
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate data and chat template without loading the model")
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout, force=True)
    path = options.config.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    LOG.info("Using config: %s", path)
    configured_run(config, options.preflight_only)


if __name__ == "__main__":
    main()
