"""Generate answers on *held-out GSM8K validation* for hyperparameter selection.

Usage: python -u -m reasoning_opt.evaluation.dev --config configs/evaluation.json

Not the official BottleCap benchmark. Runs sequentially on one GPU and stores
per-answer outputs as well as concise summary CSV/JSON on Google Drive.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .common import (discover_experiments, experiment_source_fingerprint, fingerprint,
                     load_config, save_json)

NUMBER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")
FINAL = re.compile(r"^Final answer:[ \t]*([^\n]+)[ \t]*$", re.MULTILINE)


def parse_completion(text: str, token_ids: list[int], open_id: int, close_id: int,
                     end_id: int, max_new_tokens: int) -> tuple[str | None, int, bool, str | None]:
    """Mirror BottleCap's strict SVAMP formatting (without official test data)."""
    tokens = [t for t in token_ids if t != end_id]
    opens = [i for i, t in enumerate(tokens) if t == open_id]
    closes = [i for i, t in enumerate(tokens) if t == close_id]
    if bool(opens) != bool(closes) or len(opens) > 1 or len(closes) > 1 or (opens and (opens[0] != 0 or closes[0] < opens[0])):
        return None, len(tokens), False, "unbalanced_thinking"
    reasoning_tokens = closes[0] + 1 if closes else 0
    if end_id not in token_ids:
        return None, reasoning_tokens, False, "truncated_or_unterminated"
    reply = text.replace("<|im_end|>", "").strip()
    if opens:
        if not reply.startswith("<think>") or reply.count("</think>") != 1:
            return None, reasoning_tokens, False, "malformed_thinking_block"
        reply = reply.split("</think>", 1)[1].strip()
    match = FINAL.findall(reply)
    if not match:
        return None, reasoning_tokens, True, "missing_final_answer"
    if not re.fullmatch(r"Final answer:[ \t]*[^\n]+", reply.splitlines()[-1]):
        return None, reasoning_tokens, True, "trailing_content"
    ans = match[-1].strip()
    if not NUMBER.fullmatch(ans):
        return None, reasoning_tokens, True, "invalid_numeric_answer"
    return ans, reasoning_tokens, True, None


def equal_number(prediction: str | None, answer: str) -> bool:
    if prediction is None or not NUMBER.fullmatch(answer.strip()):
        return False
    try:
        return Decimal(prediction.replace(",", "")) == Decimal(answer.strip().replace(",", ""))
    except InvalidOperation:
        return False


def eval_model(model_id: str, adapter: Path | None, rows: list[dict], path: Path, max_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(9001)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.to("cuda").eval()
    model.config.use_cache = True
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    if any(x is None or x == tokenizer.unk_token_id for x in [end_id, open_id, close_id]):
        raise ValueError("Qwen3 special tokens unavailable")
    start = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Partial runs are persisted per example to Drive, so a Colab disconnection
    # can continue from the last fully written JSONL record.
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_ids = [r["id"] for r in rows[:len(records)]]
        if [r["id"] for r in records] != expected_ids:
            raise ValueError("Cannot resume: saved record IDs do not match this validation subset")
        print(f"Resuming from {len(records)}/{len(rows)} completed examples", flush=True)
    try:
        with path.open("a", encoding="utf-8") as dest, torch.inference_mode():
            for index, row in enumerate(rows[len(records):], len(records) + 1):
                ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                      add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
                ids = ids.to("cuda")
                result = model.generate(input_ids=ids, max_new_tokens=max_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=end_id, use_cache=True)
                generated = result[0, ids.shape[1]:].tolist()
                text = tokenizer.decode(generated, skip_special_tokens=False)
                predicted, reasoning_count, boundaries_valid, error = parse_completion(
                    text, generated, open_id, close_id, end_id, max_tokens)
                record = {"id": row["id"], "correct": equal_number(predicted, str(row["answer"])),
                          "gold": row["answer"], "predicted": predicted, "output_tokens": len(generated),
                          "reasoning_tokens": reasoning_count, "thinking_present": open_id in generated,
                          "boundary_valid": boundaries_valid, "parse_error": error, "response": text}
                records.append(record)
                dest.write(json.dumps(record, ensure_ascii=False) + "\n")
                dest.flush()
                if index == 1 or index % 10 == 0 or index == len(rows):
                    print(f"Validated {index}/{len(rows)} | running accuracy={sum(x['correct'] for x in records)/index:.1%} | elapsed={time.monotonic()-start:.0f}s", flush=True)
        return {"questions": len(records), "accuracy": sum(x["correct"] for x in records) / len(records),
                "mean_output_tokens": sum(x["output_tokens"] for x in records)/len(records),
                "mean_reasoning_tokens": sum(x["reasoning_tokens"] for x in records)/len(records),
                "thinking_present_rate": sum(x["thinking_present"] for x in records)/len(records),
                "parse_failure_rate": sum(x["parse_error"] is not None for x in records)/len(records),
                "total_seconds": round(time.monotonic()-start, 1)}
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/evaluation.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    validation_file = Path(cfg["datasets_root"]) / "gsm8k-original" / "val.jsonl"
    if not validation_file.is_file():
        raise FileNotFoundError(validation_file)
    count = int(cfg["dev"]["validation_examples"])
    max_new_tokens = int(cfg["dev"].get("max_new_tokens", 1024))
    if not 1 <= count <= 200 or max_new_tokens < 32:
        raise ValueError("validation_examples must be 1..200; max_new_tokens >= 32")
    rows = [json.loads(line) for line in validation_file.read_text(encoding="utf-8").splitlines() if line.strip()][:count]
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} validation examples available")
    runs = discover_experiments(cfg)
    profile = {"model_id": cfg["model_id"], "val_file_sha256": fingerprint(validation_file),
               "count": count, "max_new_tokens": max_new_tokens, "decoding": "greedy",
               "prompt_format": "Qwen3 thinking=True, BottleCap GSM8K validation"}
    key = __import__("hashlib").sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:10]
    root = Path(cfg["evaluations_root"]) / "development" / f"gsm8k-{key}"
    print(f"Development validation: {count} held-out GSM8K examples, model count={len(runs)}", flush=True)
    print(f"Output: {root}", flush=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU required in Colab")
    summary_rows = []
    for i, (name, adapter) in enumerate(runs, 1):
        dest = root / name
        current = {"profile": profile, "source": experiment_source_fingerprint(name, adapter)}
        meta = dest / "run_metadata.json"
        result_file = dest / "summary.json"
        progress_meta = dest / "in_progress.json"
        if result_file.is_file() and meta.is_file():
            if load_config(meta) != current:
                raise RuntimeError(f"Existing evaluation has different input fingerprint: {dest}")
            print(f"[{i}/{len(runs)}] Reusing existing results: {name}", flush=True)
            metrics = load_config(result_file)
        else:
            if dest.is_dir() and any(dest.iterdir()):
                if not progress_meta.is_file() or load_config(progress_meta) != current:
                    raise RuntimeError(f"Incomplete validation with unknown inputs: {dest}; inspect before retrying")
                print(f"[{i}/{len(runs)}] Resuming partial validation: {name}", flush=True)
            else:
                dest.mkdir(parents=True, exist_ok=True)
                save_json(progress_meta, current)
                print(f"[{i}/{len(runs)}] Generating validation answers: {name}", flush=True)
            metrics = eval_model(cfg["model_id"], adapter, rows, dest / "records.jsonl", max_new_tokens)
            save_json(result_file, metrics)
            save_json(meta, current)
            progress_meta.unlink(missing_ok=True)
        if adapter is None:
            training_info = {"lora_rank": None, "learning_rate": None, "sft_eval_loss": None}
        else:
            run_folder = adapter.parent
            sft_summary = load_config(run_folder / "summary.json")
            sft_config = load_config(run_folder / "config.json")
            training_info = {
                "lora_rank": sft_config["lora"]["rank"],
                "learning_rate": sft_config["training"]["learning_rate"],
                "sft_eval_loss": sft_summary.get("evaluation_metrics", {}).get("eval_loss"),
            }
        summary_rows.append({"model": name, **training_info, **metrics})
        save_json(root / "summary.json", {"evaluation": "DEVELOPMENT - NOT OFFICIAL", "results": summary_rows})
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    print("\nDevelopment comparison (use accuracy + mean tokens, not validation loss alone):")
    for x in summary_rows:
        print(f"{x['model']:<40} acc={x['accuracy']:.1%}, tokens={x['mean_output_tokens']:.1f}, reasoning={x['mean_reasoning_tokens']:.1f}, parse_fail={x['parse_failure_rate']:.1%}")
    print(f"Saved CSV: {root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
