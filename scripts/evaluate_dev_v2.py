#!/usr/bin/env python3
"""Resumable GSM8K *development* evaluation of Qwen3-0.6B + completed LoRA adapters.

This is NOT the official BottleCap benchmark. Each PBS array index owns one model.
The pilot (--limit 10) and full run (--limit 200) share the SAME result directory.
Run on GPU nodes via PBS; --list and --summary need no GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 9001
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

NUMBER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")
FINAL = re.compile(r"^Final answer:[ \t]*([^\n]+)[ \t]*$", re.MULTILINE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(item, out, indent=2, ensure_ascii=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def models_at(base: Path):
    root = base / "experiments"
    if not root.is_dir():
        raise FileNotFoundError(f"Experiments directory does not exist: {root}")
    models = [("base", None)]
    for folder in sorted(root.glob("gsm8k-sft-*")):
        adapter = folder / "adapter"
        if (folder / "summary.json").is_file() and (adapter / "adapter_model.safetensors").is_file() and (adapter / "adapter_config.json").is_file():
            models.append((folder.name, adapter))
    if len(models) != 9:
        raise RuntimeError(f"Expected baseline + 8 finished SFT runs, got {len(models)}: {[m[0] for m in models]}")
    return models


def profile_for(base: Path):
    dataset = base / "datasets" / "gsm8k-original" / "val.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(f"Missing development dataset: {dataset}")
    profile = {
        "version": "dev_v2_sampling_8192_v1",
        "base_model": MODEL_ID,
        "dataset_sha256": sha256(dataset),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "sample",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed_policy": "torch.manual_seed(9001 + question_index)",
        "prompt_format": "Qwen3 apply_chat_template(enable_thinking=True, add_generation_prompt=True)",
        "parser": "strict Final answer: NUMBER on last line",
    }
    key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:12]
    return dataset, profile, base / "evaluations" / "development_v2" / f"gsm8k-{key}"


def source_fingerprint(adapter: Path | None):
    if adapter is None:
        return {"adapter": None}
    return {
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
    }


def parse_completion(text: str, token_ids: list[int], open_id: int, close_id: int,
                     end_id: int):
    tokens = [tid for tid in token_ids if tid != end_id]
    opens = [i for i, tid in enumerate(tokens) if tid == open_id]
    closes = [i for i, tid in enumerate(tokens) if tid == close_id]
    if (bool(opens) != bool(closes) or len(opens) > 1 or len(closes) > 1
            or (opens and (opens[0] != 0 or closes[0] < opens[0]))):
        return None, len(tokens), "unbalanced_thinking"
    thinking_tokens = closes[0] + 1 if closes else 0
    if end_id not in token_ids:
        return None, thinking_tokens, "truncated_or_unterminated"
    reply = text.replace("<|im_end|>", "").strip()
    if opens:
        if not reply.startswith("<think>") or reply.count("</think>") != 1:
            return None, thinking_tokens, "malformed_thinking_block"
        reply = reply.split("</think>", 1)[1].strip()
    answers = FINAL.findall(reply)
    if not answers:
        return None, thinking_tokens, "missing_final_answer"
    if not re.fullmatch(r"Final answer:[ \t]*[^\n]+", reply.splitlines()[-1]):
        return None, thinking_tokens, "trailing_content"
    answer = answers[-1].strip()
    if not NUMBER.fullmatch(answer):
        return None, thinking_tokens, "invalid_numeric_answer"
    return answer, thinking_tokens, None


def numeric_equal(prediction, gold):
    if prediction is None or not NUMBER.fullmatch(str(gold).strip()):
        return False
    try:
        return Decimal(prediction.replace(",", "")) == Decimal(str(gold).strip().replace(",", ""))
    except InvalidOperation:
        return False


def load_record(path: Path, row: dict, index: int, cap: int):
    if not path.is_file():
        return None
    result = read_json(path)
    if (result.get("id") != row["id"] or result.get("index") != index
            or result.get("max_new_tokens") != cap):
        raise RuntimeError(f"Saved example does not match expected validation question: {path}")
    return result


def summary_of(records: list[dict], total: int):
    size = len(records)
    return {
        "completed": size,
        "total": total,
        "accuracy": sum(bool(x["correct"]) for x in records) / size if size else None,
        "mean_output_tokens": sum(x["output_tokens"] for x in records) / size if size else None,
        "mean_reasoning_tokens": sum(x["reasoning_tokens"] for x in records) / size if size else None,
        "parse_failure_rate": sum(x["parse_error"] is not None for x in records) / size if size else None,
        "hit_token_cap": sum(x["hit_token_cap"] for x in records),
        "mean_generation_seconds": sum(x["generation_seconds"] for x in records) / size if size else None,
    }


def evaluate(base: Path, index: int, limit: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Submit this script via PBS, not on the frontend.")

    dataset_path, profile, out_root = profile_for(base)
    models = models_at(base)
    if not 0 <= index < len(models):
        raise ValueError(f"Model index must be 0..{len(models)-1}")
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 1 <= limit <= min(200, len(rows)):
        raise ValueError(f"--limit must be between 1 and {min(200, len(rows))}")

    name, adapter = models[index]
    target = out_root / name
    expected_meta = {"profile": profile, "run_name": name, "source": source_fingerprint(adapter)}
    meta_path = target / "metadata.json"
    if meta_path.exists():
        if read_json(meta_path) != expected_meta:
            raise RuntimeError(f"Cannot resume: config/data/adapter changed for {target}")
    else:
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"Unexpected existing output without metadata: {target}")
        atomic_json(meta_path, expected_meta)

    print(f"Model [{index}/{len(models)-1}]: {name}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Examples requested: {limit}; max_new_tokens: {MAX_NEW_TOKENS}", flush=True)
    print(f"Results: {target}", flush=True)

    # Detect fully completed pilot or resume runs before loading any GPU model.
    records = [load_record(target / "records" / f"{i:04d}.json", row, i, MAX_NEW_TOKENS)
               for i, row in enumerate(rows[:limit])]
    remaining = [i for i, r in enumerate(records) if r is None]
    if not remaining:
        print(f"All {limit} answers already exist; no GPU generation needed", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        open_id = tokenizer.convert_tokens_to_ids("<think>")
        close_id = tokenizer.convert_tokens_to_ids("</think>")
        if any(x is None or x == tokenizer.unk_token_id for x in (end_id, open_id, close_id)):
            raise RuntimeError("Missing Qwen3 special tokens")

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                                    trust_remote_code=False)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter))
        model = model.to("cuda").eval()
        model.config.use_cache = True

        for i in remaining:
            row = rows[i]
            # Per-question seed ensures the same sample even if PBS resumes later.
            torch.manual_seed(SEED + i)
            with torch.inference_mode():
                tokens = tokenizer.apply_chat_template(
                    row["prompt"], tokenize=True, add_generation_prompt=True,
                    enable_thinking=True, return_tensors="pt")
                ids = (tokens if isinstance(tokens, torch.Tensor)
                       else tokens["input_ids"]).to("cuda")
                t0 = time.monotonic()
                generated = model.generate(
                    input_ids=ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=end_id,
                    use_cache=True,
                )[0, ids.shape[1]:].tolist()
                elapsed = time.monotonic() - t0
            response = tokenizer.decode(generated, skip_special_tokens=False)
            answer, reasoning_tokens, error = parse_completion(response, generated, open_id, close_id, end_id)
            rec = {
                "index": i, "id": row["id"], "question": row["prompt"][0]["content"],
                "gold": str(row["answer"]), "predicted": answer,
                "correct": numeric_equal(answer, row["answer"]),
                "output_tokens": len(generated), "reasoning_tokens": reasoning_tokens,
                "thinking_present": open_id in generated, "parse_error": error,
                "hit_token_cap": len(generated) >= MAX_NEW_TOKENS and end_id not in generated,
                "generation_seconds": round(elapsed, 3), "max_new_tokens": MAX_NEW_TOKENS,
                "response": response,
            }
            atomic_json(target / "records" / f"{i:04d}.json", rec)
            records[i] = rec
            completed = sum(r is not None for r in records)
            print(f"[{name}] {completed}/{limit} | correct={rec['correct']} | "
                  f"tokens={len(generated)} | parse_error={error} | {elapsed:.1f}s", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()

    final_records = [r for r in records if r is not None]
    aggregate = {"model": name, "profile": profile, **summary_of(final_records, limit)}
    atomic_json(target / "summary.json", aggregate)
    print("SUMMARY", json.dumps(aggregate, ensure_ascii=False), flush=True)


def summarize(base: Path, limit: int):
    dataset_path, _, out_root = profile_for(base)
    models = models_at(base)
    if not 1 <= limit <= 200:
        raise ValueError("--limit must be 1..200")
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    output = []
    for name, _ in models:
        folder = out_root / name / "records"
        records = []
        for i, row in enumerate(rows):
            r = load_record(folder / f"{i:04d}.json", row, i, MAX_NEW_TOKENS)
            if r is not None:
                records.append(r)
        stats = summary_of(records, limit)
        output.append({"model": name, **stats})
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"comparison_{limit}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Saved {csv_path}")
    for x in output:
        a = "N/A" if x["accuracy"] is None else f"{x['accuracy']:.1%}"
        n = "N/A" if x["mean_output_tokens"] is None else f"{x['mean_output_tokens']:.0f}"
        print(f"{x['completed']:3}/{limit}  acc={a:>6}  tokens={n:>6}  {x['model']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/auto/brno2/brno2/ondrys/llm-reasoning"))
    p.add_argument("--index", type=int, help="0=original Qwen, 1..8=LoRA runs")
    p.add_argument("--limit", type=int, default=200, help="10 pilot then 200 full; saved answers are reused")
    p.add_argument("--list", action="store_true", help="Show PBS array index -> model mapping without GPU")
    p.add_argument("--summary", action="store_true", help="Build CSV of currently saved answers without GPU")
    args = p.parse_args()
    base = args.base.expanduser().resolve()
    if args.list:
        for i, (name, adapter) in enumerate(models_at(base)):
            print(i, name, str(adapter) if adapter else "base model")
    elif args.summary:
        summarize(base, args.limit)
    elif args.index is not None:
        evaluate(base, args.index, args.limit)
    else:
        p.error("Provide --index N, --list, or --summary")


if __name__ == "__main__":
    main()
