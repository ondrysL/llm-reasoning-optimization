# LLM Reasoning Optimization — starter v0.1

Independent research project for shortening **Qwen3-0.6B** thinking without losing answer accuracy, using SFT then GRPO.

This first deliverable **implements the dataset preparation pipeline only**. It does not yet implement SFT training, GRPO, or model evaluation. These will be added as separate components after the data format is validated.

## Why a separate repository?

All original research code lives here. We clone the unmodified [BottleCapAI ThinkingCap-Test](https://github.com/BottleCapAI/ThinkingCap-Test) repo at a **pinned Git commit** to `external/ThinkingCap-Test/` (ignored by Git). BottleCap provides the suggested mixed training dataset and benchmark/reward code; our own Python modules orchestrate training and analysis.

## Directory layout

```text
configs/data.json                 example data/training preferences
scripts/bootstrap_bottlecap.sh    clone fixed upstream revision
src/reasoning_opt/data/prepare.py data pipeline: BottleCap snapshot + GSM8K original reasoning
tests/                             offline data tests
notebooks/01_colab_setup.ipynb    minimal Colab setup/CLI notebook
external/                          cloned upstream (gitignored)
artifacts/                         generated datasets (gitignored)
```

## Run locally (Python >= 3.10)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m unittest discover -s tests -v
bash scripts/bootstrap_bottlecap.sh
python -m reasoning_opt.data.prepare --output-dir artifacts/datasets
```

`prepare` runs the original BottleCap preparation script (downloading four datasets) and loads original GSM8K **train** solutions from Hugging Face. Use `--reuse-source` to avoid repeating BottleCap downloads if the upstream `data/train.jsonl` and `val.jsonl` already exist.

### Colab with persistent Google Drive

See `notebooks/01_colab_setup.ipynb`. Work in `/content` (fast temporary disk), keep finalized data and model checkpoints on Google Drive.

In a Colab shell/notebook after setup:

```bash
python -m reasoning_opt.data.prepare \
    --upstream external/ThinkingCap-Test \
    --output-dir /content/drive/MyDrive/llm-reasoning/datasets
```

Only **source `gsm8k` rows** are used for the SFT variant, and only examples from the upstream **train / validation split** are joined to reference reasoning. Nothing is automatically shortened and no other LLM is called.

## Output format

```text
datasets/
  bottlecap/train.jsonl          original mixed train set, questions and final answers for RL
  bottlecap/val.jsonl            original mixed val set
  gsm8k-original/train.jsonl     original math questions + human-written solutions for SFT
  gsm8k-original/val.jsonl
  manifest.json                  counts, source commit, file hashes, split policy
```

SFT records have `id`, `split`, `source`, `prompt`, `answer`, `task`, `reasoning`, and `reasoning_origin`. `reasoning` contains the human-written GSM8K derivation, with calculator annotations `<<...>>` removed. The *trainer* will format this appropriately for Qwen3's open `<think>` prompt; storing raw reasoning rather than pre-tokenized targets prevents coupling data to a particular chat template.

The upstream script shuffles each source deterministically using seed 0 and holds out exactly 200 examples per source. We **do not resplit** the data, preventing SFT/RL train–validation leakage. Do not train on the BottleCap evaluation benchmarks (SVAMP, BFCL, MBPP+, IFEval).

## Reproducibility caveat

The BottleCap Git SHA and generated file SHA-256 hashes are recorded. However, its preparation code currently calls Hugging Face `load_dataset` without pinned Hub revisions. For rigorous release-quality reproducibility, archive the generated dataset snapshot, persist its manifest, and later pin or mirror exact upstream dataset revisions.

## Next milestones

1. Config-driven SFT + LoRA training, validation loss, checkpointing, Qwen3 thinking-template support.
2. Evaluation adapter wrapping BottleCap's official `evaluate.py` + `compare.py`, with a T4-compatible vLLM backend and smoke test.
3. Experiment runner / tracking and analysis dashboards.
4. Optional GRPO with correctness and correctness+length reward, tested on held-out data.
