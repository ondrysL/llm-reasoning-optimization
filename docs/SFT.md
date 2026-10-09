# SFT + LoRA training

This project trains **Qwen3-0.6B** on the **original human-written reasoning from GSM8K**. It uses the immutable BottleCapAI training/validation split prepared by the data pipeline. No teacher LLM and no shortened-reasoning labels are used.

## In Colab

First run the setup and data cells in `notebooks/experiments.ipynb`. Then run the SFT cells:

```bash
python -m pip install -e '.[training]'
python -u -m reasoning_opt.training.sft --config configs/sft.json --preflight-only
python -u -m reasoning_opt.training.sft --config configs/sft.json
```

**Requires a CUDA GPU**, with defaults targeting a Colab T4. The preflight command needs no CUDA GPU; it checks data splits, sequence length and the `<think>`/`</think>` target structure without loading the model weights.

## Config

Edit `configs/sft.json` or copy it to a new config file per experiment. Default is 1,000 training questions, 100 validation questions, 768 tokens, 1 epoch, LoRA r=8 alpha=16, fp16 with effective batch size 4. It automatically takes the first `train_examples` examples **that fit** the length cap after a fixed-seed shuffle. Overlength examples are skipped, never cut mid-answer; counts are printed. For larger experiments, set 3,000 training examples and 200 validation examples, then consider increasing sequence length if VRAM permits.

- `data.*`: source dataset, train/validation counts, max length.
- `lora.*`: adapter rank, alpha, dropout and target modules.
- `training.*`: optimizer, learning rate, epochs or max_steps, effective batch, seed, logging / eval / save frequency.
- `output.root_dir`: parent directory on Drive for a uniquely timestamped run.

## Live logging and artifacts

Every `logging_steps` optimizer updates, stdout prints step/max_steps, loss, learning rate, GPU memory and elapsed time; `eval_steps` logs validation loss, while `save_steps` saves resumable Trainer checkpoints.

Google Drive: `MyDrive/llm-reasoning/experiments/<name>-<UTC timestamp>/`

```
config.json
reproducibility.json
metrics.jsonl
train.log
summary.json
adapter/
  adapter_config.json
  adapter_model.safetensors
tensorboard/
checkpoints/
generation_samples.jsonl   # optional tiny qualitative check
```

The `adapter/` folder is the PEFT LoRA adapter and tokenizer files, **not merged model weights**. To evaluate, load base Qwen3-0.6B together with this adapter. Existing BottleCap eval integration will be added separately.

**Validation loss is teacher-forced cross entropy, not generated-answer accuracy.** The optional post-training generations are a *qualitative format smoke test*, not a benchmark. For meaningful accuracy and reasoning-token measurements, use an independent evaluator later.

If Colab disconnects, logged metrics and periodic checkpoints persist on Drive. Automatic resume from an interrupted checkpoint is **not implemented** in this first version; rerunning creates a new experiment. We can add explicit `resume` later.

## Format check

The first **supervised** completion token is the special `<think>` token. The prompt comes from `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`. If it already contains `<think>`, preflight stops instead of silently teaching malformed thinking. We supervise only target tokens; prompt labels are masked as `-100`.

## TensorBoard

In Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/llm-reasoning/experiments
```

TensorBoard shows training/eval loss and learning rate curves across runs.
