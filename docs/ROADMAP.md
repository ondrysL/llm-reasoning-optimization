# Implementation milestones

- [x] Independent Python project layout and reproducible BottleCap checkout.
- [x] Unmodified original BottleCap data snapshot for GRPO.
- [x] GSM8K original reasoning joined using BottleCap split, validated and fingerprinted.
- [x] Offline unit tests for joining and answer mismatches.
- [ ] Trainer: config + deterministic subset selection + SFT LoRA on Qwen3-0.6B.
- [ ] Notebook: configure training run and visualize train/validation loss.
- [ ] Evaluator: BottleCap baseline + LoRA, separate development vs final benchmark.
- [ ] Run baseline smoke test with T4-compatible vLLM (FlashInfer caused CUDA failure before).
- [ ] Runner: each run stores Git SHA, model revision, dataset manifest SHA, config, training logs, checkpoints.
- [ ] GRPO with correctness reward, then correctness + length reward without skipping thinking.
- [ ] Official benchmark suite once model selection is complete.
