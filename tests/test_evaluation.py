"""Offline tests: no CUDA, dataset downloads, or model weights required."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from reasoning_opt.evaluation.common import discover_experiments, fingerprint, save_json
from reasoning_opt.evaluation.dev import parse_completion, equal_number
from reasoning_opt.evaluation.benchmark import make_hardware_wrapper
from reasoning_opt.training.batch import config_paths


def test_discovery_only_completed_and_explicit_selection():
    with TemporaryDirectory() as root:
        path = Path(root)
        ready = path / "gsm8k-sft-r8-2026"
        adapter = ready / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"abc")
        (ready / "summary.json").write_text("{}")
        (path / "broken").mkdir()
        cfg = {"experiments_root": root, "experiment_names": []}
        assert [x[0] for x in discover_experiments(cfg)] == ["base", "gsm8k-sft-r8-2026"]
        cfg["experiment_names"] = ["broken"]
        with pytest.raises(FileNotFoundError):
            discover_experiments(cfg)


def test_gpu_wrapper_changes_only_llm_initialization():
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        original = "x()\n" + "enable_lora=rank is not None, max_lora_rank=rank or 16)" + "\ny()"
        (folder / "evaluate.py").write_text(original)
        target = make_hardware_wrapper(folder, "TRITON_ATTN", "half")
        assert (folder / "evaluate.py").read_text() == original
        assert 'attention_backend="TRITON_ATTN"' in target.read_text()
        assert 'dtype="half"' in target.read_text()
        assert target == make_hardware_wrapper(folder, "TRITON_ATTN", "half")


def test_strict_parser_accepts_valid_thinking():
    txt = "<think>reason</think>\nFinal answer: 42<|im_end|>"
    pred, reasoning, boundary, error = parse_completion(txt, [100, 7, 200, 8, 300], 100, 200, 300, 384)
    assert (pred, reasoning, boundary, error) == ("42", 3, True, None)
    assert equal_number(pred, "42")


def test_strict_parser_rejects_unbalanced():
    txt = "reason</think>\nFinal answer: 42<|im_end|>"
    pred, _, _, error = parse_completion(txt, [7, 200, 8, 300], 100, 200, 300, 384)
    assert pred is None and error == "unbalanced_thinking"


def test_batch_config_rejects_duplicate_experiment_names(monkeypatch):
    with TemporaryDirectory() as temp:
        monkeypatch.chdir(temp)
        Path("x.json").write_text(json.dumps({"experiment_name": "same"}))
        Path("y.json").write_text(json.dumps({"experiment_name": "same"}))
        Path("batch.json").write_text(json.dumps({"configs": ["x.json", "y.json"]}))
        with pytest.raises(ValueError, match="Duplicate"):
            config_paths(Path("batch.json"))
