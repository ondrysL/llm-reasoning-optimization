"""Tests independent of GPUs or downloaded Hugging Face models."""
import re
import unittest

from reasoning_opt.training.sft import make_target, tokenize_row, validate_overlap


class FakeTokenizer:
    marker_ids = {"<think>": 100, "</think>": 101, "<|im_end|>": 102}
    unk_token_id = -1

    def __init__(self):
        self.vocab = dict(self.marker_ids)

    def convert_tokens_to_ids(self, token):
        return self.marker_ids.get(token, -1)

    def apply_chat_template(self, prompt, **kwargs):
        return f"<|im_start|>user\n{prompt[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, add_special_tokens=False):
        parts = []
        for fragment in re.split(r"(<think>|</think>|<\|im_end\|>)", text):
            if fragment in self.marker_ids:
                parts.append(fragment)
            else:
                parts.extend(fragment.split())
        indices = []
        for part in parts:
            if part not in self.vocab:
                self.vocab[part] = len(self.vocab) + 200
            indices.append(self.vocab[part])
        return {"input_ids": indices}


class SFTTests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.row = {"id": "example", "source": "gsm8k", "answer": "15",
                    "reasoning": "10 + 5 = 15.",
                    "prompt": [{"role": "user", "content": "What is 10 + 5?"}]}

    def test_first_supervised_token_is_opening_think(self):
        example, skipped = tokenize_row(self.row, self.tok, 128)
        self.assertIsNone(skipped)
        labels = [x for x in example["labels"] if x != -100]
        self.assertEqual(labels[0], self.tok.convert_tokens_to_ids("<think>"))
        self.assertEqual(labels.count(self.tok.convert_tokens_to_ids("</think>")), 1)
        self.assertEqual(labels[-1], self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.assertEqual(len(example["labels"]), len(example["input_ids"]))

    def test_overlength_is_skipped_not_truncated(self):
        example, reason = tokenize_row(self.row, self.tok, 4)
        self.assertIsNone(example)
        self.assertEqual(reason, "too_long")

    def test_answer_and_reasoning_are_retained(self):
        target = make_target(self.row, self.tok)
        self.assertTrue(target.startswith("<think>\n10 + 5 = 15."))
        self.assertIn("</think>\n\nFinal answer: 15", target)

    def test_detects_validation_overlap(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_overlap([{"id": "a"}], [{"id": "a"}])


if __name__ == "__main__":
    unittest.main()
