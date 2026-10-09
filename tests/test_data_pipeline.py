import unittest
from reasoning_opt.data.prepare import build_sft_split, q_from_bottlecap, solution_from_gsm8k, MATH_INSTRUCTION


class DatasetTests(unittest.TestCase):
    def test_solution(self):
        reasoning, answer = solution_from_gsm8k("We compute 2+3=5. <<2+3=5>>\nTherefore five.\n#### 5")
        self.assertIn("Therefore five", reasoning)
        self.assertNotIn("<<", reasoning)
        self.assertEqual(answer, "5")

    def test_join(self):
        bottlecap = [{"prompt": [{"role": "user", "content": f"What is 2+3?\n\n{MATH_INSTRUCTION}"}], "answer": "5", "task":"math", "source":"gsm8k"}]
        original = [{"question": "What is 2+3?", "answer": "2 + 3 = 5.\n#### 5"}]
        joined = build_sft_split(bottlecap, original, split="train")
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["reasoning"], "2 + 3 = 5.")
        self.assertEqual(joined[0]["split"], "train")
        self.assertEqual(q_from_bottlecap(bottlecap[0]), "What is 2+3?")

    def test_refuse_wrong_answer(self):
        bottlecap = [{"prompt": [{"role": "user", "content": f"What is 2+3?\n\n{MATH_INSTRUCTION}"}], "answer": "7", "task":"math", "source":"gsm8k"}]
        original = [{"question": "What is 2+3?", "answer": "2+3=5\n#### 5"}]
        with self.assertRaisesRegex(ValueError, "Answer mismatch"):
            build_sft_split(bottlecap, original, split="train")


if __name__ == "__main__":
    unittest.main()
