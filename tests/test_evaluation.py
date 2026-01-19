import os
import unittest
import pandas as pd
from dotenv import load_dotenv
from flow_inference.evaluation import Evaluation


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        """Set up an Evaluation instance configured for testing."""
        load_dotenv()
        self.download_repo_name = os.getenv("EVALUATION_REPO")
        self.write_token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")

        self.evaluator = Evaluation(
            download_repo_name=self.download_repo_name,
            hf_token=self.write_token,
            splits=None
        )

    # -------------------------------------------------------------
    # TEST: Selecting splits
    # -------------------------------------------------------------
    def test_select_splits_default(self):
        dfs = {
            "train": pd.DataFrame({"a": [1]}),
            "test": pd.DataFrame({"a": [2]}),
        }
        out = self.evaluator.select_splits(dfs)
        self.assertEqual(out["a"].iloc[0], 2)  # picks test first

    def test_select_splits_requested(self):
        self.evaluator.splits = ["train"]
        dfs = {
            "train": pd.DataFrame({"a": [10]}),
            "test": pd.DataFrame({"a": [20]}),
        }
        out = self.evaluator.select_splits(dfs)
        self.assertEqual(out["a"].iloc[0], 10)

    # -------------------------------------------------------------
    # TEST: Ground truth extraction
    # -------------------------------------------------------------
    def test_extract_ground_truth(self):
        df = pd.DataFrame({"text": ["hello", "world"]})
        gt = self.evaluator._extract_ground_truth(df)
        self.assertEqual(gt, ["hello", "world"])

    def test_extract_ground_truth_missing(self):
        df = pd.DataFrame({"other": [1]})
        with self.assertRaises(ValueError):
            self.evaluator._extract_ground_truth(df)

    # -------------------------------------------------------------
    # TEST: Finding the latest inference column
    # -------------------------------------------------------------
    def test_find_latest_inference_column(self):
        df = pd.DataFrame({
            "text": ["a"],
            "inference_2024-01-01_model_x": ["1"],
            "inference_2025-01-01_model_x": ["2"],
        })
        col = self.evaluator._find_latest_inference_column(df)
        self.assertEqual(col, "inference_2025-01-01_model_x")

    def test_find_latest_inference_missing(self):
        df = pd.DataFrame({"text": ["a"]})
        with self.assertRaises(ValueError):
            self.evaluator._find_latest_inference_column(df)

    # -------------------------------------------------------------
    # TEST: Hypothesis extraction
    # -------------------------------------------------------------
    def test_extract_hypothesis(self):
        df = pd.DataFrame({"col": ["a", "b"]})
        out = self.evaluator._extract_hypothesis(df, "col")
        self.assertEqual(out, ["a", "b"])

    def test_extract_hypothesis_missing(self):
        df = pd.DataFrame({"col": ["a"]})
        with self.assertRaises(ValueError):
            self.evaluator._extract_hypothesis(df, "nothing")

    # -------------------------------------------------------------
    # TEST: CER computation
    # -------------------------------------------------------------
    def test_compute_cer(self):
        cer = self.evaluator.compute_cer(["abc"], ["abd"])
        self.assertIsInstance(cer, float)

    # -------------------------------------------------------------
    # TEST: Output file creation
    # -------------------------------------------------------------
    def test_create_output_files(self):
        files = self.evaluator.create_output_files(
            groundtruth=["a", "b"],
            hypothesis=["a", "c"],
            cer_score=0.25
        )

        self.assertIn("gt.txt", files)
        self.assertIn("hypothesis.txt", files)
        self.assertIn("evaluation_report.json", files)

        self.assertIsInstance(files["gt.txt"], bytes)
        self.assertEqual(files["gt.txt"].decode(), "a\nb")

    def test_evaluation_fails_when_no_gt_inference_overlap(self):
        df = pd.DataFrame({
            "text": ["", "", "hello", "world"],
            "inference_x": ["abc", "def", "", ""],
        })

        evaluator = Evaluation("dummy", None)

        with self.assertRaises(RuntimeError):
            evaluator._filter_eval_rows(df, "inference_x")

    def test_evaluation_computes_cer_on_valid_overlap(self):
        df = pd.DataFrame({
            "text": ["hello", "", "world"],
            "inference_x": ["hallo", "xxx", "world"],
        })

        evaluator = Evaluation("dummy", None)

        df_eval = evaluator._filter_eval_rows(df, "inference_x")

        gt = evaluator._extract_ground_truth(df_eval)
        hyp = evaluator._extract_hypothesis(df_eval, "inference_x")

        cer = evaluator.compute_cer(gt, hyp)

        assert cer >= 0.0
        assert cer <= 1.0


if __name__ == "__main__":
    unittest.main()
