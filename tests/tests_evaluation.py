import asyncio
import json
import os
import unittest
from pathlib import Path
from flow_inference.evaluation import Evaluation
from flow_inference.inference import Inference


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Set up directories and test files
        self.test_repo_name = "github-actions-test-organisation/inference_test"
        self.test_directory = os.path.join("..", "test_data", "data")
        self.test_repo_base_path = "github-actions-test-organisation___inference_test"
        self.test_in_path = "fetched"
        self.test_inference_results_folder = "inference_results"
        self.test_preprocessed_path = "preprocessed"
        self.test_evaluation_results_folder = "evaluation_results"
        self.test_out_path_evaluation = os.path.join(self.test_directory,
                                                     self.test_repo_base_path,
                                                     self.test_evaluation_results_folder)
        self.model_name = "microsoft/trocr-large-handwritten"
        self.hf_url = "https://huggingface.co/microsoft/trocr-large-handwritten"

        self.test_image_file = os.path.join(
            self.test_directory,
            "github-actions-test-organisation___inference_test",
            "preprocessed",
            "1155140_0001_47389007.line_1663284857722_42.JPG"
        )
        self.test_xml_file_with_text = os.path.join(
            self.test_directory,
            "github-actions-test-organisation___inference_test",
            "fetched",
            "567137_0001_23153854.xml"
        )
        self.test_xml_file_without_text = os.path.join(
            self.test_directory,
            "github-actions-test-organisation___inference_test",
            "fetched",
            "1155140_0001_47389007.xml"
        )
        self.test_file_inference = os.path.join(
            self.test_directory,
            "github-actions-test-organisation___inference_test",
            "inference_results",
            "567137_0001_23153854.xml"
        )
        self.inference = Inference(
            process_id=self.test_repo_base_path,
            repo_name=self.test_repo_name,
            directory=self.test_directory,
            github_access_token=None,
            trocr_model="microsoft/trocr-small-handwritten"
        )
        self.evaluator = Evaluation(
            process_id=self.test_repo_base_path,
            repo_name=self.test_repo_name,
            github_access_token=None,
            directory=self.test_directory,
            model_name=self.model_name,
            hf_url=self.hf_url
        )

    def test_process_ground_truth_with_text(self):
        """Test extracting ground truth from a real XML file."""
        self.assertTrue(os.path.exists(self.test_xml_file_with_text), "Test XML file does not exist.")
        result = self.evaluator._process_ground_truth([self.test_xml_file_with_text])
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "No text lines extracted from XML.")

    def test_process_ground_truth_without_text(self):
        """Test processing an XML file with no text, should return empty list."""
        self.assertTrue(os.path.exists(self.test_xml_file_without_text), "Test XML file does not exist.")
        result = self.evaluator._process_ground_truth([self.test_xml_file_without_text])
        self.assertEqual(result, [])

    def test_write_lines_to_file_and_count_lines(self):
        lines = ["line1", "line2", "line3"]
        temp_file = os.path.join(self.test_out_path_evaluation, "write_count_test.txt")
        Evaluation.write_lines_to_file(lines, temp_file)

        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        self.assertEqual(content, "\n".join(lines))

        count = Evaluation.count_lines(temp_file)
        self.assertEqual(count, 3)

    def test_generate_evaluation_report(self):
        """Test report generation and returned file path validation."""
        gt_file = os.path.join(self.test_out_path_evaluation, "gt.txt")
        hypothesis_file = os.path.join(self.test_out_path_evaluation, "hypothesis.txt")

        # Sample test data
        with open(gt_file, "w", encoding="utf-8") as f:
            f.write("line1\nline2\n")
        with open(hypothesis_file, "w", encoding="utf-8") as f:
            f.write("line1\nline2\nline3\n")

        # Mock model info
        cer_score = 0.05

        # Generate report
        report_file_path = self.evaluator._generate_evaluation_report(
            model_name=self.model_name,
            hf_url=self.hf_url,
            gt_file=gt_file,
            hypothesis_file=hypothesis_file,
            cer_score=cer_score
        )

        # Validate the returned file path
        self.assertEqual(report_file_path, os.path.join(self.test_out_path_evaluation, "evaluation_report.json"))

        # Validate that the report file exists
        self.assertTrue(os.path.exists(report_file_path), "Evaluation report file was not saved.")

        # Read and check JSON contents
        with open(report_file_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        self.assertEqual(report_data["model_name"], self.model_name)
        self.assertEqual(report_data["hf_url"], self.hf_url)
        self.assertEqual(report_data["num_lines_gt"], 2)
        self.assertEqual(report_data["num_lines_hypothesis"], 3)
        self.assertEqual(report_data["cer_score"], cer_score)

    def test_calculate_cer(self):
        gt_lines = ["hello", "world"]
        inferred_lines = ["helo", "world!"]
        cer_value = Evaluation._compute_cer(gt_lines, inferred_lines)
        self.assertIsInstance(cer_value, float)

    def test_perform_evaluation(self):
        """Test the perform_evaluation method end-to-end using real files and the expected folder structure."""

        # Set required attributes for report generation.
        self.evaluator.model_name = "dummy-model"
        self.evaluator.hf_url = "https://dummy.url"

        # Prepare dummy text data for ground truth and hypothesis.
        gt_text = "this is a test\nhello world"
        hypothesis_text = "this is test\nhello wrld"

        self.evaluator._fetch_xml_files = lambda: ["dummy.xml"]
        self.evaluator._process_ground_truth = lambda files: gt_text.split("\n")
        self.evaluator._perform_inference = lambda: asyncio.sleep(0, hypothesis_text.split("\n"))
        self.evaluator._compute_cer = lambda gt, hyp: 0.1
        self.evaluator._push_saved_results = lambda gt, hyp, report: None

        # Run evaluation process
        asyncio.run(self.evaluator.perform_evaluation())

        # The evaluation report should be saved
        report_file = Path(self.evaluator.out_path) / "evaluation_report.json"
        assert report_file.exists(), f"Evaluation report was not created at {report_file}"

        # Read the report and check that the CER score is correctly included.
        report_content = report_file.read_text(encoding="utf-8")
        assert '"cer_score": 0.1' in report_content, "CER score in report is incorrect."


if __name__ == "__main__":
    unittest.main()
