import asyncio
import os
import unittest
from unittest.mock import patch

from flow_inference.inference import Inference


class TestInference(unittest.TestCase):

    def setUp(self):
        # Set up directories and test files
        self.test_repo_name = "github-actions-test-organisation/inference_test"
        self.test_directory = os.path.join("..", "test_data", "data")
        self.test_repo_base_path = "github-actions-test-organisation___inference_test"
        self.test_in_path = "fetched"
        self.test_out_path = "inference_results"
        os.makedirs(self.test_out_path, exist_ok=True)
        self.test_preprocessed_path = "preprocessed"

        self.test_image_file = os.path.join(
            self.test_directory,
            "github-actions-test-organisation___inference_test",
            "preprocessed",
            "1155140_0001_47389007.line_1663284857722_42.JPG"
        )
        self.test_xml_file = os.path.join(
            self.test_directory,
            "github-actions-test-organisation___inference_test",
            "fetched",
            "1155140_0001_47389007.xml"
        )
        self.inference = Inference(
            process_id="1234",
            repo_name=self.test_repo_name,
            directory=self.test_directory,
            repo_folder="",
            github_access_token=None,
            in_path=self.test_in_path,
            out_path=self.test_out_path,
            use_cuda=False,
            trocr_model="microsoft/trocr-small-handwritten",
            trocr_processor="microsoft/trocr-base-handwritten",
        )

    def test_get_image_files(self):
        image_files = self.inference.get_image_files()
        self.assertIn(self.test_image_file, image_files)

    def test_fetch_xml_files(self):
        fetched_files, failed_files = self.inference.fetch_xml_files()

        with open(self.test_xml_file, 'r') as test_file:
            test_content = test_file.read()

        fetched_contents = []
        for fetched_file in fetched_files:
            with open(fetched_file, 'r') as file:
                fetched_contents.append(file.read())

        self.assertIn(test_content, fetched_contents)
        self.assertEqual(len(failed_files), 0)

    def test_perform_inference(self):
        self.inference.perform_inference()

        # Check that results are saved
        inferred_txt_file = os.path.join(self.test_directory,
                                         self.test_repo_base_path,
                                         self.test_out_path,
                                         "1155140_0001_47389007.txt")
        self.assertTrue(os.path.exists(inferred_txt_file))

        inferred_xml_file = os.path.join(self.test_directory,
                                         self.test_repo_base_path,
                                         self.test_out_path,
                                         "1155140_0001_47389007.xml")
        self.assertTrue(os.path.exists(inferred_xml_file))

    def test_run_inference(self):
        image_files = self.inference.get_image_files()
        results = asyncio.run(self.inference.run_inference(image_files))
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

    def test_write_to_text_files(self):
        inferred_lines = {"test_image.jpg": "This is a test inference result."}
        self.inference.write_to_text_files(inferred_lines)

        inferred_txt_file = os.path.join(self.test_directory,
                                         self.test_repo_base_path,
                                         self.test_out_path,
                                         "test_image.txt")
        self.assertTrue(os.path.exists(inferred_txt_file))
        with open(inferred_txt_file, "r") as f:
            content = f.read()
        self.assertEqual(content, "test_image.jpg This is a test inference result.")

    def test_write_to_xml_files(self):
        inferred_lines = {"TextRegion_1663284780281_22l2": "This is a test inference result."}
        self.inference.write_to_xml_files(inferred_lines, [self.test_xml_file])

        inferred_xml_file = os.path.join(self.test_directory,
                                         self.test_repo_base_path,
                                         self.test_out_path,
                                         "1155140_0001_47389007.xml")
        self.assertTrue(os.path.exists(inferred_xml_file))
        with open(inferred_xml_file, "r") as f:
            content = f.read()
        self.assertIn("This is a test inference result.", content)


if __name__ == "__main__":
    unittest.main()
