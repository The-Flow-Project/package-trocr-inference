import unittest
import os
from unittest.mock import patch, MagicMock
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from flow_inference.write_inference_to_raw_xml import InferenceToRawXMLWriter

# A tiny XML sample used in update tests
SAMPLE_XML = """<PcGts>
    <TextLine id="l1"/>
</PcGts>"""

class TestInferenceToRawXMLWriter(unittest.TestCase):

    # -----------------------
    # Setup
    # -----------------------
    def setUp(self):
        load_dotenv()
        self.raw_repo = os.getenv("RAW_XML_REPO")
        self.inference_repo = os.getenv("INFERENCE_REPO")
        self.token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")
        self.upload_repo = os.getenv("UPLOAD_RAW_XML_REPO")
        self.writer = InferenceToRawXMLWriter(
            raw_xml_repo=self.raw_repo,
            inference_repo=self.inference_repo,
            token=self.token
        )

    # -----------------------
    # Test: load_datasets
    # -----------------------
    @patch("flow_inference.write_inference_to_raw_xml.load_dataset")
    def test_load_datasets(self, mock_load):
        """Ensure load_datasets selects default split correctly."""

        raw_default = MagicMock(name="raw_default")
        inf_default = MagicMock(name="inf_default")

        # simulate: load_dataset() returns dict with "default" split
        mock_load.side_effect = [
            {"default": raw_default},
            {"default": inf_default},
        ]

        self.writer.load_datasets()

        self.assertIs(self.writer.raw_dataset, raw_default)
        self.assertIs(self.writer.inference_dataset, inf_default)

    # -----------------------
    # Test: detect_inference_column
    # -----------------------
    def test_detect_inference_column(self):
        """Auto-detect a column named inference_*."""
        df = pd.DataFrame({
            "filename": ["a.xml"],
            "line_id": ["l1"],
            "inference_text": ["HELLO"]
        })

        self.writer.inference_dataset = Dataset.from_pandas(df)
        col = self.writer.detect_inference_column()

        self.assertEqual(col, "inference_text")

    # -----------------------
    # Test: build_inference_lookup (auto column detection)
    # -----------------------
    def test_build_inference_lookup(self):
        """Verify lookup groups per filename + line_id using auto column detection."""

        df = pd.DataFrame({
            "filename": ["a.xml", "a.xml", "b.xml"],
            "line_id": ["l1", "l2", "l1"],
            "inference_pred": ["A1", "A2", "B1"]
        })

        self.writer.inference_dataset = Dataset.from_pandas(df)

        lookup = self.writer.build_inference_lookup()

        expected = {
            "a.xml": {"l1": "A1", "l2": "A2"},
            "b.xml": {"l1": "B1"}
        }

        self.assertEqual(lookup, expected)

    # -----------------------
    # Test: update_raw_xml_dataset
    # -----------------------
    def test_update_raw_xml_dataset(self):
        """Ensure XML strings are updated with <TextEquiv> tags."""

        df_raw = pd.DataFrame({
            "filename": ["a.xml", "b.xml"],
            "xml": [SAMPLE_XML, SAMPLE_XML]
        })

        self.writer.raw_dataset = Dataset.from_pandas(df_raw)

        lookup = {
            "a.xml": {"l1": "HELLO"},
            "b.xml": {"l1": "WORLD"}
        }

        updated_df = self.writer.update_raw_xml_dataset(lookup)

        inference_cols = [c for c in updated_df.columns if c.startswith("inference_xml_")]
        self.assertEqual(len(inference_cols), 1)

        col = inference_cols[0]

        self.assertIn("HELLO", updated_df.iloc[0][col])
        self.assertIn("WORLD", updated_df.iloc[1][col])

        self.assertIn("<TextEquiv>", updated_df.iloc[0][col])
        self.assertIn("<Unicode>", updated_df.iloc[0][col])

    # -----------------------
    # Test: upload_updated_dataset
    # -----------------------
    @patch("flow_inference.write_inference_to_raw_xml.HfApi.create_repo")
    @patch("flow_inference.write_inference_to_raw_xml.HfApi.repo_info")
    @patch("flow_inference.write_inference_to_raw_xml.Dataset.push_to_hub")
    def test_upload_updated_dataset(self, mock_push, mock_repo_info, mock_create):
        """Ensure repo is created when missing and dataset is uploaded with correct token."""

        mock_repo_info.side_effect = HfHubHTTPError("Repo not found", response=None)

        df = pd.DataFrame({
            "filename": ["a.xml"],
            "xml": ["<PcGts/>"]
        })

        self.writer.upload_updated_dataset(df, repo_id=self.upload_repo)

        # ---- repo_info ----
        mock_repo_info.assert_called_once()
        repo_args, repo_kwargs = mock_repo_info.call_args

        # repo_id positional
        self.assertEqual(repo_args[0], self.upload_repo)

        # keyword args
        self.assertEqual(repo_kwargs["repo_type"], "dataset")
        self.assertEqual(repo_kwargs["token"], self.token)

        # ---- create_repo ----
        mock_create.assert_called_once()
        _, create_kwargs = mock_create.call_args
        self.assertEqual(create_kwargs["repo_id"], self.upload_repo)
        self.assertEqual(create_kwargs["repo_type"], "dataset")
        self.assertEqual(create_kwargs["token"], self.token)
        self.assertTrue(create_kwargs["private"])

        # ---- push_to_hub ----
        mock_push.assert_called_once()
        push_args, push_kwargs = mock_push.call_args

        self.assertEqual(push_args[0], self.upload_repo)
        self.assertEqual(push_kwargs["token"], self.token)

    # -----------------------
    # INTEGRATION TEST: Full pipeline
    # -----------------------
    def test_process_and_upload_integration(self):
        """
        Full integration test:
          1. Downloads both datasets
          2. Detects inference column
          3. Updates XML
          4. Uploads updated dataset to HF
        """
        print("self.upload_repo =", self.upload_repo)
        self.writer.process_and_upload(output_repo=self.upload_repo)

        api = HfApi()
        files = api.list_repo_files(self.upload_repo, repo_type="dataset", token=self.token)
        self.assertTrue(len(files) > 0)


if __name__ == "__main__":
    unittest.main()
