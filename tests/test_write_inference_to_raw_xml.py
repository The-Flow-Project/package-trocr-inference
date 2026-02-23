import unittest
import os
from unittest.mock import patch, MagicMock
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pathlib import Path

from flow_inference.write_inference_to_raw_xml import InferenceToRawXMLWriter


SAMPLE_XML = """<PcGts>
    <TextLine id="l1"/>
</PcGts>"""


class TestInferenceToRawXMLWriter(unittest.TestCase):

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

    # --------------------------------------------------
    # UNIT: build lookup
    # --------------------------------------------------
    def test_build_lookup(self):
        df = pd.DataFrame({
            "project_name": ["p1", "p1"],
            "filename": ["a.xml", "a.xml"],
            "line_id": ["l1", "l2"],
            "inference_test": ["A", "B"]
        })

        lookup = self.writer._build_lookup(df)

        self.assertEqual(
            lookup,
            {"p1": {"a.xml": {"l1": "A", "l2": "B"}}}
        )

    # --------------------------------------------------
    # UNIT: update df
    # --------------------------------------------------
    def test_update_df(self):
        df = pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML]
        })

        lookup = {"p1": {"a.xml": {"l1": "HELLO"}}}

        updated, cols = self.writer._update_df(df, lookup)

        self.assertEqual(len(cols), 1)
        self.assertIn("HELLO", updated.iloc[0][cols[0]])
        self.assertIn("<TextEquiv>", updated.iloc[0][cols[0]])

    # --------------------------------------------------
    # INTEGRATION: process_and_upload
    # --------------------------------------------------
    # --------------------------------------------------
    # INTEGRATION: process_and_upload
    # --------------------------------------------------
    @patch("flow_inference.write_inference_to_raw_xml.snapshot_download")
    @patch("flow_inference.write_inference_to_raw_xml.HfApi")
    def test_process_and_upload(self, mock_api, mock_snapshot):
        api = mock_api.return_value

        # fake snapshot folders
        fake_raw = Path("/tmp/raw")
        fake_inf = Path("/tmp/inf")
        fake_raw.mkdir(parents=True, exist_ok=True)
        fake_inf.mkdir(parents=True, exist_ok=True)

        # create fake parquet files
        raw_parquet = fake_raw / "data/train/p1/000.parquet"
        raw_parquet.parent.mkdir(parents=True, exist_ok=True)

        inf_parquet = fake_inf / "data/train/000.parquet"
        inf_parquet.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML]
        }).to_parquet(raw_parquet)

        pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "line_id": ["l1"],
            "inference_test": ["HELLO"]
        }).to_parquet(inf_parquet)

        # IMPORTANT: create README in raw snapshot so your code adds it to commit ops
        (fake_raw / "README.md").write_text(
            "dataset_info:\n"
            "  features:\n"
            "    - name: xml_content\n"
            "      dtype: string\n\n"
            "# Dataset Card for raw\n\n"
            "Usage:\n"
            'from datasets import load_dataset\n'
            'dataset = load_dataset("someone/raw")\n',
            encoding="utf-8"
        )

        # snapshot_download is called twice: inference first, then raw
        mock_snapshot.side_effect = [str(fake_inf), str(fake_raw)]

        # dataset_info sha is accessed for both repos
        api.dataset_info.return_value.sha = "fake_sha"

        self.writer.process_and_upload(output_repo=self.upload_repo)

        # ----------------------------------
        # assertions
        # ----------------------------------
        api.create_repo.assert_called_once()
        api.create_commit.assert_called_once()

        commit_ops = api.create_commit.call_args.kwargs["operations"]
        self.assertTrue(len(commit_ops) >= 1)

        # at least one parquet and README (since we created README.md)
        paths = [op.path_in_repo for op in commit_ops]
        self.assertTrue(any(p.endswith(".parquet") for p in paths))
        self.assertIn("README.md", paths)

    def test_process_and_upload_integration(self):
        """
        REAL integration test (HF Hub).
        Requires valid env vars.
        """
        if not self.upload_repo:
            self.skipTest("No HF repo configured")

        self.writer.process_and_upload(output_repo=self.upload_repo)

        api = HfApi()
        files = api.list_repo_files(
            self.upload_repo,
            repo_type="dataset",
            token=self.token
        )

        self.assertTrue(any(f.endswith(".parquet") for f in files))
        self.assertIn("README.md", files)


if __name__ == "__main__":
    unittest.main()
