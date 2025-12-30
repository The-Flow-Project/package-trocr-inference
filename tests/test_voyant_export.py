import os
import zipfile
import unittest
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from flow_inference.voyant_export import VoyantExporter


class TestVoyantExporter(unittest.TestCase):
    """
    Tests for VoyantExporter:
    - unit tests using in-memory DataFrames
    - one optional Hugging Face integration test
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setUp(self):
        load_dotenv()

        self.hf_dataset = os.getenv("HUGGINGFACE_TEST_UPLOAD_REPO_NAME")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN_READ")

        # temp dir
        self.tmp_dir = Path(
            os.environ.get("TMPDIR")
            or os.environ.get("TEMP")
            or "/tmp"
        )

    def test_export_creates_zip_with_documents(self):
        df = pd.DataFrame(
            {
                "filename": ["doc1", "doc1", "doc2"],
                "line_id": ["l1", "l2", "l1"],
                "inference_test": [
                    "Hello world",
                    "Second line",
                    "Another document",
                ],
            }
        )

        zip_path = self.tmp_dir / "voyant.zip"

        exporter = VoyantExporter()
        result = exporter.export(df, zip_path)

        self.assertTrue(result.exists())

        with zipfile.ZipFile(result) as zf:
            files = sorted(zf.namelist())

            self.assertEqual(files, ["doc1.txt", "doc2.txt"])

            doc1 = zf.read("doc1.txt").decode("utf-8")
            doc2 = zf.read("doc2.txt").decode("utf-8")

        self.assertEqual(doc1, "Hello world\nSecond line")
        self.assertEqual(doc2, "Another document")

    def test_export_with_line_ids(self):
        df = pd.DataFrame(
            {
                "filename": ["doc1", "doc1"],
                "line_id": ["l1", "l2"],
                "inference_test": ["First", "Second"],
            }
        )

        zip_path = self.tmp_dir / "voyant_lines.zip"

        exporter = VoyantExporter(include_line_ids=True)
        exporter.export(df, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            text = zf.read("doc1.txt").decode("utf-8")

        self.assertEqual(text, "[l1] First\n[l2] Second")

    def test_image_extension_is_stripped_from_document_name(self):
        df = pd.DataFrame(
            {
                "filename": ["doc_001.jpg", "doc_001.jpg"],
                "line_id": ["l1", "l2"],
                "inference_test": ["First line", "Second line"],
            }
        )

        zip_path = self.tmp_dir / "voyant_strip_ext.zip"

        exporter = VoyantExporter()
        exporter.export(df, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            files = zf.namelist()

        self.assertEqual(files, ["doc_001.txt"])

    # ------------------------------------------------------------------
    # Integration test (real Hugging Face dataset)
    # ------------------------------------------------------------------

    def test_export_from_huggingface(self):
        if not self.hf_dataset or not self.hf_token:
            self.skipTest("Missing Hugging Face credentials")

        zip_path = self.tmp_dir / "voyant_hf.zip"

        result = VoyantExporter.from_huggingface(
            dataset_name=self.hf_dataset,
            split="train",
            hf_token=self.hf_token,
            zip_path=zip_path,
        )

        self.assertTrue(result.exists())

        with zipfile.ZipFile(result) as zf:
            self.assertGreater(
                len(zf.namelist()),
                0,
                "ZIP archive is empty",
            )


if __name__ == "__main__":
    unittest.main()
