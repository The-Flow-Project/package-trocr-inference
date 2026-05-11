import unittest
from unittest.mock import MagicMock
from PIL import Image
from flow_inference.image_processing import ImageHandler
from flow_inference.create_trocr_dataset import TrOCRInferenceDataset


class TestTrOCRInferenceDataset(unittest.TestCase):

    def setUp(self):
        self.records = [
            {
                "image": {"bytes": b"\x89PNG\r\n\x1a\n...", "path": None},
                "filename": "test_doc_1.png",
                "region_id": "region_001",
                "line_id": "line_001",
            },
            {
                "image": {"bytes": b"\x89PNG\r\n\x1a\n...", "path": None},
                "filename": "test_doc_1.png",
                "region_id": "region_001",
                "line_id": "line_002",
            },
        ]

        self.mock_image_handler = MagicMock(spec=ImageHandler)
        self.mock_image_handler.handle_image.return_value = [0.1, 0.2, 0.3]

        self.dataset = TrOCRInferenceDataset(self.records, self.mock_image_handler)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        """Dataset __getitem__ should process image records correctly"""
        item = self.dataset[0]

        self.assertIn("pixel_values", item)
        self.assertIn("filename", item)
        self.assertIn("line_id", item)
        self.assertIn("region_id", item)

        self.assertEqual(item["pixel_values"], [0.1, 0.2, 0.3])
        self.assertEqual(item["filename"], "test_doc_1.png")
        self.assertEqual(item["line_id"], "line_001")
        self.assertEqual(item["region_id"], "region_001")

        self.mock_image_handler.handle_image.assert_called_once()
        called_record = self.mock_image_handler.handle_image.call_args[0][0]
        self.assertIsInstance(called_record["image"], (dict, Image.Image))


if __name__ == "__main__":
    unittest.main()
