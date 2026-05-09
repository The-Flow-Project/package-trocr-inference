import unittest
import torch
from torch.utils.data import DataLoader
from flow_inference.model_handling import ModelManager
from flow_inference.image_processing import ImageHandler
from flow_inference.create_trocr_dataset import TrOCRInferenceDataset
from flow_inference.infer_textlines import InferenceHandler
from flow_inference.data_handling import HuggingFaceDataHandler
from dotenv import load_dotenv
import os


class TestInferenceHandler(unittest.TestCase):

    def setUp(self):
        """Set up model, processor, and small dataset from Hugging Face."""
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN_READ")
        hf_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")

        if not hf_token or not hf_repo_name:
            self.skipTest("Missing Hugging Face token or repo name in .env")

        # STEP 1: Load dataset from Hugging Face
        handler = HuggingFaceDataHandler(
            dataset_name=hf_repo_name,
            huggingface_token=hf_token,
        )
        handler.download_hf_dataset()
        dfs = handler.to_dataframe()
        records_dict = handler.convert_to_list_of_dicts(dfs)

        # pick a split to test (train preferred, otherwise first available)
        if "train" in records_dict:
            split_name = "train"
        else:
            split_name = next(iter(records_dict.keys()))

        all_records = records_dict[split_name]
        if not all_records:
            self.skipTest(f"No records found in split '{split_name}'")

        # take 2 samples for speed
        self.records = all_records[:2]
        assert "line_id" in self.records[0], "Test records must include line_id"

        image_obj = self.records[0].get("image")
        if isinstance(image_obj, dict) and "bytes" in image_obj:
            for r in self.records:
                r["image"] = r["image"]["bytes"]

        # validate structure
        assert isinstance(self.records[0]["image"], (bytes, bytearray)), "Expected image bytes in record"

        # STEP 2: Initialize model, processor, and inference handler
        model_manager = ModelManager()
        model = model_manager.load_model("microsoft/trocr-small-handwritten")
        processor = model_manager.load_processor("microsoft/trocr-base-handwritten")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.handler = InferenceHandler(model, processor, device)
        self.image_handler = ImageHandler(processor=processor, target_image_size=(384, 384))

        # mock model.generate to skip real inference
        self.handler.model.generate = lambda x, **_: torch.zeros((x.size(0), 5), dtype=torch.long)

    def test_collate_fn_includes_line_ids(self):
        inference_dataset = TrOCRInferenceDataset(self.records, self.image_handler)
        dataloader = DataLoader(
            inference_dataset,
            collate_fn=self.handler.custom_collate_fn,
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(dataloader))

        self.assertIn("pixel_values", batch)
        self.assertIn("filenames", batch)
        self.assertIn("line_ids", batch)

        self.assertEqual(len(batch["line_ids"]), len(self.records))

    def test_inference(self):
        """Run inference on a small batch of in-memory HF records and validate structure."""
        inference_dataset = TrOCRInferenceDataset(self.records, self.image_handler)
        inference_dataloader = DataLoader(
            inference_dataset,
            collate_fn=self.handler.custom_collate_fn,
            batch_size=2,
            shuffle=False,
        )

        predictions = self.handler.run_batch_inference(
            inference_dataloader=inference_dataloader,
            model=self.handler.model,
            device=self.handler.device,
            processor=self.handler.processor,
            max_new_tokens=10,  # small for speed
        )

        self.assertIsInstance(predictions, list, "Expected predictions to be a list")

        expected_num_predictions = len(self.records)
        self.assertEqual(
            len(predictions),
            expected_num_predictions,
            f"Expected {expected_num_predictions} predictions, got {len(predictions)}",
        )

        valid_keys = {
            (
                str(r["project_name"]),
                str(r["filename"]),
                str(r["line_id"]),
            )
            for r in self.records
        }

        for prediction in predictions:
            self.assertIsInstance(
                prediction,
                tuple,
                "Expected each prediction to be a tuple",
            )
            self.assertEqual(
                len(prediction),
                4,
                "Expected prediction tuple: (project_name, filename, line_id, predicted_text)",
            )

            project, filename, line_id, pred_text = prediction

            self.assertIsInstance(project, str, "Expected project to be a string")
            self.assertIsInstance(filename, str, "Expected filename to be a string")
            self.assertIsInstance(line_id, str, "Expected line_id to be a string")
            self.assertIsInstance(pred_text, str, "Expected predicted text to be a string")

            self.assertIn(
                (project, filename, line_id),
                valid_keys,
                "Inference output (project, filename, line_id) not found in input records",
            )


if __name__ == "__main__":
    unittest.main()
