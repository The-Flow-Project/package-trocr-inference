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
        """Set up model, processor, and small in-memory dataset from Hugging Face."""
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN_READ")
        hf_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")

        if not hf_token or not hf_repo_name:
            self.skipTest("Missing Hugging Face token or repo name in .env")

        # STEP 1: Load dataset from Hugging Face (in-memory images)
        handler = HuggingFaceDataHandler(
            dataset_name=hf_repo_name,
            huggingface_token=hf_token,
            split="train"
        )
        handler.download()
        handler.to_dataframe()
        self.records = handler.convert_df_into_dict_list()[:2]  # just take 2 samples for speed

        # ✅ Minimal fix — unwrap Hugging Face image dicts
        image_obj = self.records[0]["image"]
        if isinstance(image_obj, dict) and "bytes" in image_obj:
            for r in self.records:
                r["image"] = r["image"]["bytes"]

        # Validate structure
        assert isinstance(self.records[0]["image"], (bytes, bytearray)), "Expected image bytes in record"

        # STEP 2: Initialize model, processor, and inference handler
        model_manager = ModelManager()
        model = model_manager.load_model("microsoft/trocr-small-handwritten")
        processor = model_manager.load_processor("microsoft/trocr-base-handwritten")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.handler = InferenceHandler(model, processor, device)
        self.image_handler = ImageHandler(processor=processor, target_image_size=(384, 384))

        # ⚡️ Speed up test: mock model.generate to skip real inference
        self.handler.model.generate = lambda x, **_: torch.zeros((x.size(0), 5), dtype=torch.long)

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

        for prediction in predictions:
            self.assertIsInstance(prediction, str, "Expected each prediction to be a string")
            self.assertIn("\t", prediction, "Prediction format is incorrect: missing tab separator")
            file_name, pred_text = prediction.split("\t", 1)
            self.assertIsInstance(file_name, str, "Expected file name to be a string")
            self.assertIsInstance(pred_text, str, "Expected predicted text to be a string")


if __name__ == "__main__":
    unittest.main()
