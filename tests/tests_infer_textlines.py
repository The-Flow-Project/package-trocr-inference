import os
import unittest
import torch
from torch.utils.data import DataLoader
from flow_inference.model_handling import ModelManager
from flow_inference.image_processing import ImageHandler
from flow_inference.create_trocr_dataset import TrOCRInferenceDataset
from flow_inference.infer_textlines import InferenceHandler


class TestInferenceHandler(unittest.TestCase):

    def setUp(self):
        model_manager = ModelManager()
        model = model_manager.load_model('microsoft/trocr-small-handwritten')
        processor = model_manager.load_processor("microsoft/trocr-base-handwritten")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.handler = InferenceHandler(model, processor, device)

        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        self.test_image_path_1 = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.0.png')
        self.test_image_path_2 = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.1.png')
        self.image_paths = [self.test_image_path_1, self.test_image_path_2]
        self.image_handler = ImageHandler(processor=processor,
                                          target_image_size=(384, 384))

    def test_inference(self):
        # Set up the inference dataset
        inference_dataset = TrOCRInferenceDataset(self.image_paths, self.image_handler)
        inference_dataloader = DataLoader(
            inference_dataset,
            collate_fn=self.handler.custom_collate_fn,
            batch_size=8,
            shuffle=False,
        )

        # Run inference
        predictions = self.handler.run_batch_inference(
            inference_dataloader=inference_dataloader,
            model=self.handler.model,
            device=self.handler.device,
            processor=self.handler.processor,
            max_new_tokens=100,
        )

        # Check that predictions is a list
        self.assertIsInstance(predictions, list, "Expected predictions to be a list")

        # Check the number of predictions matches the expected number
        expected_num_predictions = len(self.image_paths)
        self.assertEqual(len(predictions), expected_num_predictions,
                         f"Expected {expected_num_predictions} predictions, got {len(predictions)}")

        # Check the structure of predictions (should be strings with file name and predicted text)
        for prediction in predictions:
            self.assertIsInstance(prediction, str, "Expected each prediction to be a string")
            # Ensure the prediction format matches: 'file_name\tpredicted_text'
            self.assertIn('\t', prediction, "Prediction format is incorrect: missing tab separator")
            file_name, pred_text = prediction.split('\t', 1)
            self.assertIsInstance(file_name, str, "Expected file name to be a string")
            self.assertIsInstance(pred_text, str, "Expected predicted text to be a string")
            self.assertNotEqual(pred_text.strip(), "", "Predicted text is an empty string")


if __name__ == '__main__':
    unittest.main()
