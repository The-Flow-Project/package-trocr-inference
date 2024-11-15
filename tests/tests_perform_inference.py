import os
import unittest
import torch
from flow_inference.model_handling import TrOCRModelHandler, TrOCRProcessorHandler
from flow_inference.perform_inference import InferenceDataset, InferenceHandler
from flow_inference.image_processing import ImageProcessor


class TestInferenceDataset(unittest.TestCase):

    def setUp(self):
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        self.test_image_path = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.0.png')
        self.image_paths = [self.test_image_path, self.test_image_path]
        processor = TrOCRProcessorHandler("microsoft/trocr-base-handwritten")
        image_processor = ImageProcessor(processor)
        self.dataset = InferenceDataset(self.image_paths, image_processor)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        item = self.dataset[0]
        self.assertIn('pixel_values', item)
        self.assertIn('label', item)
        self.assertEqual(item['label'], '1_0054.0')


class TestInferenceHandler(unittest.TestCase):

    def setUp(self):
        model = TrOCRModelHandler('microsoft/trocr-small-handwritten').get_model()
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        self.test_image_path_1 = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.0.png')
        self.test_image_path_2 = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.1.png')
        self.image_paths = [self.test_image_path_1, self.test_image_path_2]
        self.trocr_processor = TrOCRProcessorHandler("microsoft/trocr-base-handwritten").get_processor()
        image_processor = ImageProcessor(self.trocr_processor)
        self.dataset = InferenceDataset(self.image_paths, image_processor)
        self.handler = InferenceHandler(model, self.trocr_processor)

    def test_predict_single_image(self):
        pixel_values = torch.rand((1, 3, 384, 384))
        prediction = self.handler.predict_single_image(pixel_values)

        # Check that the prediction is not None
        self.assertIsNotNone(prediction, "Prediction is None")

        # Check that the prediction is not an empty string
        self.assertNotEqual(prediction.strip(), "", "Prediction is an empty string")

        # Check that the prediction is of type string
        self.assertIsInstance(prediction, str, "Prediction is not a string")

        # Check that the length of the prediction is within a reasonable range
        self.assertGreater(len(prediction.strip()), 10, "Prediction is too short (less than 10 characters)")
        self.assertLess(len(prediction.strip()), 500, "Prediction is too long (more than 500 characters)")

        # Check that the prediction only contains ASCII characters (reasonable for text output)
        self.assertTrue(all(ord(c) < 128 for c in prediction), "Prediction contains non-ASCII characters")

    def test_predict_batch(self):
        predictions = self.handler.predict_batch(self.dataset)

        # Check that predictions is a dictionary
        self.assertIsInstance(predictions, dict, "Expected predictions to be a dictionary")

        # Check the number of predictions matches the expected number
        expected_num_predictions = 2
        self.assertEqual(len(predictions), expected_num_predictions,
                         f"Expected {expected_num_predictions} predictions, got {len(predictions)}")

        # Check the structure of predictions (keys are labels, values are predicted text strings)
        for label, prediction in predictions.items():
            self.assertIsInstance(label, str, f"Expected label to be a string, got {type(label)}")
            self.assertIsInstance(prediction, str, f"Expected prediction to be a string, got {type(prediction)}")


if __name__ == '__main__':
    unittest.main()
