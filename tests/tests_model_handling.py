import unittest
from unittest.mock import patch, MagicMock
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from flow_inference.model_handling import ModelManager


class TestModelManager(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=True)
    def test_device_cuda_available(self, _mock_cuda):
        """Test device selection when CUDA is available."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, 'cuda')

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_device_cuda_and_mps_not_available(self, _mock_mps, _mock_cuda):
        """Test device selection when neither CUDA nor MPS are available."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, 'cpu')

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_device_mps_available(self, _mock_mps, _mock_cuda):
        """Test device selection when MPS is available but CUDA is not."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, 'mps')

    @patch('torch.cuda.is_available', return_value=False)
    def test_device_cuda_not_available(self, _mock_cuda):
        """Test device selection when CUDA is not available and no MPS."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, 'cpu')

    def test_device_force_cpu(self):
        """Test device selection when use_cuda is False."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, 'cpu')

    # --- Model Loading Tests ---

    @patch('transformers.VisionEncoderDecoderModel.from_pretrained')
    def test_load_model_success(self, mock_from_pretrained):
        """Test successful loading of a TrOCR model."""
        # Mock the model and device
        mock_model = MagicMock(spec=VisionEncoderDecoderModel)
        mock_from_pretrained.return_value = mock_model

        # Load the model
        model_manager = ModelManager()
        model_name = "microsoft/trocr-small-handwritten"
        loaded_model = model_manager.load_model(model_name)

        # Assertions
        mock_from_pretrained.assert_called_once_with(model_name)
        mock_model.to.assert_called_once_with(model_manager.device)
        self.assertIsInstance(loaded_model, VisionEncoderDecoderModel)

    @patch('transformers.VisionEncoderDecoderModel.from_pretrained', side_effect=Exception("Model loading failed"))
    def test_load_model_failure(self, mock_from_pretrained):
        """Test exception handling during model loading."""
        model_manager = ModelManager()
        with self.assertRaises(Exception) as context:
            model_manager.load_model("non-existent-model")

        self.assertIn("Model loading failed", str(context.exception))

    # --- Processor Loading Tests ---

    @patch('transformers.TrOCRProcessor.from_pretrained')
    def test_load_processor_success(self, mock_from_pretrained):
        """Test successful loading of a TrOCR processor."""
        # Mock the processor
        mock_processor = MagicMock(spec=TrOCRProcessor)
        mock_from_pretrained.return_value = mock_processor

        # Load the processor
        processor_name = "microsoft/trocr-base-handwritten"
        loaded_processor = ModelManager.load_processor(processor_name)

        # Assertions
        mock_from_pretrained.assert_called_once_with(processor_name)
        self.assertIsInstance(loaded_processor, TrOCRProcessor)

    @patch('transformers.TrOCRProcessor.from_pretrained', side_effect=Exception("Processor loading failed"))
    def test_load_processor_failure(self, mock_from_pretrained):
        """Test exception handling during processor loading."""
        with self.assertRaises(Exception) as context:
            ModelManager.load_processor("non-existent-processor")

        self.assertIn("Processor loading failed", str(context.exception))


if __name__ == '__main__':
    unittest.main()
