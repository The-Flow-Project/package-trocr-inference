import unittest
from unittest.mock import patch
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from flow_inference.model_handling import DeviceManager, ModelLoader, TrOCRProcessorHandler, TrOCRModelHandler


class TestDeviceManager(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=True)
    def test_device_manager_cuda_available(self, _):
        # Test with CUDA available and use_cuda=True
        device_manager = DeviceManager(use_cuda=True)
        self.assertEqual(device_manager.get_device().type, 'cuda')

        # Test with CUDA available but use_cuda=False
        device_manager = DeviceManager(use_cuda=False)
        self.assertEqual(device_manager.get_device().type, 'cpu')

    @patch('torch.cuda.is_available', return_value=False)
    def test_device_manager_cuda_not_available(self, _):
        # Test with CUDA not available and use_cuda=True
        device_manager = DeviceManager(use_cuda=True)
        self.assertEqual(device_manager.get_device().type, 'cpu')


class TestModelLoader(unittest.TestCase):
    def test_load_model(self):
        # Load a small model to avoid excessive loading time
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_loader = ModelLoader('microsoft/trocr-small-handwritten', device)

        # Before loading the model, it should be None
        self.assertIsNone(model_loader.model)

        # Load the model
        model = model_loader.load_model()

        # After loading, the model should not be None
        self.assertIsNotNone(model)
        self.assertIsInstance(model, VisionEncoderDecoderModel)

        # Check that the model is moved to the correct device
        self.assertEqual(next(model.parameters()).device, device)


class TestTrOCRModelHandler(unittest.TestCase):
    def test_model_handler_initialization(self):
        # Initialize the TrOCRModelHandler with a small model
        model_handler = TrOCRModelHandler('microsoft/trocr-small-handwritten', use_cuda=torch.cuda.is_available())

        # Check if the model is loaded and not None
        model = model_handler.get_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(model, VisionEncoderDecoderModel)

        # Check if the correct device is set
        device = model_handler.get_device()
        if torch.cuda.is_available():
            self.assertEqual(device.type, 'cuda')
        else:
            self.assertEqual(device.type, 'cpu')


class TestTrOCRProcessorHandler(unittest.TestCase):
    def test_processor_handler_initialization(self):
        # Initialize the processor handler with the default processor name
        processor_handler = TrOCRProcessorHandler('microsoft/trocr-base-handwritten')

        # Check if the processor is loaded and not None
        processor = processor_handler.get_processor()
        self.assertIsNotNone(processor)
        self.assertIsInstance(processor, TrOCRProcessor)


if __name__ == '__main__':
    unittest.main()
