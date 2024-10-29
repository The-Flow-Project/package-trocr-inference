import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


class DeviceManager:
    """Manages the device selection (CPU or CUDA)."""
    def __init__(self, use_cuda: bool = True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def get_device(self):
        return self.device


class ModelLoader:
    """Handles the loading of the VisionEncoderDecoderModel."""
    def __init__(self, model_name: str, device):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        """Loads the model if it hasn't been loaded already, and moves it to the specified device."""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"Model loaded and moved to {self.device}")
        return self.model


class TrOCRModelHandler:
    """Central handler for loading the model and managing the device."""
    def __init__(self, model_name: str, use_cuda: bool = True):
        self.device_manager = DeviceManager(use_cuda)
        self.device = self.device_manager.get_device()

        self.model_loader = ModelLoader(model_name, self.device)
        self.model = self.model_loader.load_model()

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device


class TrOCRProcessorHandler:
    """Manages the loading of the TrOCRProcessor."""
    def __init__(self, processor_name: str = 'microsoft/trocr-base-handwritten'):
        self.processor_name = processor_name
        self.processor = self._load_processor()

    def _load_processor(self):
        """Loads the TrOCRProcessor."""
        print(f"Loading processor: {self.processor_name}")
        processor = TrOCRProcessor.from_pretrained(self.processor_name)
        print(f"Processor {self.processor_name} loaded.")
        return processor

    def get_processor(self):
        return self.processor
