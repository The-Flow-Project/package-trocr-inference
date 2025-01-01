# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from typing import Union
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, PreTrainedModel
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class ModelManager:
    """Manages the Loading of the TrOCR model and processor"""
    def __init__(self, use_cuda: bool):
        # Check for CUDA first
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        # Check for MPS if CUDA isn't available
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        # Fallback to CPU if neither CUDA nor MPS is available
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

    def load_model(self, model_name: str) -> Union[PreTrainedModel, None]:
        """
        Load the TrOCR model.

        :param: model_name: Name or path of the TrOCR model.
        :return: PreTrainedModel: The loaded model (a VisionEncoderDecoderModel) or None if it was not loaded.
        """
        if model_name:
            try:
                logger.info(f"Loading model: {model_name}")
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
                model.to(self.device)
                logger.info(f"Model loaded and moved to {self.device}")
                return model
            except (OSError, ValueError) as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                return None
        else:
            logger.error(f"The model with name '{model_name}' could not be loaded.")
            raise ValueError(f"The model with name '{model_name}' could not be loaded.")

    @staticmethod
    def load_processor(processor_name: str) -> Union[TrOCRProcessor, None]:
        """
        Load the TrOCR processor.

        :param: processor_name: Name or path of the TrOCR processor.
        :return: TrOCRProcessor: The loaded TrOCR processor or None (in case of failure).
        """
        if processor_name:
            try:
                logger.info(f"Loading processor: {processor_name}")
                processor = TrOCRProcessor.from_pretrained(processor_name)
                logger.info(f"Processor {processor_name} loaded.")
                return processor
            except (OSError, ValueError) as e:
                logger.error(f"Failed to load processor '{processor_name}': {e}")
                return None
        else:
            logger.error(f"The processor with name '{processor_name}' could not be loaded.")
            raise ValueError(f"The processor with name '{processor_name}' could not be loaded.")
