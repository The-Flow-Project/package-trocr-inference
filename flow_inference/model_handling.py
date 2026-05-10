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
    def __init__(self):
        # Check for CUDA first
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        # Check for MPS if CUDA isn't available
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        # Fallback to CPU if neither CUDA nor MPS is available
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

    def load_model(self, model_name: str) -> PreTrainedModel:
        """
        Load the TrOCR model.

        :param model_name: Name or path of the TrOCR model.
        :return: Loaded VisionEncoderDecoderModel.
        """
        if not model_name:
            raise ValueError("Model name must not be empty.")

        try:
            logger.info(f"Loading model: {model_name}")
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            logger.info(f"Model loaded and moved to {self.device}")
            return model
        except (OSError, ValueError) as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

    @staticmethod
    def load_processor(processor_name: str) -> TrOCRProcessor:
        if not processor_name:
            raise ValueError("processor_name must not be empty.")

        errors = []

        for use_fast in (True, False):
            try:
                logger.info(
                    f"Loading processor: {processor_name} "
                    f"(use_fast={use_fast})"
                )
                processor = TrOCRProcessor.from_pretrained(
                    processor_name,
                    use_fast=use_fast,
                )
                logger.info(
                    f"Processor {processor_name} loaded "
                    f"(use_fast={use_fast})."
                )
                return processor
            except Exception as e:
                errors.append((use_fast, e))
                logger.warning(
                    f"Could not load processor '{processor_name}' "
                    f"with use_fast={use_fast}: {e}"
                )

        raise RuntimeError(
            f"Failed to load processor '{processor_name}' with both "
            f"fast and slow tokenizer paths:\n"
            + "\n".join(f"use_fast={uf}: {repr(err)}" for uf, err in errors)
        )
