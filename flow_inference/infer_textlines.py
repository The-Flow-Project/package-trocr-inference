# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import os
from typing import List, Union, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, TrOCRProcessor
from flow_inference.create_trocr_dataset import TrOCRInferenceDataset
from flow_inference.image_processing import ImageHandler
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class InferenceHandler:
    """
    Class for performing inference on textlines.
    """
    def __init__(self, model: PreTrainedModel, processor: TrOCRProcessor, device: torch.device):
        """
        :param model: the TrOCR model.
        :param processor: The TrOCR processor used for inference.
        :param device: cuda, mps or cpu.
        """
        self.model = model
        self.processor = processor
        self.device = device

    @staticmethod
    def custom_collate_fn(batch: List[Dict[str, Union[torch.Tensor, str]]]) \
            -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Custom collate function to stack images and file name.

        :param batch: list of dictionaries with keys 'pixel_values' and 'file_name'.
        :return: dictionary with keys 'pixel_values' and 'filenames'.
        """
        try:
            pixel_values = [item['pixel_values'] for item in batch]
            filenames = [item['filename'] for item in batch]
        except KeyError as e:
            raise KeyError(f"Missing expected key in batch item: {e}")

        # stack action
        pixel_values = torch.stack(pixel_values)

        return {'pixel_values': pixel_values, 'filenames': filenames}

    @staticmethod
    def run_batch_inference(inference_dataloader: DataLoader,
                            model: PreTrainedModel,
                            device: torch.device,
                            processor: TrOCRProcessor,
                            max_new_tokens: int = 100
                            ) -> List[str]:
        """
        Run batch inference.

        :param inference_dataloader: DataLoader.
        :param model: VisionEncoderDecoderModel.
        :param device: cuda, mps or cpu.
        :param processor: TrOCRProcessor.
        :param max_new_tokens: maximum number of new tokens to generate (default: 100)
        :return list of inference results.
        """
        inferred_txt = []

        logger.info("Starting batch inference...")

        for batch in tqdm(inference_dataloader):
            try:
                pixel_values = batch['pixel_values'].to(device)
            except KeyError as e:
                logger.error(f"Missing 'pixel_values' in batch: {e}")
                raise KeyError(f"Missing 'pixel_values' in batch: {e}")

            try:
                outputs = model.generate(pixel_values, max_new_tokens=max_new_tokens)
            except RuntimeError as e:
                logger.error(f"Error during model.generate: {e}")
                raise RuntimeError(f"Error during model.generate: {e}")

            try:
                pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
            except ValueError as e:
                logger.error(f"Error decoding predictions: {e}")
                raise ValueError(f"Error decoding predictions: {e}")

            file_names = batch['filenames']
            line = [f'{os.path.basename(file_name)}\t{pred}' for file_name, pred in zip(file_names, pred_str)]
            inferred_txt.extend(line)

        logger.info(f"Batch inference completed. Total lines processed: {len(inferred_txt)}")
        return inferred_txt

    def infer(self,
              records: List[dict],
              image_handler: ImageHandler,
              **kwargs,
              ) -> List[str]:
        """
        Run the inference for a dataset.

        :param: file_names: list with the file names.
        :param: image_handler: ImageHandler instance.
        :return: list of inference results (for batches).
        """
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        batch_size = kwargs.get('batch_size', 8)

        if not records:
            logger.error("No file names provided for inference.")
            raise ValueError("No file names provided for inference.")

        if not isinstance(image_handler, ImageHandler):
            logger.error(f"Invalid type for image_handler: expected ImageHandler, got {type(image_handler)}")
            raise TypeError("image_handler must be an instance of ImageHandler.")

        logger.info(f"Preparing dataset for inference with {len(records)} lines.")

        try:
            inference_dataset = TrOCRInferenceDataset(
                records=records,
                image_handler=image_handler
            )

            print('Number of lines to infer:', len(inference_dataset))

            inference_dataloader = DataLoader(
                inference_dataset,
                collate_fn=self.custom_collate_fn,
                batch_size=batch_size,
                shuffle=False,
            )
        except FileNotFoundError as e:
            logger.error(f"File not found during dataset preparation: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected keys in dataset: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing dataset or dataloader: {e}")
            raise

        logger.info("Dataset and DataLoader initialized successfully.")

        try:
            list_inferred = self.run_batch_inference(
                inference_dataloader=inference_dataloader,
                model=self.model,
                device=self.device,
                processor=self.processor,
                max_new_tokens=max_new_tokens,
            )
        except KeyError as e:
            logger.error(f"KeyError during inference: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"RuntimeError during inference: {e}")
            raise
        except ValueError as e:
            logger.error(f"ValueError during inference: {e}")
            raise

        return list_inferred
