"""Run batched TrOCR inference on line-level image records.

This module prepares Hugging Face-style records for PyTorch batching, runs a
TrOCR-compatible model on processed line images, decodes generated token IDs,
and returns predictions together with their source metadata.
"""

# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import time
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
    """Run TrOCR inference for line-level OCR/HTR records.

    The handler wraps a loaded model, processor, and device. It prepares batches
    with line metadata, executes generation in inference mode, and returns
    decoded text predictions mapped back to project, filename, region, and line
    identifiers.
    """
    def __init__(self, model: PreTrainedModel, processor: TrOCRProcessor, device: torch.device):
        """Initialize the inference handler.

        Args:
            model: Loaded TrOCR-compatible model used for generation.
            processor: Processor used to decode generated token IDs.
            device: Torch device used for inference.
        """
        self.model = model
        self.model.eval()
        self.processor = processor
        self.device = device

    @staticmethod
    def custom_collate_fn(batch: List[Dict[str, Union[torch.Tensor, str]]]) \
            -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Stack image tensors and preserve line-level metadata.

        Args:
            batch: Batch items produced by ``TrOCRInferenceDataset``.

        Returns:
            Dictionary containing stacked ``pixel_values`` and metadata lists for
            filenames, region IDs, line IDs, and project names.

        Raises:
            KeyError: If a required batch item key is missing.
        """
        try:
            pixel_values = [item["pixel_values"] for item in batch]
            filenames = [item["filename"] for item in batch]
            region_ids = [item["region_id"] for item in batch]
            line_ids = [item["line_id"] for item in batch]
        except KeyError as e:
            raise KeyError(f"Missing expected key in batch item: {e}")

        project_names = [item.get("project_name", "") for item in batch]

        # stack action
        pixel_values = torch.stack(pixel_values)

        return {'pixel_values': pixel_values,
                'filenames': filenames,
                "region_ids": region_ids,
                'line_ids': line_ids,
                'project_names': project_names
                }

    @staticmethod
    def run_batch_inference(inference_dataloader: DataLoader,
                            model: PreTrainedModel,
                            device: torch.device,
                            processor: TrOCRProcessor,
                            max_new_tokens: int = 100
                            ) -> List[tuple[str, str, str, str, str]]:
        """Run model generation for all batches in a DataLoader.

        Args:
            inference_dataloader: DataLoader yielding processed image tensors and metadata.
            model: TrOCR-compatible model used for text generation.
            device: Torch device used for inference.
            processor: Processor used to decode generated token IDs.
            max_new_tokens: Maximum number of new tokens generated per line image.

        Returns:
            Inference results as tuples of ``project_name``, ``filename``,
            ``region_id``, ``line_id``, and predicted text.

        Raises:
            KeyError: If a required batch key is missing.
            RuntimeError: If model generation fails.
            ValueError: If prediction decoding fails.
        """
        inferred_txt = []

        logger.info("Starting batch inference...")
        start_time = time.perf_counter()
        total_lines = 0
        use_amp = device.type == "cuda"

        with torch.inference_mode():
            for batch in tqdm(inference_dataloader):
                try:
                    pixel_values = batch['pixel_values'].to(device,
                                                            non_blocking=True)
                except KeyError as e:
                    logger.error(f"Missing 'pixel_values' in batch: {e}")
                    raise KeyError(f"Missing 'pixel_values' in batch: {e}")

                try:
                    with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=use_amp,
                    ):
                        outputs = model.generate(pixel_values,
                                                 max_new_tokens=max_new_tokens)
                except RuntimeError as e:
                    logger.error(f"Error during model.generate: {e}")
                    raise RuntimeError(f"Error during model.generate: {e}")

                try:
                    pred_str = processor.batch_decode(outputs,
                                                      skip_special_tokens=True)
                except ValueError as e:
                    logger.error(f"Error decoding predictions: {e}")
                    raise ValueError(f"Error decoding predictions: {e}")

                line_ids = batch["line_ids"]
                region_ids = batch["region_ids"]
                filenames = batch["filenames"]
                project_names = batch["project_names"]

                inferred_txt.extend(
                    (
                        str(project),
                        str(filename),
                        str(region_id),
                        str(line_id),
                        str(pred),
                    )
                    for project, filename, region_id, line_id, pred
                    in zip(project_names, filenames, region_ids, line_ids, pred_str)
                )

                total_lines += len(pred_str)

            elapsed = time.perf_counter() - start_time
            speed = total_lines / elapsed if elapsed > 0 else 0.0

            logger.info(
                f"Batch inference completed: {total_lines} lines "
                f"in {elapsed:.2f}s ({speed:.2f} lines/sec)"
            )
            return inferred_txt

    def infer(self,
              records: List[dict],
              image_handler: ImageHandler,
              **kwargs,
              ) -> List[tuple[str, str, str, str, str]]:
        """Run inference for a collection of line-level records.

        Args:
            records: Hugging Face-style records containing image data and metadata.
            image_handler: Image handler used to process record images.
            **kwargs: Optional inference settings such as ``batch_size``,
                ``num_workers``, and ``max_new_tokens``.

        Returns:
            Inference results as tuples of ``project_name``, ``filename``,
            ``region_id``, ``line_id``, and predicted text.

        Raises:
            TypeError: If ``image_handler`` is not an ``ImageHandler`` instance.
            KeyError: If required record or batch keys are missing.
            RuntimeError: If model generation fails.
            ValueError: If image processing or decoding fails.
        """
        max_new_tokens = kwargs.get('max_new_tokens', 64)
        batch_size = kwargs.get('batch_size', 8)
        num_workers = kwargs.get("num_workers", 0)

        if not records:
            logger.error("No records provided for inference.")
            return []

        if not isinstance(image_handler, ImageHandler):
            logger.error(f"Invalid type for image_handler: expected ImageHandler, got {type(image_handler)}")
            raise TypeError("image_handler must be an instance of ImageHandler.")

        logger.info(f"Preparing dataset for inference with {len(records)} lines.")

        try:
            inference_dataset = TrOCRInferenceDataset(
                records=records,
                image_handler=image_handler
            )

            logger.info(f"Number of lines to infer: {len(inference_dataset)}")

            inference_dataloader = DataLoader(
                inference_dataset,
                collate_fn=self.custom_collate_fn,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=(self.device.type == "cuda"),
                num_workers=num_workers,
                persistent_workers=False
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
