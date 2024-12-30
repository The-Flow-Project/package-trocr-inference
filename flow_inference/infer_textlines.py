import os
from typing import List, Union, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, TrOCRProcessor
from flow_inference.create_trocr_dataset import TrOCRInferenceDataset
from flow_inference.image_processing import ImageHandler


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
        :return: dictionary with keys 'pixel_values' and 'file_names'.
        """
        pixel_values = [item['pixel_values'] for item in batch]
        file_names = [item['file_name'] for item in batch]

        # stack action
        pixel_values = torch.stack(pixel_values)

        return {'pixel_values': pixel_values, 'file_names': file_names}

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

        # infer with the model
        print('Running inference...')

        for batch in tqdm(inference_dataloader):
            # predict using generate
            pixel_values = batch['pixel_values'].to(device)
            outputs = model.generate(pixel_values, max_new_tokens=max_new_tokens)

            # decode
            pred_str = processor.batch_decode(outputs, skip_special_tokens=True)

            file_names = batch['file_names']
            line = [f'{os.path.basename(file_name)}\t{pred}' for file_name, pred in zip(file_names, pred_str)]
            inferred_txt.extend(line)

        return inferred_txt

    def infer(self,
              file_names: List[str],
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

        inference_dataset = TrOCRInferenceDataset(
            file_names=file_names,
            image_handler=image_handler
        )

        print('Number of lines to infer:', len(inference_dataset))

        inference_dataloader = DataLoader(
            inference_dataset,
            collate_fn=self.custom_collate_fn,
            batch_size=8,
            shuffle=False,
        )

        list_inferred = self.run_batch_inference(
            inference_dataloader=inference_dataloader,
            model=self.model,
            device=self.device,
            processor=self.processor,
            max_new_tokens=max_new_tokens,
        )

        return list_inferred
