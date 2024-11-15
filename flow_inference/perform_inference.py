import os
from typing import Dict

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class InferenceDataset(Dataset):
    def __init__(self, image_paths, image_processor):
        """
        Args:
            image_paths (list of str): List of file paths to images.
            image_processor (ImageProcessor): An image processor object.
        """
        self.image_paths = image_paths
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the file path and process the image
        image_path = self.image_paths[idx]
        pixel_values = self.image_processor.process_from_path(image_path)

        # Extract filename (without extension) as label
        label = os.path.splitext(os.path.basename(image_path))[0]

        # Return the processed image and the label
        return {'pixel_values': pixel_values, 'label': label}


class InferenceHandler:
    def __init__(self, model, processor, device='cpu'):
        self.model = model
        self.processor = processor
        self.device = device

    def predict_single_image(self, pixel_values):
        """Predicts the text for a single image's pixel values."""
        # Move pixel values to the appropriate device
        pixel_values = pixel_values.to(self.device)

        # Perform the prediction using the model's generate method
        outputs = self.model.generate(
            pixel_values,
            max_new_tokens=100
        )

        # Decode the output to a readable string
        pred_str = self.processor.decode(outputs[0], skip_special_tokens=True)

        return pred_str

    def predict_batch(self, dataset) -> Dict[str, str]:
        """Predicts text for a batch of images from a dataset and returns a dictionary of predictions."""
        predictions = {}

        for idx, batch in tqdm(enumerate(dataset)):
            pixel_values = batch["pixel_values"].to(self.device)

            pred_str = self.predict_single_image(pixel_values)

            print(f"Batch label: {batch['label']}")
            label = batch['label']
            print(f"Prediction: {pred_str}")

            predictions[label] = pred_str

        print(predictions)
        return predictions
