import unittest
from unittest.mock import patch, MagicMock
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from flow_inference.model_handling import ModelManager


class TestModelManager(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_available(self, _mock_cuda):
        """Test device selection when CUDA is available."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, "cuda")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_device_cuda_and_mps_not_available(self, _mock_mps, _mock_cuda):
        """Test device selection when neither CUDA nor MPS are available."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, "cpu")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_device_mps_available(self, _mock_mps, _mock_cuda):
        """Test device selection when MPS is available but CUDA is not."""
        model_manager = ModelManager()
        self.assertEqual(model_manager.device.type, "mps")

    # --- Model Loading Tests ---

    @patch("transformers.VisionEncoderDecoderModel.from_pretrained")
    def test_load_model_success(self, mock_from_pretrained):
        """Test successful loading of a TrOCR model."""
        mock_model = MagicMock(spec=VisionEncoderDecoderModel)
        mock_from_pretrained.return_value = mock_model

        model_manager = ModelManager()
        model_name = "microsoft/trocr-small-handwritten"

        loaded_model = model_manager.load_model(model_name)

        mock_from_pretrained.assert_called_once_with(model_name)
        mock_model.to.assert_called_once_with(model_manager.device)

        mock_model.eval.assert_called_once()

        self.assertIs(loaded_model, mock_model)

    @patch(
        "transformers.VisionEncoderDecoderModel.from_pretrained",
        side_effect=OSError("Model loading failed"),
    )
    def test_load_model_failure_raises(self, mock_from_pretrained):
        """Test that model loading raises when from_pretrained fails."""
        model_manager = ModelManager()

        with self.assertRaises(OSError) as context:
            model_manager.load_model("non-existent-model")

        self.assertIn("Model loading failed", str(context.exception))
        mock_from_pretrained.assert_called_once_with("non-existent-model")

    @patch("transformers.VisionEncoderDecoderModel.from_pretrained")
    def test_load_model_empty_name_raises(self, mock_from_pretrained):
        """Test that an empty model name raises immediately."""
        model_manager = ModelManager()

        with self.assertRaises(ValueError) as context:
            model_manager.load_model("")

        self.assertIn("Model name must not be empty", str(context.exception))
        mock_from_pretrained.assert_not_called()

    # --- Processor Loading Tests ---

    @patch("transformers.TrOCRProcessor.from_pretrained")
    def test_load_processor_success_fast(self, mock_from_pretrained):
        """Test successful loading of a TrOCR processor with use_fast=True."""
        mock_processor = MagicMock(spec=TrOCRProcessor)
        mock_from_pretrained.return_value = mock_processor

        processor_name = "microsoft/trocr-base-handwritten"

        loaded_processor = ModelManager.load_processor(processor_name)

        mock_from_pretrained.assert_called_once_with(
            processor_name,
            use_fast=True,
        )
        self.assertIs(loaded_processor, mock_processor)

    @patch("transformers.TrOCRProcessor.from_pretrained")
    def test_load_processor_falls_back_to_slow_same_processor(
        self,
        mock_from_pretrained,
    ):
        """
        If fast loading fails, retry the same processor with use_fast=False.

        This must not fall back to another checkpoint such as
        microsoft/trocr-base-handwritten.
        """
        mock_processor = MagicMock(spec=TrOCRProcessor)
        mock_from_pretrained.side_effect = [
            OSError("fast processor loading failed"),
            mock_processor,
        ]

        processor_name = "microsoft/trocr-small-printed"

        loaded_processor = ModelManager.load_processor(processor_name)

        self.assertIs(loaded_processor, mock_processor)
        self.assertEqual(mock_from_pretrained.call_count, 2)

        mock_from_pretrained.assert_any_call(
            processor_name,
            use_fast=True,
        )
        mock_from_pretrained.assert_any_call(
            processor_name,
            use_fast=False,
        )

        called_processor_names = [
            call.args[0] for call in mock_from_pretrained.call_args_list
        ]
        self.assertEqual(
            called_processor_names,
            [processor_name, processor_name],
        )
        self.assertNotIn(
            "microsoft/trocr-base-handwritten",
            called_processor_names,
        )

    @patch("transformers.TrOCRProcessor.from_pretrained")
    def test_load_processor_raises_if_fast_and_slow_fail(
        self,
        mock_from_pretrained,
    ):
        """Test that processor loading raises when both fast and slow loading fail."""
        mock_from_pretrained.side_effect = [
            OSError("fast processor loading failed"),
            OSError("slow processor loading failed"),
        ]

        processor_name = "non-existent-processor"

        with self.assertRaises(RuntimeError) as context:
            ModelManager.load_processor(processor_name)

        self.assertIn("Failed to load processor", str(context.exception))
        self.assertIn(processor_name, str(context.exception))
        self.assertIn("use_fast=True", str(context.exception))
        self.assertIn("use_fast=False", str(context.exception))

        self.assertEqual(mock_from_pretrained.call_count, 2)

        mock_from_pretrained.assert_any_call(
            processor_name,
            use_fast=True,
        )
        mock_from_pretrained.assert_any_call(
            processor_name,
            use_fast=False,
        )

    @patch("transformers.TrOCRProcessor.from_pretrained")
    def test_load_processor_empty_name_raises(self, mock_from_pretrained):
        """Test that an empty processor name raises immediately."""
        with self.assertRaises(ValueError) as context:
            ModelManager.load_processor("")

        self.assertIn("processor_name", str(context.exception))
        mock_from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()