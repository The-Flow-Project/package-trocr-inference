# flow-inference

TrOCR inference and evaluation for OCR/HTR workflows, developed for the Flow Project.

`flow-inference` is a Python package for running TrOCR-based OCR/HTR inference on line-level datasets. It connects Hugging Face datasets and models with document-processing workflows used in the Flow Project.

The package can download prepared line-level datasets, run text recognition models on document image lines, write predictions back into timestamped `inference_*` columns, evaluate predictions with Character Error Rate (CER), write inferred text back into raw XML records, and export document-level text for downstream analysis.

## Overview

`flow-inference` is designed for OCR/HTR workflows where line images have already been extracted from document images with [`pagexml-hf`](https://the-flow-project.github.io/pagexml-hf/).

A typical workflow is:

1. Start with a line-level Hugging Face dataset.
2. Run TrOCR inference on selected dataset splits.
3. Store predictions in a timestamped `inference_*` column.
4. Optionally upload the updated dataset to the Hugging Face Hub.
5. Evaluate predictions against ground-truth transcriptions.
6. Optionally write predictions back into raw XML records or export text for Voyant.

## Features

- **TrOCR Inference**: Run OCR/HTR inference with TrOCR vision-encoder-decoder models
- **Hugging Face Integration**: Load and process line-based datasets from the Hugging Face Hub
- **Line-Level Prediction**: Generate text predictions for document image lines
- **Evaluation Metrics**: Evaluate predictions with Character Error Rate (CER)
- **XML Writeback**: Write inferred text back into raw XML records
- **Voyant Export**: Export inference results for downstream text analysis
- **Status Tracking**: Track inference and evaluation progress
- **Upload Modes**: Create, replace, or update Hugging Face dataset repositories

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/The-Flow-Project/package-trocr-inference.git
cd package-trocr-inference
pip install .
```

Or, if you use `uv`, sync the project environment:

```bash
git clone https://github.com/The-Flow-Project/package-trocr-inference.git
cd package-trocr-inference
uv sync
```

For development and documentation work, include the optional dependencies:

```bash
uv sync --extra dev --extra docs
```

## Supported Workflows

### 1. Inference

Runs a TrOCR-compatible model on line-level image records and stores predictions in a new timestamped `inference_*` column.

### 2. Evaluation

Compares predictions against the `text` ground-truth column and computes Character Error Rate (CER). Evaluation artifacts are uploaded to the dataset repository.

### 3. Raw XML Writeback

Inserts inferred text into matching `TextLine` and `TextRegion` elements in raw XML records. The updated XML is stored in a timestamped `inference_xml_*` column.

### 4. Voyant Export

Exports inferred text as one `.txt` file per document and bundles the files into a ZIP archive for use in Voyant Tools or similar downstream text-analysis workflows.

## Input Dataset Structure

The main inference workflow expects a Hugging Face dataset containing line-level records.

Required columns for inference:

- `image`: cropped line image or Hugging Face image object
- `filename`: source page or image filename
- `region_id`: parent text region identifier
- `line_id`: text line identifier

Required columns for evaluation:

- `text`: ground-truth transcription
- ``inference_*``: column containing model predictions to evaluate (created by the inference workflow)

Optional columns:

- `project_name`: project or collection identifier

## Output Columns and Artifacts

Inference output is written to timestamped columns:

```text
inference_<timestamp>_model_<model_name>
```

Example:

```text
inference_20260531_143012_123456_model_microsoft_trocr-small-handwritten
```

Evaluation creates text and JSON artifacts:

```text
evaluation/<timestamp>/
├── gt.txt
├── hypothesis.txt
└── evaluation_report.json
```

Raw XML writeback creates timestamped XML columns:

```text
inference_xml_<timestamp>
```

## Usage

### Run Inference

```python
from flow_inference.inference import Inference

runner = Inference(
    download_repo_name="my-org/my-line-dataset",
    hf_token="hf_...",
    trocr_model="microsoft/trocr-small-handwritten",
    splits=["train"],
    push_to_hub=True,
    upload_repo_name="my-org/my-inference-output",
    upload_mode="new_repo",
    private_repo=True,
)

updated_dfs = runner.perform_inference()
```

### Evaluate Inference Output

The evaluation workflow uses the latest non-XML `inference_*` column and compares it against the `text` column.

```python
from flow_inference.evaluation import Evaluation

evaluator = Evaluation(
    evaluation_repo_name="my-org/my-inference-output",
    hf_token="hf_...",
    splits=["train"],
)

files = evaluator.perform_evaluation()
```

### Write Inference Back to Raw XML

```python
from flow_inference.write_inference_to_raw_xml import InferenceToRawXMLWriter

writer = InferenceToRawXMLWriter(
    raw_xml_repo="my-org/my-raw-xml-dataset",
    inference_repo="my-org/my-inference-output",
    token="hf_...",
)

result = writer.process_and_upload(
    output_repo="my-org/my-raw-xml-with-inference",
    upload_mode="new_repo",
    private=True,
)
```

### Export for Voyant

```python
from flow_inference.voyant_export import VoyantExporter

zip_path = VoyantExporter.from_huggingface(
    dataset_name="my-org/my-inference-output",
    split="train",
    hf_token="hf_...",
    zip_path="voyant_export.zip",
)
```

## Upload Modes

The package supports several upload modes when writing datasets back to the Hugging Face Hub:

- `new_repo`: create a new target repository and fail if it already exists
- `replace`: replace dataset files in an existing target repository
- `update`: update a compatible existing repository while preserving previous inference columns

By default, the package refuses to upload into the source repository. Set `allow_source_repo_update=True` only when updating the source repository is intentional.

## Use Cases

`flow-inference` is useful for:

- Running OCR/HTR prediction on line-level datasets
- Comparing model output against ground-truth transcriptions
- Preserving inference results in Hugging Face dataset repositories
- Creating evaluation artifacts for model comparison
- Writing recognized text back into XML-based document exports
- Preparing document-level text exports for analysis tools such as Voyant

## Authentication

For private Hugging Face repositories or uploads, provide a Hugging Face token.

You can pass the token directly in Python:

```python
hf_token = "hf_..."
```

Or set it as an environment variable:

```bash
export HF_TOKEN=hf_...
```

## Requirements

- Python >= 3.12
- PyTorch >= 2.6
- Transformers
- Hugging Face Datasets
- Hugging Face Hub
- pandas
- Pillow
- lxml
- jiwer
- evaluate

A CUDA-compatible GPU is recommended for faster inference, but CPU inference is also supported.

## Documentation

The API documentation is available here:

https://the-flow-project.github.io/package-trocr-inference/

## License

MIT License.
