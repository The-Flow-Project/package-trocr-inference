flow-inference
==============

TrOCR inference and evaluation for OCR/HTR workflows, developed for the
`Flow Project <https://flow-project.net>`_.

Overview
--------

``flow-inference`` is a Python package for running TrOCR-based OCR/HTR inference
and evaluation workflows on line-level datasets. It connects Hugging Face
datasets and models with document-processing workflows used in the Flow Project.

The package can download prepared line-level datasets, run text recognition
models on document image lines, write predictions back into timestamped
``inference_*`` columns, evaluate predictions with Character Error Rate (CER),
write inferred text back into raw XML records, and export document-level text
for downstream analysis.

Features
--------

- **TrOCR Inference** - Run OCR/HTR inference with TrOCR vision-encoder-decoder models
- **Hugging Face Integration** - Load and process line-based datasets from the Hugging Face Hub
- **Line-Level Prediction** - Generate text predictions for document image lines
- **Evaluation Metrics** - Evaluate predictions with Character Error Rate (CER)
- **XML Writeback** - Write inferred text back into raw XML records
- **Voyant Export** - Export inference results for downstream text analysis
- **Status Tracking** - Track inference and evaluation progress during longer-running jobs

Installation
------------

Install the package from the repository:

.. code-block:: bash

   git clone https://github.com/The-Flow-Project/package-trocr-inference.git
   cd package-trocr-inference
   pip install .

Or install it in a local ``uv`` environment:

.. code-block:: bash

   git clone https://github.com/The-Flow-Project/package-trocr-inference.git
   cd package-trocr-inference
   uv sync

For development with documentation dependencies:

.. code-block:: bash

   uv sync --extra dev --extra docs

Supported Workflows
-------------------

``flow-inference`` supports several related OCR/HTR workflows:

- **Inference** - Run a TrOCR-compatible model on line-level image records and store predictions in a new ``inference_*`` column.
- **Evaluation** - Compare predictions against the ``text`` ground-truth column and compute Character Error Rate (CER).
- **Raw XML Writeback** - Insert inferred text into matching ``TextLine`` and ``TextRegion`` elements in raw XML records.
- **Voyant Export** - Export inferred text as document-level ``.txt`` files for downstream text analysis.

Typical Workflow
----------------

A common workflow consists of:

1. Prepare a line-level OCR/HTR dataset with `pagexml-hf <https://the-flow-project.github.io/pagexml-hf/>`_.
2. Run TrOCR inference on one or more selected dataset splits.
3. Store predictions in a timestamped ``inference_*`` column.
4. Optionally upload the updated dataset to the Hugging Face Hub.
5. Evaluate predictions against the ``text`` ground-truth column.
6. Optionally write predictions back into raw XML records or export text for Voyant.

Input Dataset Structure
-----------------------

The main inference workflow expects a Hugging Face dataset containing line-level records.

Required columns for inference:

- ``image``: cropped line image or Hugging Face image object
- ``filename``: source page or image filename
- ``region_id``: parent text region identifier
- ``line_id``: text line identifier

Required columns for evaluation:

- ``text``: ground-truth transcription
- ``inference_*``: column containing model predictions to evaluate (created by the inference workflow)

Optional columns:

- ``project_name``: project or collection identifier

Output Columns and Artifacts
----------------------------

Inference output is written to timestamped columns:

.. code-block:: text

   inference_<timestamp>_model_<model_name>

Example:

.. code-block:: text

   inference_20260531_143012_123456_model_microsoft_trocr-small-handwritten

Evaluation creates text and JSON artifacts:

.. code-block:: text

   evaluation/<timestamp>/
   ├── gt.txt
   ├── hypothesis.txt
   └── evaluation_report.json

Raw XML writeback creates timestamped XML columns:

.. code-block:: text

   inference_xml_<timestamp>

Usage
-----

Run Inference
~~~~~~~~~~~~~

Use the ``Inference`` class to run TrOCR inference on a Hugging Face dataset:

.. code-block:: python

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

Evaluate Inference Output
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``Evaluation`` class to evaluate the latest ``inference_*`` column
against the ``text`` column:

.. code-block:: python

   from flow_inference.evaluation import Evaluation

   evaluator = Evaluation(
       evaluation_repo_name="my-org/my-inference-output",
       hf_token="hf_...",
       splits=["test"],
   )

   files = evaluator.perform_evaluation()

Write Inference Back to Raw XML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``InferenceToRawXMLWriter`` to insert inferred text into raw XML records:

.. code-block:: python

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

Export for Voyant
~~~~~~~~~~~~~~~~~

Use ``VoyantExporter`` to export inference output as one text file per document:

.. code-block:: python

   from flow_inference.voyant_export import VoyantExporter

   zip_path = VoyantExporter.from_huggingface(
       dataset_name="my-org/my-inference-output",
       split="train",
       hf_token="hf_...",
       zip_path="voyant_export.zip",
   )

Upload Modes
------------

Several upload modes are supported when writing datasets back to the Hugging Face Hub:

- ``new_repo``: create a new target repository and fail if it already exists
- ``replace``: replace dataset files in an existing target repository
- ``update``: update a compatible existing repository while preserving previous inference columns

By default, the package refuses to upload into the source repository. Set
``allow_source_repo_update=True`` only when updating the source repository is
intentional.

Use Cases
---------

``flow-inference`` is useful for:

- Running OCR/HTR prediction on line-level datasets
- Comparing model output against ground-truth transcriptions
- Preserving inference results in Hugging Face dataset repositories
- Creating evaluation artifacts for model comparison
- Writing recognized text back into XML-based document exports
- Preparing document-level text exports for analysis tools such as Voyant

Authentication
--------------

For private Hugging Face repositories or uploads, provide a Hugging Face token.

You can pass the token directly in Python:

.. code-block:: python

   hf_token = "hf_..."

Or set it as an environment variable:

.. code-block:: bash

   export HF_TOKEN=hf_...

Requirements
------------

- Python >= 3.12
- PyTorch
- Transformers
- Hugging Face Datasets
- Hugging Face Hub
- pandas
- Pillow
- lxml

License
-------

MIT License.


API Reference
=============

Main Package
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   flow_inference


Core Modules
------------

Inference
~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.inference.Inference
   flow_inference.infer_textlines.InferenceHandler
   flow_inference.model_handling.ModelManager
   flow_inference.create_trocr_dataset.TrOCRInferenceDataset


Evaluation
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.evaluation.Evaluation


Data Handling
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.data_handling.HuggingFaceDataHandler
   flow_inference.configure_dataset_card.HuggingFaceReadmeBuilder
   flow_inference.configure_dataset_card.ReadmeStats


Image and XML Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.image_processing.ImageHandler
   flow_inference.xml_processing.XMLProcessor
   flow_inference.write_inference_to_raw_xml.InferenceToRawXMLWriter


Export
~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.voyant_export.VoyantExporter


Status and Logging
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.status.Status
   flow_inference.utils.logging.inference_logger


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`