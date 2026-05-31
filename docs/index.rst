flow-inference
==============

Python package for running TrOCR-based OCR/HTR inference and evaluation workflows,
developed for the `Flow Project <https://flow-project.net>`_.

``flow-inference`` connects Hugging Face datasets and models with document-level
OCR/HTR workflows. It can load line-based datasets, run text recognition models on
line images, evaluate predictions against reference transcriptions, and write
inferred text back into XML-based document exports.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Typical workflow
----------------

A common workflow consists of:

1. Loading a line-based OCR/HTR dataset from the Hugging Face Hub.
2. Loading a TrOCR model.
3. Running inference on line images and transferring the results to Hugging Face.
4. Evaluating predictions against available ground truth.
5. Writing predictions back into XML exports.


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