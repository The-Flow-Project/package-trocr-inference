flow-inference
==============

Python package for performing TrOCR inference and evaluation for OCR/HTR tasks,
developed for the `Flow Project <https://flow-project.net>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:


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

   flow_inference.inference
   flow_inference.infer_textlines
   flow_inference.model_handling


Evaluation
~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.evaluation


Data Handling
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.data_handling
   flow_inference.create_trocr_dataset
   flow_inference.configure_dataset_card


Image and XML Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.image_processing
   flow_inference.xml_processing
   flow_inference.write_inference_to_raw_xml


Export
~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.voyant_export


Status and Utilities
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   flow_inference.utils.logging.inference_logger


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`