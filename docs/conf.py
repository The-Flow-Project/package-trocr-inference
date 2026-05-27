"""Sphinx configuration for flow-inference documentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from flow_inference import __version__
except ImportError:
    __version__ = "0.0.0"

project = "flow-inference"
copyright = "2026, Jonas Widmer, Dana Meyer"
author = "Jonas Widmer, Dana Meyer"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "sphinx_rtd_theme"
html_show_sphinx = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}