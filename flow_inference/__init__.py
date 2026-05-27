"""Flow Inference - TrOCR inference and evaluation package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flow-inference")
except PackageNotFoundError:
    __version__ = "0.0.0"

__license__ = "MIT"

__all__ = [
    "__version__",
]