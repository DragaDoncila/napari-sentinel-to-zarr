try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# replace the asterisk with named imports
from .napari_sentinel_to_zarr import napari_get_writer
from .sentinel_to_zarr import main


__all__ = ["napari_get_writer", "main"]
