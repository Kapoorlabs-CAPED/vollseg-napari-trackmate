try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._sample_data import make_sample_data
from ._temporal_plots import TemporalStatistics
from ._widget import plugin_wrapper_track
from ._writer import write_multiple, write_single_image

__all__ = (
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "plugin_wrapper_track",
    "TemporalStatistics",
)
