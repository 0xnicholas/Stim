"""System One wire protocol, rendering, and derived quantities for Stim."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stim-systemone")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
