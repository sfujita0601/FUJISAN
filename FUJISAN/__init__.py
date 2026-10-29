"""FUJISAN: prediction of protein functional identity from sequence and structure similarity."""

try:
    from importlib.metadata import version as _version

    __version__ = _version("fujisan")
except Exception:  # not installed (e.g. run from a plain checkout)
    __version__ = "unknown"
