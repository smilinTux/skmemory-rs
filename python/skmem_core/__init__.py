"""Python bindings for the skmemory Rust engine (skmem).

``Engine`` is the scoped store API. Scope is always derived from the process
(``SKMEMORY_READER``); no scope object is exposed.
"""

from ._skmem_core import Engine, __version__, version

__all__ = ["Engine", "__version__", "version"]
