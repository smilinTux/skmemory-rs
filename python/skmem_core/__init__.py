"""Python bindings for the skmemory Rust engine (skmem).

``Engine`` is the scoped store API. Scope is always derived from the process
(``SKMEMORY_READER``); no scope object is exposed.
"""

from ._skmem_core import RELEASES_GIL, Engine, __version__, version

__all__ = ["RELEASES_GIL", "Engine", "__version__", "version"]
