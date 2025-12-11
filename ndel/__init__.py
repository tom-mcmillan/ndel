"""Compatibility shim for the ndel package name.

The core implementation lives in the flat modules under `src/`. This package
re-exports those symbols so `python -m ndel.index` and `import ndel` work.
"""

from src import *  # noqa: F401,F403
