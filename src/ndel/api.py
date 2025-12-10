from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable

from ndel.config import NdelConfig
from ndel.py_analyzer import analyze_python_source
from ndel.render import render_pipeline


def describe_python_source(source: str, config: NdelConfig | None = None) -> str:
    """Analyze Python DS/ML code into NDEL text.

    This is static: code is not executed. The optional config can influence
    naming (aliases) and, in the future, privacy and abstraction behavior.
    """

    pipeline = analyze_python_source(source, config=config)
    return render_pipeline(pipeline, config=config)


def describe_callable(func: Callable, config: NdelConfig | None = None) -> str:
    """Analyze a callable's source code and render NDEL text.

    Uses inspect.getsource under the hood. Raises RuntimeError if the source
    cannot be retrieved (e.g. built-in or dynamically generated).
    """

    try:
        source = inspect.getsource(func)
    except OSError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Could not retrieve source for {func!r}") from exc

    return describe_python_source(textwrap.dedent(source), config=config)


__all__ = ["describe_python_source", "describe_callable"]
