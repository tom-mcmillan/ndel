"""Entry point shim so `python -m ndel.index` works."""

from src.index import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
