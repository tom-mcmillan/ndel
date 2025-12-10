from __future__ import annotations

import argparse
import sys

from .render import render_pipeline
from .semantic_model import Dataset, Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ndel",
        description="Describe Python/SQL data pipelines in human-readable NDEL",
    )

    subparsers = parser.add_subparsers(dest="command")

    describe_parser = subparsers.add_parser(
        "describe",
        help="Describe Python/SQL into NDEL (experimental)",
    )
    describe_parser.add_argument(
        "--example",
        action="store_true",
        help="Render a built-in example pipeline",
    )

    return parser


def _example_pipeline() -> Pipeline:
    return Pipeline(
        name="example_pipeline",
        description="Example pipeline rendering",
        datasets=[
            Dataset(
                name="users_activity_30d",
                description="User activity in last 30 days",
                notes=["source: analytics warehouse"],
            )
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "describe":
        if getattr(args, "example", False):
            pipeline = _example_pipeline()
            print(render_pipeline(pipeline))
            return 0

        parser.print_help()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
