"""Standalone entry point without changing the original ``newkcat`` CLI."""

import argparse

from .cli import configure, dispatch


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m newkcat.sequence")
    configure(parser)
    return dispatch(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
