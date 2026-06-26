"""Minimal smoke entrypoint for the fspt package."""

from __future__ import annotations

from . import __version__


def main() -> int:
    print(f"fspt {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
