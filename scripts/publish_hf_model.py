#!/usr/bin/env python3
"""Deprecated compatibility-publisher guard.

The old publisher released a third-party compatibility artifact. Flychess now
publishes only the independent FlyNet bundle through
``scripts/publish_flynet_model.py``. Keeping this guard under the old filename
prevents an accidental re-release of the retired payload.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    parser.error("deprecated publisher; use scripts/publish_flynet_model.py for an independent release")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
