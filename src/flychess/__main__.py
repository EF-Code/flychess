"""Allow ``python -m flychess`` to run the command-line experiment."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
