"""Run the educational Flask application: python -m app.server."""

from __future__ import annotations

import argparse
from pathlib import Path

from app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Handwritten digit recognition web application")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for local development")
    parser.add_argument("--root", type=Path, default=None, help="Directory containing data, models and reports")
    args = parser.parse_args()
    create_app(project_root=args.root).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
