"""Command-line entry point: ``viewer <dataset_dir> [options]``.

Starts a local FastAPI server for one dataset folder and (by default) opens it
in the browser, mirroring how the Inspect viewer is launched.
"""

import argparse
import threading
import webbrowser
from pathlib import Path

from viewer import dataset_io


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viewer",
        description="Browse a generated dataset folder in the web viewer.",
    )
    parser.add_argument("dataset_dir", type=Path, help="Path to the dataset folder")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7341, help="Bind port")
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open a browser window"
    )
    args = parser.parse_args(argv)

    # Validate up front so a bad path fails with a clear message, not a stack
    # trace from inside the server startup.
    try:
        dataset_io.resolve_dataset(args.dataset_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    import uvicorn

    from viewer.server import create_app

    app = create_app(args.dataset_dir)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {args.dataset_dir} at {url}")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
