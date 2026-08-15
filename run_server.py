"""
Start the local model server.

    venv\\Scripts\\python.exe run_server.py

Then open http://127.0.0.1:8000/app/ for the control panel, or
http://127.0.0.1:8000/animation.html for the existing tools -- the URL space is
the same one Live Server serves, so both work under either.

Binds to 127.0.0.1 only. Live Server binds 0.0.0.0, which puts the Model folder
on every network interface; there is no reason for this one to do the same.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="loopback by default; think before widening it")
    p.add_argument("--reload", action="store_true", help="restart on code changes")
    args = p.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is not installed in this interpreter:\n"
            f"  {sys.executable}\n"
            "Install it with:  venv\\Scripts\\python.exe -m pip install fastapi uvicorn")

    from mim.api import app, data_mounts

    print(f"Model folder : {os.path.dirname(os.path.abspath(__file__))}")
    for route, directory in data_mounts().items():
        print(f"Mounted      : {route} -> {directory}")
    print(f"Interpreter  : {sys.executable}")
    print(f"Control panel: http://{args.host}:{args.port}/app/")
    print(f"Animation    : http://{args.host}:{args.port}/animation.html")

    uvicorn.run("mim.api:app" if args.reload else app,
                host=args.host, port=args.port, reload=args.reload,
                log_level="warning")


if __name__ == "__main__":
    main()
