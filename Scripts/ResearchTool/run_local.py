from __future__ import annotations

import argparse
import socket
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Satellite Pipeline Explorer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail if --port is busy instead of trying the next free port.",
    )
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(host: str, start_port: int, strict: bool) -> int:
    if strict or port_is_free(host, start_port):
        return start_port

    for port in range(start_port + 1, start_port + 50):
        if port_is_free(host, port):
            return port

    raise RuntimeError(f"No free port found in range {start_port}-{start_port + 49}.")


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parent
    port = choose_port(args.host, args.port, args.strict_port)
    if port != args.port:
        print(f"Port {args.port} is busy, using {port} instead.")
    print(f"Open http://{args.host}:{port}")
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=port,
        reload=args.reload,
        reload_dirs=[str(root / "backend"), str(root / "frontend")] if args.reload else None,
    )
