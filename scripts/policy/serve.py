"""A policy network on the GPU for imitation bots in other processes: a bot with
$BOT_POLICY_SERVER=SOCKET (and the same $BOT_POLICY_TABLES) sends each decision here (bot::
policy_remote's protocol, a batch of one per decision) and plays the scores. A thread per
connection; the network scores one batch at a time.

    .venv/bin/python scripts/policy/serve.py MODEL.pt SOCKET [--tables runs/policy/tables/tables8.bin]

MODEL.pt: train.py's model or rl.py's iterN.pt.
"""

import argparse
import os
import socketserver
import struct
import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl import Tables, answer  # noqa: E402
from train import Policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("socket")
    ap.add_argument("--tables", default="runs/policy/tables/tables8.bin")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tables = Tables(a.tables, dev)
    ck = torch.load(a.model, map_location="cpu")
    args = ck["args"]
    model = Policy(tables.dim, args["dim"], args["layers"], args["heads"], strings=bool(args["strings"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            def read(n: int) -> bytes:
                b = self.rfile.read(n)
                if len(b) != n:
                    raise EOFError
                return b

            try:
                while struct.unpack("<I", read(4))[0] == 1:
                    with lock:
                        out = answer(read, [model], tables)
                    self.wfile.write(out)
                    self.wfile.flush()
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass

    if os.path.exists(a.socket):
        os.unlink(a.socket)
    with socketserver.ThreadingUnixStreamServer(a.socket, Handler) as server:
        server.daemon_threads = True
        print(f"serving {a.model} at {a.socket}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
