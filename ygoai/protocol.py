"""Engine protocol: AIs as separate processes talking JSON lines over stdio.

This is the Yu-Gi-Oh counterpart of UCI. The arbiter owns the duel and sends
each engine only what its player is allowed to know (see observe.py):

  arbiter -> engine   {"type": "isready"}
  engine  -> arbiter  {"ready": true}
  arbiter -> engine   {"type": "init", "player": 0|1, "deck": {"main": [...], "extra": [...]},
                       "extra_order": [...], "opponent_deck_size": n,
                       "opponent_extra_size": m, "config": {...}}
  arbiter -> engine   {"type": "prompt", "obs": [msg, ...], "prompt": {...}, "state": {...}}
  engine  -> arbiter  {"response": {...}}
  arbiter -> engine   {"type": "end", "obs": [msg, ...], "result": {...}}
  arbiter -> engine   {"type": "quit"}

"obs" holds every observation since the previous prompt. A prompt is one of
the core's selection messages with hidden information removed; the response
format for each prompt type is documented in actions.py. "state" is the
player's view of the whole field at that moment (state.py). "isready" (sent
once, before the first game) lets the engine load what it needs before any
clock runs. An engine may be reused for several games: a new "init" starts a
new game (the arbiter starts a fresh engine for each game).

`EngineProcess` is the arbiter-side Agent wrapper around such a process;
`serve` runs any in-process Agent as an engine (python -m ygoai.engine).
"""

from __future__ import annotations

import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import time

from .agents.base import Agent


class EngineError(RuntimeError):
    pass


_PARENT_ENV = "YGOAI_ENGINE_PARENT"  # (the arbiter-side process: see _end_with_parent)
# (engines fork (the belief's probes, a search's nodes): no BLAS thread pools, a fork
# would copy their state but not their threads, and a child could wait on them forever)
SINGLE_THREADED = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
READY_TIMEOUT = 30.0  # (seconds an engine may take to load; one that does not know "isready" is not waited for longer)


class EngineProcess(Agent):
    wants_state = True

    def __init__(self, command: list[str] | str, name: str | None = None, cwd: str | None = None):
        if isinstance(command, str):
            command = command.split()
        self.command = command
        self.name = name or command[-1]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, cwd=cwd,
                                     env={**SINGLE_THREADED, **os.environ, _PARENT_ENV: str(os.getpid())})
        self.pending: list[dict] = []
        self.last_think_time = 0.0
        # (loading the engine is not on its clock: wait until it is ready)
        self._send({"type": "isready"})
        if select.select([self.proc.stdout], [], [], READY_TIMEOUT)[0]:
            reply = self._recv()
            if not reply.get("ready"):
                raise EngineError(f"{self.name} sent {reply!r} instead of ready")

    def _send(self, obj: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise EngineError(f"{self.name} closed its input") from e

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise EngineError(f"{self.name} exited")
        return json.loads(line)

    def start(self, player: int, init: dict) -> None:
        super().start(player, init)
        self.pending = []
        self._send({"type": "init", **init})

    def observe(self, msg: dict) -> None:
        self.pending.append(msg)

    def respond(self, prompt: dict, state: dict | None) -> dict:
        t0 = time.monotonic()
        msg = {"type": "prompt", "obs": self.pending, "prompt": prompt, "state": state}
        if self.clock is not None:
            msg["clock"] = {"remaining": self.clock[0], "increment": self.clock[1]}
        self._send(msg)
        self.pending = []
        reply = self._recv()
        self.last_think_time = time.monotonic() - t0
        if "response" not in reply:
            raise EngineError(f"{self.name} sent {reply!r} instead of a response")
        return reply["response"]

    def end(self, result: dict) -> None:
        self._send({"type": "end", "obs": self.pending, "result": result})
        self.pending = []

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send({"type": "quit"})
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
                self.proc.wait()
        for pipe in (self.proc.stdin, self.proc.stdout):
            if pipe:
                pipe.close()


def _end_with_parent() -> None:
    """An engine started by EngineProcess ends with that process, however it ends
    (killed, its run stopped): it would otherwise live on, holding its memory."""
    parent = os.environ.get(_PARENT_ENV)
    if not parent:
        return
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        return
    if os.getppid() != int(parent):
        os._exit(0)  # (it died before the line above)


def serve(agent_factory, stdin=sys.stdin, stdout=sys.stdout) -> None:
    """Runs agents created by `agent_factory()` as an engine on stdio."""
    _end_with_parent()
    agent, fresh = None, False  # (fresh: created for "isready", no game yet)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        t = msg["type"]
        if t == "isready":
            if agent is None:
                agent, fresh = agent_factory(), True
                agent.warm_up()
            stdout.write(json.dumps({"ready": True}) + "\n")
            stdout.flush()
        elif t == "init":
            if agent is None or not fresh:
                agent = agent_factory()
            fresh = False
            agent.start(msg["player"], msg)
        elif t == "prompt":
            for o in msg.get("obs", ()):
                agent.observe(o)
            clock = msg.get("clock")
            agent.clock = (clock["remaining"], clock["increment"]) if clock else None
            resp = agent.respond(msg["prompt"], msg.get("state"))
            stdout.write(json.dumps({"response": resp}, separators=(",", ":")) + "\n")
            stdout.flush()
        elif t == "end":
            for o in msg.get("obs", ()):
                agent.observe(o)
            agent.end(msg["result"])
        elif t == "quit":
            break
