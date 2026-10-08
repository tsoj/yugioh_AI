"""Engine protocol: AIs as separate processes talking JSON lines over stdio.

This is the Yu-Gi-Oh counterpart of UCI. The arbiter owns the duel and sends
each engine only what its player is allowed to know (see observe.py):

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
player's view of the whole field at that moment (state.py). An engine may be
reused for several games: a new "init" starts a new game.

`EngineProcess` is the arbiter-side Agent wrapper around such a process;
`serve` runs any in-process Agent as an engine (python -m ygoai.engine).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

from .agents.base import Agent


class EngineError(RuntimeError):
    pass


class EngineProcess(Agent):
    wants_state = True

    def __init__(self, command: list[str] | str, name: str | None = None, cwd: str | None = None):
        if isinstance(command, str):
            command = command.split()
        self.command = command
        self.name = name or command[-1]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, cwd=cwd)
        self.pending: list[dict] = []
        self.last_think_time = 0.0

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


def serve(agent_factory, stdin=sys.stdin, stdout=sys.stdout) -> None:
    """Runs agents created by `agent_factory()` as an engine on stdio."""
    agent = None
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        t = msg["type"]
        if t == "init":
            agent = agent_factory()
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
