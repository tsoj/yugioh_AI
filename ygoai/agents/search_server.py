"""The alpha-beta search in a small process of its own.

The search forks a process at every node, and a fork costs by the size of the
process: the engine's holds the belief (its reconstructions, scipy, the deck
models' clients), ~130-350 MB, against ~40 MB here. The duel cannot be copied,
but the core is deterministic: the engine's duels keep a journal of every call
(core.journal_duels), and the server makes the same calls on a duel of its own
(the mirror), checking each result against the journal's checksum. At each
decision the client sends what the journal gained since the last one, and the
server searches its mirror in a forked process (so the mirror stays as the
journal left it; its tables live on in shared memory).

`SearchClient.search` returns None when the server cannot answer (the mirror
does not match, the server failed or is gone): the agent then searches itself.

  python -m ygoai.agents.search_server READ_FD WRITE_FD PARENT_PID   (started by SearchClient)
"""

from __future__ import annotations

import ctypes
import os
import pickle
import random
import signal
import subprocess
import sys
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

READY_TIMEOUT = 60.0
REPLY_MARGIN = 30.0  # (seconds past the search's hard bound before the server counts as hung)


class SearchClient:
    def __init__(self):
        to_server, from_client = os.pipe()
        to_client, from_server = os.pipe()
        root = Path(__file__).resolve().parents[2]
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ygoai.agents.search_server", str(to_server), str(from_server), str(os.getpid())],
            pass_fds=(to_server, from_server), cwd=root, stdin=subprocess.DEVNULL)
        os.close(to_server)
        os.close(from_server)
        self.out = Connection(from_client, readable=False)
        self.inp = Connection(to_client, writable=False)
        self.key, self.sent = None, 0  # (the mirrored duel and how much of its journal the server has)
        self.alive = True
        self.desyncs = 0
        self._request(("warm",), READY_TIMEOUT)

    def _request(self, msg, timeout: float):
        if not self.alive:
            return None
        try:
            self.out.send(msg)
            if not self.inp.poll(timeout):
                raise TimeoutError("search server did not answer")
            return self.inp.recv()
        except (OSError, EOFError, TimeoutError, pickle.PickleError):
            self.close()  # (gone or hung: no more requests, the agent searches itself)
            return None

    def search(self, duel, player: int, seed: int, clock, prompt: dict, cands: list[dict], t0: float,
               budget: float | None) -> tuple[int, int, int] | None:
        """(best candidate index, depth completed, nodes), or None."""
        if duel._key != self.key:
            self.key, self.sent = duel._key, 0
        journal = duel.journal
        delta = journal[self.sent:]
        timeout = REPLY_MARGIN + (60.0 if budget is None else 4 * budget)
        reply = self._request(("search", duel._key, self.sent, delta, player, seed, clock, prompt, cands, t0,
                               budget), timeout)
        if reply is None:
            return None
        if reply[0] in ("ok", "error"):
            self.sent = len(journal)  # (the mirror has it all: an error is the search's)
            return reply[1] if reply[0] == "ok" else None
        self.key, self.sent = None, 0  # (the server dropped its mirror: all of it next time)
        if reply[0] == "desync":
            self.desyncs += 1
        return None

    def close(self) -> None:
        if self.alive:
            self.alive = False
            for c in (self.out, self.inp):
                try:
                    c.close()
                except OSError:
                    pass
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass


def _serve(inp: Connection, out: Connection) -> None:
    from ..core import Duel, replay_call
    from .alphabeta import AlphaBetaAgent, _huge_pages
    mirror, key, n = None, None, 0
    agents: dict[int, AlphaBetaAgent] = {}
    while True:
        try:
            msg = inp.recv()
        except EOFError:
            return
        if msg[0] == "warm":
            from ..carddb import default_db
            Duel((0, 0, 0, 0), None, default_db()).close()  # (compiles the utility scripts)
            out.send(("ready",))
            continue
        _, k, start, delta, player, seed, clock, prompt, cands, t0, budget = msg
        if k != key or start != n:
            if mirror is not None:
                mirror.close()
            mirror, key, n = None, k, 0
            if start != 0:
                out.send(("resync",))
                continue
        ok = True
        try:
            for i, entry in enumerate(delta):
                if entry[0] == "__init__":
                    mirror = Duel(*entry[1])
                elif not replay_call(mirror, entry):
                    print(f"search server: journal entry {n + i} ({entry[0]}) does not match", file=sys.stderr)
                    ok = False
                    break
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            ok = False
        if not ok:
            if mirror is not None:
                mirror.close()
            mirror, key, n = None, None, 0
            out.send(("desync",))
            continue
        n += len(delta)
        agent = agents.get(player)
        if agent is None:
            agent = agents[player] = AlphaBetaAgent(seed, completion=False)
            agent.player = player
            agent.collapse_pages = False  # (here, before the fork)
        agent.rng = random.Random(seed)
        agent.clock = clock
        agent.prepare()
        _huge_pages()
        out.send(_search_in_fork(agent, mirror, prompt, cands, t0, budget))


def _search_in_fork(agent, duel, prompt, cands, t0, budget) -> tuple:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        data = b""
        try:
            nodes = agent.nodes
            best, done = agent.search(duel, prompt, cands, t0, budget)
            data = pickle.dumps((best, done, agent.nodes - nodes))
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
        finally:
            with os.fdopen(w, "wb") as f:
                f.write(data)
            os._exit(0)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        data = f.read()
    os.waitpid(pid, 0)
    return ("ok", pickle.loads(data)) if data else ("error",)


def main() -> None:
    read_fd, write_fd, parent = (int(x) for x in sys.argv[1:4])
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: ends with the engine
    except (OSError, AttributeError):
        pass
    if os.getppid() != parent:
        return
    _serve(Connection(read_fd, writable=False), Connection(write_fd, readable=False))


if __name__ == "__main__":
    main()
