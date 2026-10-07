"""Deck completion by the learned deck models.

The models and their inference code live with the deck corpus
(download_human_decks/complete_deck.py, retrained there): an existence model
predicts which cards a deck holds given the ones seen so far, a count model how
many copies of each. Their guesses are realistic, which the AI's belief state
needs; the replay's own constraints apply on top, as for the corpus completion:
no win-condition pieces, no card that adds a duel-wide effect no known card
brings (see script_index), only cards the card database knows.

Inference runs on the CPU, single-threaded and deterministic: a replay must
come out the same in every run (a GPU is for training).
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from ..carddb import CardDB, default_db
from ..decks import canonical_code
from ..paths import ROOT
from .completion import DeckCompletion

MODELS_DIR = Path(os.environ.get("YGOAI_DECK_MODELS", ROOT / "download_human_decks"))


class NeuralCompletion(DeckCompletion):
    def __init__(self, db: CardDB | None = None, models_dir: Path = MODELS_DIR):
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0, str(models_dir))
        import complete_deck as cd
        from train_deck_counts import CountModel
        from train_deck_existence import ExistenceModel
        self.cd = cd
        self.db = db or default_db()
        self.ex, ex_ckpt = cd.load_model(ExistenceModel, models_dir / "deck_existence.pt", "cpu")
        self.cnt, cnt_ckpt = cd.load_model(CountModel, models_dir / "deck_counts.pt", "cpu")
        if ex_ckpt["vocab"] != cnt_ckpt["vocab"] or ex_ckpt["is_extra"] != cnt_ckpt["is_extra"]:
            raise RuntimeError("the deck models were trained on different vocabularies")
        self.vocab, self.is_extra = ex_ckpt["vocab"], ex_ckpt["is_extra"]
        self.avg_copies = cd.compute_avg_copies_per_card(models_dir / "decks", self.vocab)
        self._fallback = None
        self._cache: dict = {}

    def _corpus(self):
        if self._fallback is None:
            from .corpus import CorpusCompletion
            self._fallback = CorpusCompletion(self.db)
        return self._fallback

    def stand_ins(self, need: tuple, known: list[int] = ()) -> list[int]:
        """The cards of that kind the deck most likely holds, given its known
        cards (the existence model), then effect-less ones as before."""
        from .completion import fits_need
        from .script_index import global_keys, win_condition_codes
        canon = [canonical_code(self.db, c) for c in known]
        key = ("stand_ins", need, tuple(sorted(canon)))
        if key not in self._cache:
            wins = win_condition_codes()
            have = set().union(*(global_keys(c) for c in canon))
            out = []
            for code in self._ranked(canon):
                card = self.db.get(code)
                if card is not None and code not in wins and global_keys(code) <= have and fits_need(card, need):
                    out.append(code)
                    if len(out) >= 12:
                        break
            self._cache[key] = out + [c for c in super().stand_ins(need) if c not in out]
        return list(self._cache[key])

    def _ranked(self, canon: list[int]) -> list[int]:
        """Every card of the vocabulary, the likeliest to be in the deck first."""
        import torch
        V = len(self.vocab)
        c_vis = torch.zeros(V, dtype=torch.long)
        for c in canon:
            j = self.vocab.get(c)
            if j is not None and c_vis[j] < 3:
                c_vis[j] += 1
        with torch.no_grad():
            presence = self.cd.predict_presence(self.ex, c_vis, 55, "cpu")
        inv = {i: c for c, i in self.vocab.items()}
        order = torch.argsort(presence, descending=True, stable=True).tolist()
        return [inv[j] for j in order if c_vis[j] == 0]

    def complete(self, known_main, known_extra, n_main, n_extra, present=()):
        key = (tuple(sorted(known_main)), tuple(sorted(known_extra)), n_main, n_extra,
               tuple(sorted(set(present))))
        if key not in self._cache:
            self._cache[key] = self._complete(known_main, known_extra, n_main, n_extra, present)
        fill_main, fill_extra = self._cache[key]
        return list(fill_main), list(fill_extra)

    def _complete(self, known_main, known_extra, n_main, n_extra, present):
        import torch
        from .script_index import global_keys, win_condition_codes
        canon = lambda c: canonical_code(self.db, c)  # noqa: E731
        km = [canon(c) for c in known_main]
        ke = [canon(c) for c in known_extra]
        with torch.no_grad():
            main, extra, _ = self.cd.complete_deck(
                self.ex, self.cnt, self.vocab, self.is_extra, km + ke,
                main_size=len(km) + n_main, extra_size=len(ke) + n_extra,
                avg_copies=self.avg_copies, max_iter=60, device="cpu")
        wins = win_condition_codes()
        have = set().union(*(global_keys(canon(c)) for c in list(present) + km + ke))

        def usable(code):
            return self.db.get(code) is not None and code not in wins and global_keys(code) <= have

        def fill(model_cards, known, n, extra):
            rest = Counter(model_cards) - Counter(known)
            out = [c for c in model_cards if rest[c] > 0 and not rest.subtract([c]) and usable(c)]
            # (in the model's order; what it could not supply, the corpus completion does)
            if len(out) < n:
                counts = Counter(known) + Counter(out)
                spare = self._corpus().complete(known_main, known_extra, n_main if not extra else 0,
                                                n_extra if extra else 0, present)[1 if extra else 0]
                for c in spare:
                    if len(out) >= n:
                        break
                    if counts[canon(c)] < 3:
                        out.append(c)
                        counts[canon(c)] += 1
            return out[:n]
        return fill(main, km, n_main, False), fill(extra, ke, n_extra, True)


# -- one model process per run ---------------------------------------------------
# Each pool worker loading PyTorch and the models costs ~1.5 GB; a run with dozens
# of workers would not fit in memory. The run's parent starts one model process
# (ensure_service, before its worker pool) and the workers ask it over a local
# socket (neural_completion). The answers are the same, and the service's cache
# serves every worker.

_ENV = "YGOAI_NEURAL_SERVICE"  # (address|authkey of the run's model process)


class RemoteNeuralCompletion(DeckCompletion):
    """NeuralCompletion's answers from the run's model process."""

    def __init__(self, spec: str, db: CardDB | None = None):
        address, key = spec.split("|")
        self.address, self.authkey = "\0" + address, bytes.fromhex(key)  # (an abstract socket)
        self.db = db or default_db()
        self._conn = None
        self._conn_pid = None
        self._cache: dict = {}

    def _ask(self, *request):
        if request not in self._cache:
            from multiprocessing.connection import Client
            if self._conn is None or self._conn_pid != os.getpid():
                # (a forked probe opens its own connection: never the parent's socket)
                self._conn = Client(self.address, authkey=self.authkey)
                self._conn_pid = os.getpid()
            self._conn.send(request)
            answer = self._conn.recv()
            if isinstance(answer, Exception):
                raise answer
            self._cache[request] = answer
        return self._cache[request]

    def complete(self, known_main, known_extra, n_main, n_extra, present=()):
        fill_main, fill_extra = self._ask("complete", tuple(known_main), tuple(known_extra), n_main, n_extra,
                                          tuple(sorted(set(present))))
        return list(fill_main), list(fill_extra)

    def stand_ins(self, need: tuple, known: list[int] = ()) -> list[int]:
        return list(self._ask("stand_ins", tuple(need), tuple(sorted(known))))


def neural_completion(db: CardDB | None = None) -> DeckCompletion:
    """The run's model process if there is one, else the models in this process."""
    spec = os.environ.get(_ENV)
    return RemoteNeuralCompletion(spec, db) if spec else NeuralCompletion(db)


def ensure_service() -> None:
    """Starts the model process for this run and its children (once; it ends
    with this process)."""
    if os.environ.get(_ENV):
        return
    import atexit
    import secrets
    import subprocess
    address = f"ygoai-neural-{os.getpid()}-{secrets.token_hex(4)}"
    key = secrets.token_hex(16)
    proc = subprocess.Popen([sys.executable, "-m", "ygoai.belief.neural_completion", address, key,
                             str(os.getpid())], stdout=subprocess.PIPE, text=True,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), cwd=str(ROOT))
    line = proc.stdout.readline().strip()
    if line != "ready":
        proc.kill()
        raise RuntimeError(f"the deck model process did not start ({line!r})")
    atexit.register(proc.kill)
    os.environ[_ENV] = f"{address}|{key}"


def _serve(address: str, key: str, parent: int) -> None:
    import threading
    import time
    from multiprocessing.connection import Listener
    model = NeuralCompletion()
    lock = threading.Lock()
    listener = Listener("\0" + address, authkey=bytes.fromhex(key))

    def client(conn):
        with conn:
            while True:
                try:
                    request = conn.recv()
                except (EOFError, OSError):
                    return
                try:
                    with lock:
                        if request[0] == "complete":
                            answer = model.complete(list(request[1]), list(request[2]), request[3], request[4],
                                                    list(request[5]))
                        else:
                            answer = model.stand_ins(request[1], list(request[2]))
                except Exception as e:  # (to the asking worker)
                    answer = e
                conn.send(answer)

    def watch():
        while True:
            time.sleep(2)
            try:
                os.kill(parent, 0)
            except OSError:
                os._exit(0)  # (the run is over)
    threading.Thread(target=watch, daemon=True).start()
    print("ready", flush=True)
    while True:
        conn = listener.accept()
        threading.Thread(target=client, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    _serve(sys.argv[1], sys.argv[2], int(sys.argv[3]))
