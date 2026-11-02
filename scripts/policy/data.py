"""The policy data of crates/bot/src/bin/bot-policy-data.rs: its shards as numpy arrays.

  data = PolicyData("runs/policy/DIR")
  data.hdr  # (decisions, HDR) int64, columns H[...]
  data.tok  # (tokens, TOK) int16, columns T[...]
  data.opt  # (options, OPT) int32, columns O[...]
  data.games[g]  # {"codes": [...], "decks": [...], ...}

A decision's token and option rows start at its H_TOK_START / H_OPT_START, already offset into
the concatenated arrays here. Code indices (T_CODE, ...) are index + 1 into its game's "codes".
"""

from __future__ import annotations

import json
import os

import numpy as np

# (the columns, as in bot-policy-data.rs)
H = {n: i for i, n in enumerate(
    "GAME DECISION PLAYER TURN TURN_PLAYER PHASE LP_ME LP_OPP PROMPT MIN MAX FLAGS HINT_CODE HINT_STR "
    "DESC_CODE DESC_STR DEPTH VALUE RESULT TOK_START N_TOK OPT_START N_OPT PREV_KIND PREV_CODE DECK_ME "
    "DECK_OPP EXTRA_OPP CODE".split())}
T = {n: i for i, n in enumerate(
    "CODE CON OWNER LOC SEQ POS ATK DEF BASE_ATK BASE_DEF LEVEL SCALE COUNTERS OVERLAYS REF STATUS COUNT".split())}
O = {n: i for i, n in enumerate("KIND TOKEN AUX DESC_CODE DESC_STR CHOSEN".split())}

LOCS = ["-", "hand", "mzone", "szone", "grave", "removed", "extra", "overlay", "deck", "chain", "event"]
# (a text reference to the card itself, H_HINT_STR / O_DESC_STR)
S_CARD = 16
KINDS = ["summon", "spsummon", "repos", "mset", "sset", "activate", "to_bp", "to_ep", "shuffle",
         "b_activate", "attack", "to_m2", "b_to_ep", "chain_pass", "chain", "yes", "no", "option", "card",
         "select", "unselect", "finish", "place", "position", "tribute", "sum", "number", "attribute", "race"]
N_KINDS = len(KINDS)


class PolicyData:
    def __init__(self, path: str):
        meta = json.load(open(os.path.join(path, "meta.json")))
        assert (meta["hdr"], meta["tok"], meta["opt"]) == (len(H), len(T), len(O)), meta
        self.prompts: list[str] = meta["prompts"]
        shards = []
        while os.path.exists(os.path.join(path, f"{len(shards)}.hdr")):
            shards.append(os.path.join(path, str(len(shards))))
        # (the tokens read into one array: concatenating the shards would need twice the memory)
        sizes = [os.path.getsize(f"{s}.tok") // (2 * len(T)) for s in shards]
        self.tok = np.empty((sum(sizes), len(T)), dtype="<i2")
        hdrs, opts = [], []
        n_tok = n_opt = 0
        for s, size in zip(shards, sizes):
            h = np.fromfile(f"{s}.hdr", dtype="<i8").reshape(-1, len(H))
            o = np.fromfile(f"{s}.opt", dtype="<i4").reshape(-1, len(O))
            with open(f"{s}.tok", "rb") as f:
                f.readinto(memoryview(self.tok[n_tok:n_tok + size]).cast("B"))
            h[:, H["TOK_START"]] += n_tok
            h[:, H["OPT_START"]] += n_opt
            n_tok += size
            n_opt += len(o)
            hdrs.append(h), opts.append(o)
        self.hdr = np.concatenate(hdrs)
        self.opt = np.concatenate(opts)
        self.games: dict[int, dict] = {}
        with open(os.path.join(path, "games.jsonl")) as f:
            for line in f:
                g = json.loads(line)
                self.games[g["game"]] = g

    def tokens(self, i: int) -> np.ndarray:
        h = self.hdr[i]
        return self.tok[h[H["TOK_START"]]:h[H["TOK_START"]] + h[H["N_TOK"]]]

    def options(self, i: int) -> np.ndarray:
        h = self.hdr[i]
        return self.opt[h[H["OPT_START"]]:h[H["OPT_START"]] + h[H["N_OPT"]]]

    def code(self, game: int, index: int) -> int:
        return self.games[game]["codes"][index - 1] if index > 0 else 0
