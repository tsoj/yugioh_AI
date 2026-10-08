"""The policy/value network (numpy inference; trained by scripts/train_policy_value.py).

- cards: [static features | dynamic features] -> MLP -> card embedding; summed
  per group (decks by count) -> [groups x dim | scalars] -> trunk MLP -> h
- value: h -> logit of "the player to decide wins"
- policy: h -> query q; a candidate's embedding is the sum of embeddings of its
  parts (decision type, kind, select hint, effect index, card location, card is
  own) plus a linear map of its card's features; its score is q . embedding
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import features as F

ACT_CATS = (len(F.PROMPTS), len(F.KINDS), F.HINTS, F.EFFECTS)
N_LOC = len(F.LOCATIONS) + 1  # (location index + 1; 0: no card)


def _relu(x):
    return np.maximum(x, 0.0)


class PolicyValueNet:
    def __init__(self, path: str | Path):
        path = Path(path)
        w = np.load(path)
        self.w = {k: w[k].astype(np.float32) for k in w.files if k != "codes"}
        self.codes = [int(c) for c in w["codes"]]
        self.index = {c: i for i, c in enumerate(self.codes)}
        self.meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        self._static_cache: dict[int, np.ndarray] = {}

    def _static(self, codes: list[int]) -> np.ndarray:
        """Static features of cards (from the table, else from the card database)."""
        out = np.empty((len(codes), F.STATIC_DIM), dtype=np.float32)
        table = self.w["static_table"]
        for i, c in enumerate(codes):
            j = self.index.get(c)
            if j is not None:
                out[i] = table[j]
            else:
                row = self._static_cache.get(c)
                if row is None:
                    from ..carddb import default_db
                    row = self._static_cache[c] = np.asarray(
                        F.static_features(default_db().get(c) if c else None), dtype=np.float32)
                out[i] = row
        return out

    def _cards(self, static: np.ndarray, raw) -> np.ndarray:
        x = np.concatenate([static, F.expand_dynamic(raw)], axis=1)
        h = _relu(x @ self.w["card.0.weight"].T + self.w["card.0.bias"])
        return h @ self.w["card.2.weight"].T + self.w["card.2.bias"]

    def trunk(self, cards: list[tuple], scalars: list[float]) -> np.ndarray:
        """cards: (code, group, count, raw) as features.position gives them (with codes as codes)."""
        dim = self.w["card.2.bias"].shape[0]
        pooled = np.zeros((len(F.GROUPS), dim), dtype=np.float32)
        if cards:
            emb = self._cards(self._static([c[0] for c in cards]), [c[3] for c in cards])
            groups = np.asarray([c[1] for c in cards])
            counts = np.asarray([c[2] for c in cards], dtype=np.float32)[:, None]
            np.add.at(pooled, groups, emb * counts)
        x = np.concatenate([pooled.reshape(-1), np.asarray(scalars, dtype=np.float32)])
        h = _relu(x @ self.w["trunk.0.weight"].T + self.w["trunk.0.bias"])
        return _relu(h @ self.w["trunk.2.weight"].T + self.w["trunk.2.bias"])

    def value(self, h: np.ndarray) -> float:
        """Logit of "the player to decide wins"."""
        return float(h @ self.w["value.weight"][0] + self.w["value.bias"][0])

    def scores(self, h: np.ndarray, acts: list[tuple]) -> np.ndarray:
        """acts: features.actions' tuples (with codes as codes) -> one score each."""
        q = h @ self.w["query.weight"].T + self.w["query.bias"]
        cat = np.asarray([a[:4] for a in acts])
        e = sum(self.w[f"act_cat.{i}.weight"][cat[:, i]] for i in range(4))
        e = e + self.w["act_loc.weight"][np.asarray([a[5] + 1 for a in acts])]
        e = e + self.w["act_own.weight"][np.asarray([int(a[6]) for a in acts])]
        has = np.asarray([a[4] != 0 for a in acts], dtype=np.float32)[:, None]
        x = np.concatenate([self._static([a[4] for a in acts]), F.expand_dynamic([a[7] for a in acts])], axis=1)
        e = e + has * (x @ self.w["act_card.weight"].T)
        return e @ q
