"""Openings and match statistics (the matches themselves are the Rust arbiter's, ygoai-match).

An "opening" is a pair of decks plus a seed (which fixes both starting deck orders and the core's
RNG); a match plays every opening twice with the engines swapping seats. The statistics: W/D/L, an
Elo estimate with a 95% interval from the pentanomial statistics of the game pairs, and the
sequential probability ratio test (GSPRT) between two Elo hypotheses, as ygoai-match computes them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .carddb import default_db
from .decks import Deck, DeckPool


@dataclass
class Opening:
    seed: int
    decks: list[Deck]


def random_openings(n: int, seed: int, mirror_prob: float = 0.0, pool: DeckPool | None = None) -> list[Opening]:
    pool = pool or DeckPool(default_db())
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d0 = pool.sample(rng)
        d1 = d0 if rng.random() < mirror_prob else pool.sample(rng)
        out.append(Opening(seed=seed * 1_000_003 + i, decks=[d0, d1]))
    return out


# ---------------------------------------------------------------------------

def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


@dataclass
class MatchStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    # pentanomial counts of pair scores 0, 0.5, 1, 1.5, 2
    penta: list[int] = field(default_factory=lambda: [0] * 5)

    def add_pair(self, scores: list[float]) -> None:
        for s in scores:
            if s == 1.0:
                self.wins += 1
            elif s == 0.0:
                self.losses += 1
            else:
                self.draws += 1
        self.penta[int(round(sum(scores) * 2))] += 1

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    def _pair_mean_var(self) -> tuple[float, float, int]:
        n = sum(self.penta)
        if n == 0:
            return 0.5, 0.0, 0
        vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        mean = sum(v * c for v, c in zip(vals, self.penta)) / n
        var = sum(c * (v - mean) ** 2 for v, c in zip(vals, self.penta)) / n
        return mean, var, n

    def elo(self) -> tuple[float, float]:
        """Elo difference and half width of the 95% confidence interval."""
        mean, var, n = self._pair_mean_var()
        if n == 0:
            return 0.0, float("inf")
        se = math.sqrt(var / n) if n > 1 else 0.5
        lo, hi = score_to_elo(mean - 1.96 * se), score_to_elo(mean + 1.96 * se)
        return score_to_elo(mean), (hi - lo) / 2

    def llr(self, elo0: float, elo1: float) -> float:
        """Generalized SPRT log-likelihood ratio (normal approximation, pentanomial)."""
        mean, var, n = self._pair_mean_var()
        if n < 2 or var <= 0:
            return 0.0
        s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
        return n * ((mean - s0) ** 2 - (mean - s1) ** 2) / (2 * var)

    def summary(self, elo0: float | None = None, elo1: float | None = None) -> str:
        e, err = self.elo()
        text = (f"games {self.games}: +{self.wins} ={self.draws} -{self.losses}  "
                f"elo {e:+.1f} +/- {err:.1f}  penta {self.penta}")
        if elo0 is not None and elo1 is not None:
            text += f"  LLR {self.llr(elo0, elo1):+.2f}"
        return text


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


__all__ = ["Opening", "MatchStats", "random_openings", "score_to_elo", "elo_to_score", "sprt_bounds"]
