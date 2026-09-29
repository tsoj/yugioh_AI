"""Integration tests (need build/libocgcore.so, see scripts/build_core.sh)."""

import random
import sys
import unittest

from ygoai.agents import GreedyAgent, RandomAgent
from ygoai.carddb import default_db
from ygoai.decks import DeckPool
from ygoai.game import ObservationLog, init_info, new_record, play, replay
from ygoai.match import MatchStats
from ygoai.protocol import EngineProcess


def make_record(seed, agents=("random", "greedy")):
    pool = DeckPool(default_db())
    rng = random.Random(seed)
    rec = new_record([pool.sample(rng), pool.sample(rng)], seed)
    cls = {"random": RandomAgent, "greedy": GreedyAgent}
    play(rec, [cls[agents[0]](seed), cls[agents[1]](seed + 1)])
    return rec


class GameTest(unittest.TestCase):
    def test_game_finishes_cleanly(self):
        rec = make_record(1)
        self.assertEqual(rec.result["retries"], 0)
        self.assertEqual(rec.result["script_errors"], 0)
        self.assertIn(rec.result["winner"], (0, 1, None))

    def test_replay_is_deterministic(self):
        rec = make_record(2)
        logs = []
        for _ in range(2):
            log = ObservationLog(0)
            replay(rec, [log]).close()
            logs.append(log.items)
        self.assertEqual(logs[0], logs[1])

    def test_record_roundtrip(self):
        from ygoai.game import GameRecord
        rec = make_record(3)
        rec2 = GameRecord.from_json(rec.to_json())
        a, b = ObservationLog(1), ObservationLog(1)
        replay(rec, [a]).close()
        replay(rec2, [b]).close()
        self.assertEqual(a.items, b.items)


class ProtocolTest(unittest.TestCase):
    def test_external_engine(self):
        pool = DeckPool(default_db())
        rng = random.Random(4)
        rec = new_record([pool.sample(rng), pool.sample(rng)], 4)
        engine = EngineProcess([sys.executable, "-m", "ygoai.engine", "random", "1"])
        try:
            play(rec, [engine, GreedyAgent(5)], illegal_is_loss=True)
        finally:
            engine.close()
        self.assertNotIn("agent_error", str(rec.result["reason"]))
        self.assertNotEqual(rec.result["reason"], "illegal_response")


class StatsTest(unittest.TestCase):
    def test_elo(self):
        s = MatchStats()
        for i in range(30):
            s.add_pair([1.0, 0.5] if i % 3 else [0.0, 1.0])
        e, err = s.elo()
        self.assertGreater(e, 0)
        self.assertGreater(s.llr(0, 50), 0)


class BeliefTest(unittest.TestCase):
    def test_belief_replay_consistent(self):
        from ygoai.belief.check import check_game
        rec = make_record(7, ("greedy", "greedy"))
        results = check_game(rec, fractions=(0.5, 1.0))
        failed = [r.error for r in results if not r.ok]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
