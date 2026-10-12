"""Rules of the patched C++ core (need build/libocgcore.so, see scripts/build_core.sh)."""

import unittest

from ygoai import constants as C
from ygoai.actions import encode
from ygoai.core import CppDuel as Duel
from ygoai.game import run_until_prompt

MYSTICAL_ELF = 15025844  # (a Level 4 Normal Monster)


def duel_with_monster() -> Duel:
    duel = Duel((1, 2, 3, 4))
    for p in (0, 1):
        for _ in range(40):
            duel.new_card(p, MYSTICAL_ELF, C.LOCATION_DECK)
    duel.new_card(0, MYSTICAL_ELF, C.LOCATION_MZONE, position=C.POS_FACEUP_ATTACK)
    duel.start()
    run_until_prompt(duel)
    return duel


def answer(duel: Duel, prompt: dict, resp: dict | None) -> dict:
    """Answers `prompt` (None: nothing to answer) and returns the next prompt
    that is not a chain (chains are always declined)."""
    while True:
        if resp is not None:
            duel.set_response(encode(prompt, resp))
            step = run_until_prompt(duel, prompt)
            assert not step.ended
            prompt = step.prompt
        if prompt["type"] != "SELECT_CHAIN":
            return prompt
        resp = {"index": -1}


LEVEL = """Debug.Message(tostring(Duel.GetFieldCard(0, LOCATION_MZONE, 0):GetLevel()))"""


class TimestampTest(unittest.TestCase):
    """An effect whose condition holds again starts applying anew (a new
    timestamp), checked at fixed points of the game, not by lazy evaluations."""

    def test_condition_renews_timestamp(self):
        duel = duel_with_monster()
        out = duel.lua_messages("""
            local c = Duel.GetFieldCard(0, LOCATION_MZONE, 0)
            local function level() Debug.Message(tostring(c:GetLevel())) end
            ygoai_cond = true
            local e1 = Effect.CreateEffect(c)
            e1:SetType(EFFECT_TYPE_SINGLE)
            e1:SetCode(EFFECT_UPDATE_LEVEL)
            e1:SetValue(2)
            e1:SetCondition(function() return ygoai_cond end)
            c:RegisterEffect(e1)
            Duel.AdjustInstantly()
            local e2 = Effect.CreateEffect(c)
            e2:SetType(EFFECT_TYPE_SINGLE)
            e2:SetCode(EFFECT_CHANGE_LEVEL)
            e2:SetValue(2)
            c:RegisterEffect(e2)
            Duel.AdjustInstantly()
            level()  -- "+2", then "becomes 2"
            ygoai_cond = false
            c:GetLevel()  -- (a lazy evaluation while the condition fails)
            ygoai_cond = true
            Duel.AdjustInstantly()
            level()  -- unchanged: no fixed point saw the condition fail
            ygoai_cond = false
            Duel.AdjustInstantly()
            level()  -- only "becomes 2"
            ygoai_cond = true
            level()  -- (lazy: the "+2" applies with its old timestamp)
            Duel.AdjustInstantly()
            level()  -- "becomes 2", then the renewed "+2"
        """)
        self.assertEqual(out, ["2", "2", "2", "2", "4"])
        duel.close()

    def test_phase_condition(self):
        """A monster with "+2 Level during the Main Phase" whose Level becomes 2
        in the Battle Phase: in Main Phase 2 the "+2" applies anew, after it."""
        duel = Duel((1, 2, 3, 4))
        for p in (0, 1):
            for _ in range(40):
                duel.new_card(p, MYSTICAL_ELF, C.LOCATION_DECK)
        duel.new_card(0, MYSTICAL_ELF, C.LOCATION_MZONE, position=C.POS_FACEUP_ATTACK)
        duel.start()
        prompt = run_until_prompt(duel).prompt
        prompt = answer(duel, prompt, None)
        self.assertEqual(prompt["type"], "SELECT_IDLECMD")
        duel.lua_messages("""
            local c = Duel.GetFieldCard(0, LOCATION_MZONE, 0)
            local e1 = Effect.CreateEffect(c)
            e1:SetType(EFFECT_TYPE_SINGLE)
            e1:SetCode(EFFECT_UPDATE_LEVEL)
            e1:SetValue(2)
            e1:SetCondition(function() return Duel.IsMainPhase() end)
            c:RegisterEffect(e1)
        """)
        self.assertEqual(duel.lua_messages(LEVEL), ["6"])
        # turn 1: to the End Phase; turn 2 (player 1): to the Battle Phase
        prompt = answer(duel, prompt, {"kind": "to_ep"})
        self.assertEqual(duel.info()["turn"], 2)
        prompt = answer(duel, prompt, {"kind": "to_bp"})
        self.assertEqual(prompt["type"], "SELECT_BATTLECMD")
        self.assertEqual(duel.info()["phase"], C.PHASE_BATTLE_STEP)
        out = duel.lua_messages("""
            local c = Duel.GetFieldCard(0, LOCATION_MZONE, 0)
            local e2 = Effect.CreateEffect(c)
            e2:SetType(EFFECT_TYPE_SINGLE)
            e2:SetCode(EFFECT_CHANGE_LEVEL)
            e2:SetValue(2)
            c:RegisterEffect(e2)
        """ + LEVEL)
        self.assertEqual(out, ["2"])
        prompt = answer(duel, prompt, {"kind": "to_m2"})
        self.assertEqual(prompt["type"], "SELECT_IDLECMD")
        self.assertEqual(duel.info()["phase"], C.PHASE_MAIN2)
        self.assertEqual(duel.lua_messages(LEVEL), ["4"])
        duel.close()


if __name__ == "__main__":
    unittest.main()
