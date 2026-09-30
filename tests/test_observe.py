"""Unit tests of the per-player observation filter (no core needed)."""

import unittest

from ygoai import constants as C
from ygoai.observe import filter_message, filter_prompt, is_hidden_prompt, to_core_response

X = 12345678


def loc(con, loc, seq, pos):
    return {"con": con, "loc": loc, "seq": seq, "pos": pos}


class FilterTest(unittest.TestCase):
    def test_move_to_opponent_hand_is_hidden(self):
        m = {"type": "MOVE", "code": X, "from": loc(1, C.LOCATION_DECK, 5, C.POS_FACEDOWN_DEFENSE),
             "to": loc(1, C.LOCATION_HAND, 2, C.POS_FACEDOWN), "reason": C.REASON_EFFECT}
        self.assertEqual(filter_message(m, 0)["code"], 0)
        self.assertEqual(filter_message(m, 1)["code"], X)

    def test_move_to_grave_is_public(self):
        m = {"type": "MOVE", "code": X, "from": loc(1, C.LOCATION_HAND, 0, C.POS_FACEDOWN),
             "to": loc(1, C.LOCATION_GRAVE, 3, C.POS_FACEUP), "reason": C.REASON_DISCARD}
        self.assertEqual(filter_message(m, 0)["code"], X)

    def test_set_is_hidden(self):
        m = {"type": "MOVE", "code": X, "from": loc(1, C.LOCATION_HAND, 0, C.POS_FACEDOWN),
             "to": loc(1, C.LOCATION_SZONE, 2, C.POS_FACEDOWN), "reason": 0x2000400}
        f = filter_message(m, 0)
        self.assertEqual(f["code"], 0)
        # the stale reason (how the card entered the hand) must not leak
        self.assertEqual(f["reason"], 0)
        s = {"type": "SET", "code": X, **loc(1, C.LOCATION_SZONE, 2, C.POS_FACEDOWN)}
        self.assertEqual(filter_message(s, 0)["code"], 0)

    def test_stale_reason_on_field_moves_is_hidden(self):
        # equipping a monster (mzone -> szone) keeps the reason of how the card
        # once entered the hand; a special summon sets its own reason
        m = {"type": "MOVE", "code": X, "from": loc(1, C.LOCATION_MZONE, 3, C.POS_FACEUP_ATTACK),
             "to": loc(1, C.LOCATION_SZONE, 3, C.POS_FACEUP), "reason": 0x2000400}
        self.assertEqual(filter_message(m, 0)["reason"], 0)
        ss = {**m, "from": loc(1, C.LOCATION_SZONE, 0, C.POS_FACEUP),
              "to": loc(1, C.LOCATION_MZONE, 1, C.POS_FACEUP_ATTACK), "reason": C.REASON_SPSUMMON}
        self.assertEqual(filter_message(ss, 0)["reason"], C.REASON_SPSUMMON)

    def test_pile_positions_are_canonical(self):
        # 8 (starting deck) vs 10 (returned to the deck) must not be distinguishable
        a = {"type": "MOVE", "code": X, "from": loc(1, C.LOCATION_DECK, 5, C.POS_FACEDOWN_DEFENSE),
             "to": loc(1, C.LOCATION_HAND, 2, C.POS_FACEDOWN), "reason": 0}
        b = {**a, "from": loc(1, C.LOCATION_DECK, 5, C.POS_FACEDOWN)}
        self.assertEqual(filter_message(a, 0), filter_message(b, 0))

    def test_opponent_draw_hidden(self):
        m = {"type": "DRAW", "player": 1, "cards": [{"code": X, "pos": C.POS_FACEDOWN}]}
        self.assertEqual(filter_message(m, 0)["cards"][0]["code"], 0)
        self.assertEqual(filter_message(m, 1)["cards"][0]["code"], X)

    def test_opponent_hand_shuffle_hidden(self):
        m = {"type": "SHUFFLE_HAND", "player": 1, "codes": [X, X + 1]}
        self.assertEqual(filter_message(m, 0)["codes"], [0, 0])
        self.assertEqual(filter_message(m, 1)["codes"], [X, X + 1])

    def test_confirm_to_other_player_hidden(self):
        # player 1 looks at its own face-down card: player 0 must not learn it
        m = {"type": "CONFIRM_CARDS", "player": 1,
             "cards": [{"code": X, "con": 1, "loc": C.LOCATION_SZONE, "seq": 1}]}
        self.assertEqual(filter_message(m, 0)["cards"][0]["code"], 0)
        # player 1 reveals a hand card to player 0
        m = {"type": "CONFIRM_CARDS", "player": 0,
             "cards": [{"code": X, "con": 1, "loc": C.LOCATION_HAND, "seq": 1}]}
        self.assertEqual(filter_message(m, 0)["cards"][0]["code"], X)

    def test_card_hint_of_hidden_card_dropped(self):
        m = {"type": "CARD_HINT", **loc(1, C.LOCATION_HAND, 0, C.POS_FACEDOWN),
             "hint_type": C.CHINT_DESC_ADD, "value": X << 4}
        self.assertIsNone(filter_message(m, 0))
        self.assertIsNotNone(filter_message(m, 1))
        # hints emitted while cards are created (location 0) reveal deck contents
        created = {**m, "con": 2, "loc": 0, "seq": 0, "pos": 0}
        self.assertIsNone(filter_message(created, 0))

    def test_facedown_position_change_hidden(self):
        m = {"type": "POS_CHANGE", "code": X, "con": 1, "loc": C.LOCATION_MZONE, "seq": 0,
             "prev_pos": C.POS_FACEDOWN_DEFENSE, "pos": C.POS_FACEDOWN_ATTACK}
        self.assertEqual(filter_message(m, 0)["code"], 0)

    def test_private_hints(self):
        m = {"type": "HINT", "hint_type": C.HINT_SELECTMSG, "player": 1, "data": X}
        self.assertIsNone(filter_message(m, 0))
        m = {"type": "HINT", "hint_type": C.HINT_OPSELECTED, "player": 1, "data": 7}
        self.assertIsNotNone(filter_message(m, 0))
        self.assertIsNone(filter_message(m, 1))

    def test_prompt_strips_opponent_hidden_cards(self):
        p = {"type": "SELECT_CARD", "player": 0, "cancelable": 0, "min": 1, "max": 1,
             "cards": [{"code": X, **loc(1, C.LOCATION_HAND, 0, C.POS_FACEDOWN)},
                       {"code": X + 1, **loc(1, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)},
                       {"code": X + 2, **loc(0, C.LOCATION_DECK, 3, C.POS_FACEDOWN)}]}
        f = filter_prompt(p, 0)
        codes = {(c["con"], c["loc"]): c["code"] for c in f["cards"]}
        self.assertEqual(codes, {(1, C.LOCATION_HAND): 0, (1, C.LOCATION_MZONE): X + 1,
                                 (0, C.LOCATION_DECK): X + 2})

    def test_empty_chain_prompt_is_hidden(self):
        # empty chain windows exist because of the opponent's hidden pending triggers
        empty = {"type": "SELECT_CHAIN", "player": 0, "spe_count": 0, "forced": 0, "chains": []}
        self.assertTrue(is_hidden_prompt(empty))
        self.assertFalse(is_hidden_prompt(dict(empty, forced=1)))
        chain = {"code": X, "desc": 0, "client_mode": 0, **loc(0, C.LOCATION_HAND, 0, C.POS_FACEDOWN)}
        self.assertFalse(is_hidden_prompt(dict(empty, chains=[chain])))

    def test_option_order_is_canonical(self):
        # the core lists chain options by effect id, which reflects hidden history
        a = {"code": X, "desc": 0, "client_mode": 0, **loc(0, C.LOCATION_HAND, 3, C.POS_FACEDOWN)}
        b = {"code": X, "desc": 0, "client_mode": 0, **loc(0, C.LOCATION_HAND, 1, C.POS_FACEDOWN)}
        p = {"type": "SELECT_CHAIN", "player": 0, "spe_count": 0, "forced": 0, "chains": [a, b]}
        f = filter_prompt(p, 0)
        self.assertEqual([c["seq"] for c in f["chains"]], [1, 3])
        # choosing the first option as seen (hand1) is core option 1
        self.assertEqual(to_core_response(p, 0, {"index": 0}), {"index": 1})
        self.assertEqual(to_core_response(p, 0, {"index": -1}), {"index": -1})

    def test_card_selection_order_is_canonical(self):
        a = {"code": X, **loc(0, C.LOCATION_EXTRA, 15, C.POS_FACEUP)}
        b = {"code": X, **loc(0, C.LOCATION_HAND, 1, C.POS_FACEUP)}
        h = {"code": X + 1, **loc(0, C.LOCATION_HAND, 0, C.POS_FACEUP)}
        p = {"type": "SELECT_UNSELECT_CARD", "player": 0, "finishable": 0, "cancelable": 0, "min": 1, "max": 1,
             "select_cards": [h], "unselect_cards": [a, b]}
        f = filter_prompt(p, 0)
        self.assertEqual([c["loc"] for c in f["unselect_cards"]], [C.LOCATION_HAND, C.LOCATION_EXTRA])
        self.assertEqual(to_core_response(p, 0, {"index": 1}), {"index": 2})
        self.assertEqual(to_core_response(p, 0, {"index": 0}), {"index": 0})
        s = {"type": "SELECT_CARD", "player": 0, "cancelable": 0, "min": 1, "max": 2, "cards": [a, h, b]}
        self.assertEqual([c["seq"] for c in filter_prompt(s, 0)["cards"]], [0, 1, 15])
        self.assertEqual(to_core_response(s, 0, {"indices": [0, 2]}), {"indices": [1, 0]})

    def test_unknown_message_raises(self):
        with self.assertRaises(KeyError):
            filter_message({"type": "SOMETHING_NEW"}, 0)


if __name__ == "__main__":
    unittest.main()
