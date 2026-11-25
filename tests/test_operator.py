import unittest

import pandas as pd

from analysis.operator import (
    aggressiveness_label,
    aggressiveness_score,
    calculate_buy_rates,
    calculate_handoff_rate,
    estimate_spawn_anchors,
    operator_engagements,
)


class OperatorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rounds = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 1, "team_id": "c9", "map": "Lotus", "side": "attacker"},
            {"series_id": "s1", "game_id": "g1", "round_number": 2, "team_id": "c9", "map": "Lotus", "side": "attacker"},
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "team_id": "c9", "map": "Lotus", "side": "defender"},
            {"series_id": "s1", "game_id": "g1", "round_number": 4, "team_id": "c9", "map": "Lotus", "side": "defender"},
            {"series_id": "s1", "game_id": "g2", "round_number": 1, "team_id": "c9", "map": "Lotus", "side": "defender"},
        ])
        self.buys = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 1, "team_id": "c9", "map": "Lotus", "side": "attacker", "player_id": "p1", "player_name": "OXY", "agent": "jett", "has_operator": False},
            {"series_id": "s1", "game_id": "g1", "round_number": 2, "team_id": "c9", "map": "Lotus", "side": "attacker", "player_id": "p1", "player_name": "OXY", "agent": "jett", "has_operator": True},
            {"series_id": "s1", "game_id": "g1", "round_number": 2, "team_id": "c9", "map": "Lotus", "side": "attacker", "player_id": "p2", "player_name": "Xeppaa", "agent": "omen", "has_operator": False},
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p1", "player_name": "OXY", "agent": "jett", "has_operator": False},
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p2", "player_name": "Xeppaa", "agent": "omen", "has_operator": True},
            {"series_id": "s1", "game_id": "g2", "round_number": 1, "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p2", "player_name": "Xeppaa", "agent": "omen", "has_operator": True},
        ])

    def test_buy_rate_uses_observed_snapshot_rounds_and_reports_coverage(self):
        rates = calculate_buy_rates(self.rounds, self.buys, "c9", "lotus")

        self.assertEqual(rates["overall"]["eligible_rounds"], 5)
        self.assertEqual(rates["overall"]["observed_rounds"], 4)
        self.assertEqual(rates["overall"]["held_rounds"], 3)
        self.assertEqual(rates["overall"]["rate"], 75)
        self.assertEqual(rates["overall"]["coverage"], 80)
        self.assertEqual(rates["by_side"]["attacker"]["held_rounds"], 1)
        self.assertEqual(rates["by_side"]["defender"]["held_rounds"], 2)
        self.assertEqual(rates["by_side"]["defender"]["eligible_rounds"], 3)
        self.assertEqual(rates["by_side"]["defender"]["observed_rounds"], 2)
        self.assertEqual(len(rates["strip_rounds"]), 5)
        missing_snapshot = rates["strip_rounds"].query("game_id == 'g1' and round_number == '4'").iloc[0]
        self.assertFalse(missing_snapshot["snapshot_available"])
        oxy = rates["players"].set_index("player_name").loc["OXY"]
        self.assertEqual(oxy["rounds"], 1)
        self.assertAlmostEqual(oxy["share"], 100 / 3)

    def test_handoff_rate_compares_op_rounds_inside_each_game(self):
        rates = calculate_buy_rates(self.rounds, self.buys, "c9", "Lotus")
        handoff = calculate_handoff_rate(rates["rounds"])

        self.assertEqual(handoff["handoffs"], 1)
        self.assertEqual(handoff["comparisons"], 1)
        self.assertEqual(handoff["rate"], 100)

    def test_handoff_rate_orders_round_numbers_numerically(self):
        round_holders = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": "2", "holders": ("p1",)},
            {"series_id": "s1", "game_id": "g1", "round_number": "10", "holders": ("p2",)},
            {"series_id": "s1", "game_id": "g1", "round_number": "11", "holders": ("p1",)},
        ])

        handoff = calculate_handoff_rate(round_holders)

        self.assertEqual(handoff["handoffs"], 2)
        self.assertEqual(handoff["comparisons"], 2)
        self.assertEqual(handoff["rate"], 100)

    def test_handoff_does_not_bridge_a_missing_buy_snapshot(self):
        round_holders = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 1, "holders": ("p1",), "snapshot_available": True},
            {"series_id": "s1", "game_id": "g1", "round_number": 2, "holders": (), "snapshot_available": False},
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "holders": ("p2",), "snapshot_available": True},
        ])

        handoff = calculate_handoff_rate(round_holders)

        self.assertEqual(handoff["handoffs"], 0)
        self.assertEqual(handoff["comparisons"], 0)

    def test_player_and_agent_shares_are_independent_breakdowns(self):
        rounds = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": number, "team_id": "c9", "map": "Lotus", "side": "defender"}
            for number in (1, 2)
        ])
        buys = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 1, "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p1", "player_name": "OXY", "agent": "jett", "has_operator": True},
            {"series_id": "s1", "game_id": "g1", "round_number": 2, "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p1", "player_name": "OXY", "agent": "yoru", "has_operator": True},
        ])

        rates = calculate_buy_rates(rounds, buys, "c9", "Lotus")

        self.assertEqual(len(rates["players"]), 1)
        self.assertEqual(rates["players"].iloc[0]["rounds"], 2)
        self.assertEqual(set(rates["agents"]["agent"]), {"jett", "yoru"})
        self.assertTrue(rates["agents"]["share"].eq(50).all())

    def test_complete_snapshots_with_no_operator_return_empty_share_tables(self):
        buys = self.buys.copy()
        buys["has_operator"] = False

        rates = calculate_buy_rates(self.rounds, buys, "c9", "Lotus")

        self.assertEqual(rates["overall"]["observed_rounds"], 4)
        self.assertEqual(rates["overall"]["held_rounds"], 0)
        self.assertTrue(rates["players"].empty)
        self.assertTrue(rates["agents"].empty)

    def test_aggressiveness_score_thresholds_and_missing_spawns(self):
        defender_spawn = (0, 0)
        attacker_spawn = (10, 0)

        passive = aggressiveness_score((2, 0), defender_spawn, attacker_spawn)
        standard = aggressiveness_score((5, 0), defender_spawn, attacker_spawn)
        aggressive = aggressiveness_score((8, 0), defender_spawn, attacker_spawn)
        self.assertEqual(aggressiveness_label(passive), "passive")
        self.assertEqual(aggressiveness_label(standard), "standard")
        self.assertEqual(aggressiveness_label(aggressive), "aggressive")
        self.assertIsNone(aggressiveness_score((1, 2), None, attacker_spawn))
        self.assertIsNone(aggressiveness_score((1, 2), defender_spawn, defender_spawn))

    def test_spawn_anchors_use_map_side_medians(self):
        positions = pd.DataFrame([
            {"map": "Lotus", "side": "defender", "x": 1, "y": 10},
            {"map": "lotus", "side": "defender", "x": 3, "y": 14},
            {"map": "lotus", "side": "attacker", "x": 100, "y": 200},
        ])

        anchors = estimate_spawn_anchors(positions)
        self.assertEqual(anchors["lotus"]["defender"], (2.0, 12.0))
        self.assertEqual(anchors["lotus"]["attacker"], (100.0, 200.0))

    def test_operator_engagements_require_freeze_time_holder_and_deduplicate_ticks(self):
        buys = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "team_id": "c9", "map": "Lotus", "has_operator": True, "player_id": "p1"},
        ])
        kills = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "map": "Lotus", "killer_team_id": "c9", "killer_id": "p1", "killer_name": "OXY", "killer_side": "defender", "killer_agent": "jett", "victim_id": "enemy", "weapon": "operator", "seconds_into_round": 10, "killer_x": 20, "killer_y": 30},
        ])
        damage = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "map": "Lotus", "team_id": "c9", "player_id": "p1", "player_name": "OXY", "side": "defender", "agent": "jett", "victim_id": "enemy", "weapon": "operator", "seconds_into_round": 9, "x": 18, "y": 28},
            {"series_id": "s1", "game_id": "g1", "round_number": 3, "map": "Lotus", "team_id": "c9", "player_id": "p1", "player_name": "OXY", "side": "defender", "agent": "jett", "victim_id": "enemy", "weapon": "operator", "seconds_into_round": 9.5, "x": 19, "y": 29},
        ])

        engagements = operator_engagements(kills, damage, buys, "c9", "Lotus")

        self.assertEqual(len(engagements), 1)
        self.assertEqual(engagements.iloc[0]["event_type"], "kill")
        self.assertEqual(engagements.iloc[0]["source_events"], 3)


if __name__ == "__main__":
    unittest.main()