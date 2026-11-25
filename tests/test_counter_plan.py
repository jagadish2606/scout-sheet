import unittest

import pandas as pd

from analysis.counter_plan import confidence_level, generate_counter_cards


class CounterPlanTests(unittest.TestCase):
    def test_confidence_threshold_boundaries(self):
        self.assertEqual(confidence_level(40, 50), "High")
        self.assertEqual(confidence_level(60, 49.9), "Medium")
        self.assertEqual(confidence_level(20, 35), "Medium")
        self.assertIsNone(confidence_level(19, 100))
        self.assertIsNone(confidence_level(40, 34.9))

    def test_utility_cards_count_each_ability_in_its_own_rounds(self):
        rounds = pd.DataFrame([
            {
                "series_id": "s1" if number <= 25 else "s2", "game_id": "g1",
                "round_number": number, "team_id": "c9", "map": "Lotus", "side": "defender",
            }
            for number in range(1, 51)
        ])
        casts = pd.DataFrame([
            {
                "series_id": "s1" if number <= 25 else "s2", "game_id": "g1",
                "round_number": number, "team_id": "c9", "map": "Lotus", "side": "defender",
                "player_id": "p1", "player_name": "OXY", "agent": "cypher",
                "ability": ability, "seconds_into_round": 10.0, "x": 100.0, "y": 100.0,
            }
            for ability, limit in (("trapwire", 30), ("spycam", 10))
            for number in range(1, limit + 1)
        ])
        buys = pd.DataFrame([
            {
                "series_id": row.series_id, "game_id": row.game_id, "round_number": row.round_number,
                "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p1",
                "player_name": "OXY", "agent": "cypher", "has_operator": False,
            }
            for row in rounds.itertuples(index=False)
        ])
        empty_events = pd.DataFrame()
        cards = generate_counter_cards(
            rounds, casts, buys, empty_events, empty_events, pd.DataFrame(), "c9", "Cloud9",
            "Lotus", "defender", {"lotus": {"A": {"x": 200.0, "y": 200.0}}}, self_scout=True,
        )

        utility_cards = [card for card in cards if card["type"] == "Utility"]
        self.assertEqual(len(utility_cards), 1)
        card = utility_cards[0]
        self.assertIn("Expect Trapwire at A site in 60% of defense rounds", card["tendency"])
        self.assertIn("most casts happen 0–20 s", card["tendency"])
        self.assertEqual(card["evidence"], "30/50 rounds (60%), 2 series.")
        self.assertEqual(card["confidence"], "High")
        self.assertTrue(card["response"].startswith("Cloud9 is predictable here — consider varying"))
        self.assertEqual(card["agent"], "cypher")
        self.assertEqual(card["navigation_tab"], "Utility Map")

    def test_operator_and_side_economy_cards_use_round_counts_and_self_scout_language(self):
        rounds = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "g1", "round_number": number,
                "team_id": "c9", "map": "Lotus", "side": side,
            }
            for side, start in (("defender", 1), ("attacker", 51))
            for number in range(start, start + 50)
        ])
        buys = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "g1", "round_number": row.round_number,
                "team_id": "c9", "map": "Lotus", "side": row.side, "player_id": "p1",
                "player_name": "OXY", "agent": "jett",
                "has_operator": row.side == "defender" and row.round_number <= 20
                or row.side == "attacker" and row.round_number <= 52,
            }
            for row in rounds.itertuples(index=False)
        ])
        cards = generate_counter_cards(
            rounds, pd.DataFrame(), buys, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            "c9", "Cloud9", "Lotus", "defender",
        )

        by_type = {card["type"]: card for card in cards}
        self.assertIn("Operator", by_type)
        self.assertIn("Side economy", by_type)
        self.assertEqual(by_type["Operator"]["confidence"], "Medium")
        self.assertIn("OXY holds Cloud9's Operator in 100%", by_type["Operator"]["tendency"])
        self.assertIn("20/20", by_type["Operator"]["tendency"])
        self.assertNotIn("Early/late", by_type["Operator"]["tendency"])
        self.assertNotIn("first 20 s", by_type["Operator"]["response"])
        self.assertIn("40% of defense rounds versus 4% on attack", by_type["Side economy"]["tendency"])

        sparse_buys = buys[buys["side"].eq("defender") | buys["round_number"].le(55)]
        sparse_cards = generate_counter_cards(
            rounds, pd.DataFrame(), sparse_buys, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            "c9", "Cloud9", "Lotus", "defender",
        )
        self.assertNotIn("Side economy", {card["type"] for card in sparse_cards})

    def test_cards_sort_by_confidence_then_frequency_and_limit_to_eight(self):
        rounds = pd.DataFrame([
            {
                "series_id": "s1" if number <= 25 else "s2", "game_id": "g1",
                "round_number": number, "team_id": "c9", "map": "Lotus", "side": "defender",
            }
            for number in range(1, 51)
        ])
        casts = pd.DataFrame([
            {
                "series_id": "s1" if number <= 25 else "s2", "game_id": "g1",
                "round_number": number, "team_id": "c9", "map": "Lotus", "side": "defender",
                "player_id": "p1", "player_name": "OXY", "agent": "cypher",
                "ability": "trapwire", "seconds_into_round": 10.0,
                "x": cell * 700 + 100.0, "y": 100.0,
            }
            for cell in range(9)
            for number in range(1, (41 if cell == 0 else 26))
        ])
        buys = pd.DataFrame([
            {
                "series_id": "s1" if number <= 25 else "s2", "game_id": "g1",
                "round_number": number, "team_id": "c9", "map": "Lotus", "side": "defender",
                "player_id": "p1", "player_name": "OXY", "agent": "cypher", "has_operator": False,
            }
            for number in range(1, 51)
        ])

        cards = generate_counter_cards(
            rounds, casts, buys, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            "c9", "Cloud9", "Lotus", "defender",
        )

        self.assertEqual(len(cards), 8)
        self.assertEqual(cards[0]["confidence"], "High")
        self.assertEqual(cards[0]["frequency"], 80.0)
        self.assertTrue(all(card["confidence_order"] == 0 for card in cards))


if __name__ == "__main__":
    unittest.main()