import unittest

import pandas as pd

from analysis.counter_plan import confidence_level, generate_counter_cards, generate_counter_plan


class CounterPlanTests(unittest.TestCase):
    @staticmethod
    def _operator_profile_tables(early=False, aggressive=False):
        rounds = pd.DataFrame([
            {
                "series_id": "s1", "game_id": map_name.lower(), "round_number": number,
                "team_id": "c9", "map": map_name, "side": "defender",
            }
            for map_name in ("Lotus", "Ascent")
            for number in range(1, 21)
        ])
        buys = pd.DataFrame([
            {
                "series_id": row.series_id, "game_id": row.game_id, "round_number": row.round_number,
                "team_id": "c9", "map": row.map, "side": row.side, "player_id": "p1",
                "player_name": "OXY", "agent": "jett", "has_operator": True,
            }
            for row in rounds.itertuples(index=False)
        ])
        kills = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "ascent", "map": "Ascent", "round_number": number,
                "killer_id": "p1", "killer_name": "OXY", "killer_team_id": "c9",
                "killer_side": "defender", "killer_agent": "jett", "victim_id": f"v{number}",
                "weapon": "operator", "seconds_into_round": 5 if early else 35 + number,
                "killer_x": 950 if aggressive else 50, "killer_y": 0,
            }
            for number in range(1, 11)
        ])
        spawn_positions = pd.DataFrame([
            {"map": map_name, "side": side, "x": x, "y": 0}
            for map_name in ("Lotus", "Ascent")
            for side, x in (("defender", 0), ("attacker", 1000))
        ])
        return rounds, buys, kills, spawn_positions

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
        self.assertIn("mostly during 0–20 s", card["tendency"])
        self.assertEqual(card["evidence"], "30/50 rounds (60%), 2 series.")
        self.assertEqual(card["confidence"], "High")
        self.assertIn(
            "Opponents can expect Trapwire at A site in 60% of your defense rounds (30/50)",
            card["tendency"],
        )
        self.assertEqual(card["response"], "Consider varying timing or position.")
        self.assertEqual(len(card["facts"]), 2)
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
        self.assertEqual(by_type["Operator"]["map"], "All maps")
        self.assertEqual(by_type["Side economy"]["confidence"], "High")
        self.assertIn("40% of defense rounds versus 4% on attack", by_type["Side economy"]["tendency"])

        sparse_buys = buys[buys["side"].eq("defender") | buys["round_number"].le(55)]
        sparse_cards = generate_counter_cards(
            rounds, pd.DataFrame(), sparse_buys, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            "c9", "Cloud9", "Lotus", "defender",
        )
        self.assertNotIn("Side economy", {card["type"] for card in sparse_cards})

        self_cards = generate_counter_cards(
            rounds, pd.DataFrame(), buys, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            "c9", "Cloud9", "Lotus", "defender", self_scout=True,
        )
        self_economy = next(card for card in self_cards if card["type"] == "Side economy")
        self.assertIn("your Operator", self_economy["tendency"])
        self.assertIn("Consider varying", self_economy["response"])
        self.assertNotIn("Expect the Operator more", self_economy["response"])
        self.assertNotIn("dry-peek", self_economy["response"])

    def test_side_buy_rate_gap_is_a_main_card_even_when_prevalence_is_low(self):
        rounds = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "g1", "round_number": number,
                "team_id": "team-a", "map": "Lotus", "side": side,
            }
            for side, start in (("defender", 1), ("attacker", 51))
            for number in range(start, start + 50)
        ])
        buys = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "g1", "round_number": row.round_number,
                "team_id": "team-a", "map": "Lotus", "side": row.side,
                "player_id": "p1", "player_name": "Player", "agent": "jett",
                "has_operator": (row.side == "defender" and row.round_number <= 15)
                or (row.side == "attacker" and row.round_number <= 55),
            }
            for row in rounds.itertuples(index=False)
        ])

        plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            "team-a", "Test Team", "Lotus", "defender",
        )
        card = next(card for card in plan["cards"] if card["type"] == "Side economy")

        self.assertIn("30% of defense rounds versus 10% on attack", card["tendency"])
        self.assertEqual(card["confidence"], "High")
        self.assertNotIn(card, plan["watch_list"])

    def test_operator_location_uses_selected_map_engagements_only(self):
        rounds, buys, kills, spawn_positions = self._operator_profile_tables()
        centres = {
            "ascent": {"A": {"x": 50.0, "y": 0.0}},
            "lotus": {"B": {"x": 50.0, "y": 0.0}},
        }

        lotus_plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, kills, pd.DataFrame(), spawn_positions,
            "c9", "G2 Esports", "Lotus", "defender", centres,
        )
        ascent_plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, kills, pd.DataFrame(), spawn_positions,
            "c9", "G2 Esports", "Ascent", "defender", centres,
        )
        lotus_response = next(card for card in lotus_plan["cards"] if card["type"] == "Operator")["response"]
        ascent_response = next(card for card in ascent_plan["cards"] if card["type"] == "Operator")["response"]

        self.assertIn("across the map pool", lotus_response)
        self.assertNotIn("Ascent A site", lotus_response)
        self.assertIn("Ascent A site", ascent_response)

    def test_operator_late_passive_profile_never_gets_first_twenty_advice(self):
        rounds, buys, kills, spawn_positions = self._operator_profile_tables()

        plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, kills, pd.DataFrame(), spawn_positions,
            "c9", "G2 Esports", "Lotus", "defender",
        )
        card = next(card for card in plan["cards"] if card["type"] == "Operator")

        self.assertEqual(card["map"], "All maps")
        self.assertIn("40/40", card["tendency"])
        self.assertIn("late", card["response"])
        self.assertIn("passive Op holding deep", card["response"])
        self.assertNotIn("first 20 s", card["response"])
        self.assertIn("No attack buy snapshots recorded (0 observed)", " ".join(card["facts"]))
        self.assertNotIn("attack 0/0 (0%)", " ".join(card["facts"]))

    def test_operator_early_aggressive_profile_gets_matching_advice(self):
        rounds, buys, kills, spawn_positions = self._operator_profile_tables(early=True, aggressive=True)

        plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, kills, pd.DataFrame(), spawn_positions,
            "c9", "G2 Esports", "Ascent", "defender",
        )
        card = next(card for card in plan["cards"] if card["type"] == "Operator")

        self.assertIn("early, aggressive Op", card["response"])
        self.assertIn("first 20 s", card["response"])

    def test_operator_cards_prefer_saved_full_data_anchors_over_team_proxy(self):
        rounds, buys, kills, spawn_positions = self._operator_profile_tables(early=True)

        plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, kills, pd.DataFrame(), spawn_positions,
            "c9", "G2 Esports", "Ascent", "defender",
            spawn_estimates={"ascent": {"defender": (1000.0, 0.0), "attacker": (0.0, 0.0)}},
        )
        card = next(card for card in plan["cards"] if card["type"] == "Operator")

        self.assertIn("early, aggressive Op", card["response"])
        self.assertIn("aggressive", " ".join(card["facts"]))

    def test_operator_mixed_profile_mentions_both_early_and_late_patterns(self):
        rounds, buys, kills, spawn_positions = self._operator_profile_tables()
        kills.loc[:4, "seconds_into_round"] = 5
        kills.loc[:4, "killer_x"] = 950
        kills.loc[5:, "killer_x"] = 50

        plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, kills, pd.DataFrame(), spawn_positions,
            "c9", "G2 Esports", "Lotus", "defender",
        )
        card = next(card for card in plan["cards"] if card["type"] == "Operator")

        self.assertIn("both early pressure and late deep holds", card["response"])

    def test_self_scout_operator_response_uses_our_point_of_view(self):
        rounds, buys, kills, spawn_positions = self._operator_profile_tables(early=True, aggressive=True)

        plan = generate_counter_plan(
            rounds, pd.DataFrame(), buys, kills, pd.DataFrame(), spawn_positions,
            "c9", "Cloud9", "Lotus", "defender", self_scout=True,
        )
        card = next(card for card in plan["cards"] if card["type"] == "Operator")

        self.assertIn("your Operator", card["tendency"])
        self.assertIn("Consider varying", card["response"])
        self.assertNotIn("dry-peek", card["response"])
        self.assertNotIn("Expect an", card["response"])

    def test_watch_list_includes_twenty_five_to_under_thirty_five_percent(self):
        rounds = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "g1", "round_number": number,
                "team_id": "c9", "map": "Lotus", "side": "defender",
            }
            for number in range(1, 51)
        ])
        casts = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "g1", "round_number": number,
                "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p1",
                "player_name": "OXY", "agent": "cypher", "ability": "trapwire",
                "seconds_into_round": 5, "x": cell * 1000 + 100, "y": 100,
            }
            for cell, count in ((0, 13), (1, 17), (2, 18))
            for number in range(1, count + 1)
        ])
        buys = pd.DataFrame([
            {
                "series_id": "s1", "game_id": "g1", "round_number": number,
                "team_id": "c9", "map": "Lotus", "side": "defender", "player_id": "p1",
                "player_name": "OXY", "agent": "jett", "has_operator": False,
            }
            for number in range(1, 51)
        ])

        plan = generate_counter_plan(
            rounds, casts, buys, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            "c9", "G2 Esports", "Lotus", "defender",
        )

        percentages = {round(card["frequency"]) for card in plan["watch_list"]}
        self.assertEqual(percentages, {26, 34})
        self.assertTrue(all(card["confidence"] == "Low" for card in plan["watch_list"]))
        self.assertTrue(all(card["frequency"] < 35 for card in plan["watch_list"]))

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