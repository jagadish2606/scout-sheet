import unittest

import pandas as pd

from analysis.zones import calculate_zone_percentage


class ZonePercentageTests(unittest.TestCase):
    def setUp(self):
        self.rounds = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "round_number": 1, "team_id": "team-a", "team_name": "Alpha", "map": "Lotus", "side": "defender"},
            {"series_id": "s1", "game_id": "g1", "round_number": 2, "team_id": "team-a", "team_name": "Alpha", "map": "Lotus", "side": "defender"},
            {"series_id": "s1", "game_id": "g2", "round_number": 1, "team_id": "team-a", "team_name": "Alpha", "map": "Lotus", "side": "defender"},
            {"series_id": "s2", "game_id": "g3", "round_number": 1, "team_id": "team-a", "team_name": "Alpha", "map": "Lotus", "side": "defender"},
            {"series_id": "s1", "game_id": "g1", "round_number": 1, "team_id": "team-b", "team_name": "Bravo", "map": "Lotus", "side": "attacker"},
            {"series_id": "s2", "game_id": "g4", "round_number": 1, "team_id": "team-a", "team_name": "Alpha", "map": "Ascent", "side": "defender"},
        ])
        self.casts = pd.DataFrame([
            self._cast("s1", "g1", 1, 100, 200, "Player One"),
            self._cast("s1", "g1", 1, 110, 210, "Player One"),
            self._cast("s1", "g1", 2, 150, 250, "Player Two"),
            self._cast("s1", "g2", 1, 170, 270, "Player One"),
            self._cast("s2", "g3", 1, 900, 900, "Player Two", ability="smoke"),
            self._cast("s1", "g1", 99, 100, 200, "Player One"),
            self._cast("s1", "g1", 1, None, None, "Player One"),
        ])

    @staticmethod
    def _cast(series_id, game_id, round_number, x, y, player_name, ability="trapwire"):
        return {
            "series_id": series_id, "game_id": game_id, "round_number": round_number,
            "team_id": "team-a", "team_name": "Alpha", "map": "Lotus", "side": "defender",
            "player_id": player_name.lower().replace(" ", "-"), "player_name": player_name,
            "agent": "cypher", "ability": ability, "seconds_into_round": 10.0,
            "x": x, "y": y,
        }

    def test_percentage_uses_distinct_rounds_and_game_scoped_round_keys(self):
        zones = calculate_zone_percentage(self.rounds, self.casts, "team-a", "Lotus", "defender")
        first = zones.iloc[0]
        self.assertEqual((first["cell_x"], first["cell_y"]), (0, 0))
        self.assertEqual(first["rounds_used"], 3)
        self.assertEqual(first["total_rounds"], 4)
        self.assertEqual(first["percentage"], 75.0)
        self.assertEqual(first["series_count"], 1)

    def test_cast_filters_do_not_change_round_denominator_and_unmatched_round_is_ignored(self):
        zones = calculate_zone_percentage(
            self.rounds, self.casts, "team-a", "Lotus", "defender",
            category="placeable", player_id="player-one", seconds_range=(0, 30),
        )
        self.assertEqual(zones.iloc[0]["rounds_used"], 2)
        self.assertEqual(zones.iloc[0]["total_rounds"], 4)
        self.assertEqual(zones.iloc[0]["percentage"], 50.0)

    def test_no_matching_rounds_returns_no_zone(self):
        zones = calculate_zone_percentage(self.rounds, self.casts, "missing", "Lotus", "defender")
        self.assertTrue(zones.empty)


if __name__ == "__main__":
    unittest.main()