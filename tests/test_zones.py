import unittest

import pandas as pd

from analysis.sites import disambiguate_zone_labels, label_zones
from analysis.zones import aggregate_zone_cells, calculate_zone_percentage, filter_casts, merge_zone_cells


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

    def test_remote_casts_are_excluded_by_default(self):
        casts = pd.concat([
            self.casts.iloc[:1],
            pd.DataFrame([self._cast(
                "s1", "g1", 1, 200, 300, "Player One", ability="stars",
            ) | {"agent": "astra"}]),
        ], ignore_index=True)
        selected = filter_casts(casts, category="all", seconds_range=(-30, 30))
        self.assertEqual(selected["ability"].tolist(), ["trapwire"])
        included = filter_casts(
            casts, category="all", seconds_range=(-30, 30), include_remote=True,
        )
        self.assertEqual(included["ability"].tolist(), ["trapwire", "stars"])

    def test_buy_phase_can_be_included_independently_of_live_window(self):
        casts = self.casts.iloc[:3].copy()
        casts["seconds_into_round"] = [-20.0, 10.0, 120.0]
        selected = filter_casts(casts, seconds_range=(0, 30), include_buy_phase=True)
        self.assertEqual(selected["seconds_into_round"].tolist(), [-20.0, 10.0])
        selected = filter_casts(casts, seconds_range=(0, 30), include_buy_phase=False)
        self.assertEqual(selected["seconds_into_round"].tolist(), [10.0])
        self.assertTrue(filter_casts(casts.iloc[:0]).empty)

    def test_representative_cast_is_a_real_ability_agent_player_combination(self):
        casts = pd.DataFrame([
            self._cast("s1", "g1", 1, 100, 200, "Player One"),
            self._cast("s1", "g1", 2, 100, 200, "Player One", "spycam"),
            self._cast("s1", "g2", 1, 100, 200, "Player Two", "spycam"),
            self._cast("s1", "g2", 1, 100, 200, "Player Two", "spycam"),
        ])
        zone = calculate_zone_percentage(self.rounds, casts, "team-a", "Lotus", "defender").iloc[0]
        self.assertEqual((zone["ability"], zone["agent"], zone["player_name"]), ("spycam", "cypher", "Player Two"))
        self.assertEqual(zone["rounds_used"], 2)

    def test_adjacent_cells_merge_transitively_and_union_distinct_rounds(self):
        casts = pd.DataFrame([
            self._cast("s1", "g1", 1, 100, 100, "Player One"),
            self._cast("s1", "g1", 1, 500, 100, "Player One"),
            self._cast("s1", "g1", 2, 500, 100, "Player One"),
            self._cast("s1", "g1", 3, 900, 100, "Player Two"),
            self._cast("s2", "g3", 1, 2100, 100, "Player Two"),
            self._cast("s1", "g1", 2, 500, 100, "Player Two", ability="spycam"),
        ])
        centres = {"lotus": {"A": {"x": 1000.0, "y": 100.0}}}

        denominator = self.rounds[
            self.rounds["team_id"].eq("team-a") & self.rounds["map"].eq("Lotus")
        ]
        cells = label_zones(aggregate_zone_cells(casts, denominator), "Lotus", centres)
        zones = merge_zone_cells(cells)
        zones = disambiguate_zone_labels(zones, "Lotus", centres)
        trapwire_zones = zones[zones["ability"].eq("trapwire")]
        merged = trapwire_zones.iloc[0]

        self.assertEqual(int(merged["rounds_used"]), 3)
        self.assertEqual(int(merged["total_rounds"]), 4)
        self.assertEqual(float(merged["percentage"]), 75.0)
        self.assertEqual(len(merged["cells"]), 3)
        self.assertEqual(len(trapwire_zones), 2)
        self.assertNotEqual(trapwire_zones.iloc[0]["site_label"], trapwire_zones.iloc[1]["site_label"])

    def test_cells_within_six_metres_merge_by_edge_distance(self):
        casts = pd.DataFrame([
            self._cast("s1", "g1", 1, 100, 100, "Player One"),
            self._cast("s1", "g1", 2, 900, 900, "Player One"),
            self._cast("s1", "g2", 1, 2100, 100, "Player One"),
        ])
        cells = aggregate_zone_cells(casts, self.rounds)
        cells["site_label"] = "mid"

        zones = merge_zone_cells(cells)

        self.assertEqual(len(zones), 2)
        self.assertEqual(zones.iloc[0]["rounds_used"], 2)
        self.assertEqual(set(zones.iloc[0]["cells"]), {(0, 0), (2, 2)})


if __name__ == "__main__":
    unittest.main()