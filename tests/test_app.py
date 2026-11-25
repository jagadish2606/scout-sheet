import math
import unittest
from unittest.mock import patch

import pandas as pd

from analysis.abilities import CATEGORIES
from app import (
    _category_counts,
    _map_figure,
    _most_common_category,
    _padded_axis_range,
    _sentinel_summary,
    _separate_zone_markers,
    _set_ability_category,
)


class UtilityMapAppTests(unittest.TestCase):
    def test_map_axis_ranges_use_five_percent_padding(self):
        shape = {"x_min": -100.0, "x_max": 300.0, "y_min": -50.0, "y_max": 150.0}

        self.assertEqual(_padded_axis_range(shape, "x"), [-120.0, 320.0])
        self.assertEqual(_padded_axis_range(shape, "y"), [-60.0, 160.0])

    def test_figure_has_explicit_extent_and_numeric_zone_labels(self):
        shape = {
            "density": [[0.0, 1.0], [1.0, 0.0]], "x": [0.0, 10.0], "y": [0.0, 10.0],
            "x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0, "bins": 2,
        }
        casts = pd.DataFrame(columns=["x", "y"])
        zones = pd.DataFrame([
            {"x": 1.0, "y": 1.0, "site_label": "A site", "ability_name": "Trapwire",
             "rounds_used": 5, "total_rounds": 10, "percentage": 50.0},
            {"x": 1.1, "y": 1.1, "site_label": "B site", "ability_name": "Spycam",
             "rounds_used": 4, "total_rounds": 10, "percentage": 40.0},
        ])

        figure = _map_figure(shape, casts, zones, False, "Lotus", {})
        markers = next(trace for trace in figure.data if trace.name == "Top zones")

        self.assertEqual(figure.layout.height, 650)
        self.assertEqual(list(figure.layout.xaxis.range), [-0.5, 10.5])
        self.assertEqual(list(figure.layout.yaxis.range), [-0.5, 10.5])
        self.assertEqual(list(markers.text), ["1", "2"])
        self.assertNotIn("site_label", str(markers.customdata))
        self.assertGreaterEqual(math.dist((markers.x[0], markers.y[0]), (markers.x[1], markers.y[1])), 0.9)

    def test_category_counts_and_switch_target(self):
        casts = pd.DataFrame({"category": ["smoke", "smoke", "recon", "placeable"]})
        counts = _category_counts(casts)

        self.assertEqual(counts["smoke"], 2)
        self.assertEqual(counts["placeable"], 1)
        self.assertTrue(all(category in counts for category in CATEGORIES))
        self.assertEqual(_most_common_category(counts, "placeable"), "smoke")
        self.assertIsNone(_most_common_category(_category_counts(casts.iloc[:0]), "placeable"))
        state = {}
        with patch("app.st.session_state", state):
            _set_ability_category("smoke")
        self.assertEqual(state["ability_category"], "smoke")

    def test_sentinel_summary_names_agents_and_distinguishes_missing_data(self):
        buys = pd.DataFrame([
            {"team_id": "team-a", "map": "Lotus", "agent": "killjoy"},
            {"team_id": "team-a", "map": "Lotus", "agent": "sage"},
            {"team_id": "team-b", "map": "Lotus", "agent": "cypher"},
        ])
        no_sentinel = pd.DataFrame([{"team_id": "team-a", "map": "Lotus", "agent": "jett"}])

        self.assertEqual(
            _sentinel_summary(buys, "team-a", "lotus"),
            "Sentinel agents on this map: Killjoy, Sage.",
        )
        self.assertEqual(_sentinel_summary(no_sentinel, "team-a", "Lotus"), "No sentinel played on this map.")
        self.assertEqual(_sentinel_summary(buys.iloc[:0], "team-a", "Lotus"), "Sentinel lineup data is unavailable for this map.")


if __name__ == "__main__":
    unittest.main()