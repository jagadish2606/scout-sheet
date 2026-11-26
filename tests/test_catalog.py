import unittest

import pandas as pd

from analysis.catalog import circuit_for_event, team_circuit_labels


class CircuitCatalogTests(unittest.TestCase):
    def test_event_names_map_to_vct_masters_and_regional_game_changers(self):
        self.assertEqual(circuit_for_event("VCT Americas - Kickoff 2025"), "VCT Americas")
        self.assertEqual(circuit_for_event("Masters Bangkok 2025"), "Masters")
        self.assertEqual(
            circuit_for_event("VCT Game Changers North America Stage 1"),
            "Game Changers – North America",
        )
        self.assertEqual(
            circuit_for_event("Game Changers BR - Split 1 2026"), "Game Changers – Brazil",
        )
        self.assertEqual(circuit_for_event("Game Changers LATAM - Split 2 2026"), "Game Changers – Latin America")
        self.assertEqual(circuit_for_event("Game Changers JPN - Split 1 2026"), "Game Changers – Japan")
        self.assertEqual(circuit_for_event("Game Changers KR - Split 1 2026"), "Game Changers – Korea")
        self.assertEqual(circuit_for_event("Game Changers NA - Split 1 2026"), "Game Changers – North America")
        self.assertEqual(circuit_for_event("Game Changers Pacific - 2025"), "Game Changers – Southeast Asia/Oceania")
        self.assertEqual(circuit_for_event("Game Changers Championship - 2025"), "Game Changers – Championship")

    def test_team_choices_are_grouped_by_circuit_with_distinct_series_counts(self):
        rounds = pd.DataFrame([
            {"team_id": "c9", "team_name": "Cloud9", "series_id": "1"},
            {"team_id": "c9", "team_name": "Cloud9", "series_id": "1"},
            {"team_id": "c9", "team_name": "Cloud9", "series_id": "2"},
            {"team_id": "c9", "team_name": "Cloud9", "series_id": "3"},
            {"team_id": "op", "team_name": "Opponent", "series_id": "4"},
        ])
        event_map = {
            "1": "VCT Americas Kickoff", "2": "VCT Americas Stage 1",
            "3": "Masters Toronto", "4": "Game Changers EMEA Stage 1",
        }

        labels = team_circuit_labels(rounds, event_map)

        cloud9 = labels[labels["team_id"].eq("c9")]
        self.assertEqual(cloud9["circuit"].tolist(), ["VCT Americas", "Masters"])
        self.assertEqual(cloud9["series_count"].tolist(), [2, 1])
        self.assertEqual(cloud9.iloc[0]["label"], "Cloud9 · 2 series")
        self.assertNotIn("c9", cloud9.iloc[0]["label"])
        self.assertEqual(labels.iloc[-1]["circuit"], "Game Changers – EMEA")


if __name__ == "__main__":
    unittest.main()