import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.abilities import CATEGORIES
from analysis.landing import build_walkable_mask
from analysis.text import team_possessive
from app import (
    _category_counts,
    _map_figure,
    _casts_for_position,
    _casts_for_arrow_zone,
    _most_common_category,
    _operator_headline,
    _operator_strip_figure,
    _navigate_counter_card,
    _padded_axis_range,
    _sentinel_summary,
    _separate_zone_markers,
    _set_ability_category,
)


class UtilityMapAppTests(unittest.TestCase):
    def test_team_possessive_handles_s_and_other_endings(self):
        self.assertEqual(team_possessive("G2 Esports"), "G2 Esports'")
        self.assertEqual(team_possessive("Cloud9"), "Cloud9's")
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
        self.assertEqual(figure.layout.xaxis.constrain, "domain")
        self.assertEqual(figure.layout.yaxis.constrain, "domain")
        self.assertEqual(figure.layout.yaxis.scaleanchor, "x")
        self.assertEqual(list(markers.text), ["1", "2"])
        self.assertNotIn("site_label", str(markers.customdata))
        self.assertGreaterEqual(math.dist((markers.x[0], markers.y[0]), (markers.x[1], markers.y[1])), 0.9)

    def test_zone_markers_are_offset_from_site_letters(self):
        shape = {"x_min": 0.0, "x_max": 1000.0, "y_min": 0.0, "y_max": 1000.0}
        zones = pd.DataFrame([{"x": 500.0, "y": 500.0}])
        sites = {"B": {"x": 500.0, "y": 500.0}}

        marker_x, marker_y = _separate_zone_markers(zones, shape, sites)

        self.assertGreater(math.dist((marker_x[0], marker_y[0]), (500.0, 500.0)), 100.0)
        self.assertGreaterEqual(marker_x[0], -50.0)
        self.assertLessEqual(marker_x[0], 1050.0)
        self.assertGreaterEqual(marker_y[0], -50.0)
        self.assertLessEqual(marker_y[0], 1050.0)

    def test_padded_axes_contain_all_walkable_cells_for_reference_map_extents(self):
        extents = {
            "Lotus": (-2000.0, 12000.0, -6000.0, 7000.0),
            "Ascent": (-12000.0, 12000.0, -14000.0, 6000.0),
            "Haven": (-14000.0, 14000.0, -14000.0, 5000.0),
        }
        for map_name, (x_min, x_max, y_min, y_max) in extents.items():
            with self.subTest(map_name=map_name):
                shape = {
                    "density": [[0.1] * 5 for _ in range(5)],
                    "x": np.linspace(x_min, x_max, 5), "y": np.linspace(y_min, y_max, 5),
                    "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
                }
                mask = build_walkable_mask(shape)
                x_grid, y_grid = np.meshgrid(shape["x"], shape["y"])
                x_range = _padded_axis_range(shape, "x")
                y_range = _padded_axis_range(shape, "y")

                self.assertTrue(np.all((x_grid[mask] >= x_range[0]) & (x_grid[mask] <= x_range[1])))
                self.assertTrue(np.all((y_grid[mask] >= y_range[0]) & (y_grid[mask] <= y_range[1])))

    def test_filtered_heat_is_clipped_to_the_walkable_mask(self):
        density = np.zeros((5, 5), dtype=float)
        density[2, 2] = 1.0
        shape = {
            "density": density, "x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float),
            "x_min": 0.0, "x_max": 4.0, "y_min": 0.0, "y_max": 4.0, "bins": 5,
        }
        casts = pd.DataFrame([{"x": 4.0, "y": 4.0}])

        figure = _map_figure(shape, casts, pd.DataFrame(), False, "Lotus", {})
        walkable = build_walkable_mask(shape)
        heat = np.asarray(figure.data[1].z, dtype=float)

        self.assertTrue(np.isnan(heat[~walkable]).all())

    def test_estimated_landing_positions_drive_zones_and_draw_caster_arrows(self):
        shape = {
            "density": [[0.0, 1.0], [1.0, 0.0]], "x": [0.0, 10.0], "y": [0.0, 10.0],
            "x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0, "bins": 2,
        }
        casts = pd.DataFrame([{
            "x": 1.0, "y": 1.0, "landing_x": 8.0, "landing_y": 7.0,
            "caster_x": 1.0, "caster_y": 1.0, "estimated": True,
        }, {
            "x": 405.0, "y": 1.0, "landing_x": 405.0, "landing_y": 2.0,
            "caster_x": 400.0, "caster_y": 1.0, "estimated": True,
        }])
        zones = pd.DataFrame([{
            "cell_x": 0, "cell_y": 0, "x": 200.0, "y": 200.0,
            "site_label": "A site", "ability_name": "Ability", "rounds_used": 1,
            "total_rounds": 1, "percentage": 100.0,
        }])

        landing_casts = _casts_for_position(casts, "Estimated landing")
        figure = _map_figure(
            shape, landing_casts, zones, True, "Lotus", {}, "Estimated landing", arrow_zone=1,
        )

        self.assertEqual((landing_casts.loc[0, "x"], landing_casts.loc[0, "y"]), (8.0, 7.0))
        self.assertEqual((casts.loc[0, "x"], casts.loc[0, "y"]), (1.0, 1.0))
        self.assertEqual(len(figure.layout.annotations), 1)
        self.assertEqual((figure.layout.annotations[0].ax, figure.layout.annotations[0].ay), (1.0, 1.0))
        self.assertEqual((figure.layout.annotations[0].x, figure.layout.annotations[0].y), (8.0, 7.0))

    def test_arrow_zone_selection_and_fallback_sample_are_bounded(self):
        casts = pd.DataFrame([
            {
                "x": float(index * 400 + 1), "y": 1.0, "caster_x": float(index * 400),
                "caster_y": 1.0, "estimated": True,
            }
            for index in range(100)
        ])
        zones = pd.DataFrame([{"cell_x": 3, "cell_y": 0}])

        selected = _casts_for_arrow_zone(casts, zones, selected_zone=1)
        sample = _casts_for_arrow_zone(casts, zones, selected_zone=None)

        self.assertEqual(len(selected), 1)
        self.assertEqual(math.floor(selected.iloc[0]["x"] / 400), 3)
        self.assertEqual(len(sample), 80)

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

    def test_operator_strip_labels_games_and_leaves_a_gap(self):
        rounds = pd.DataFrame([
            {"series_id": "s1", "game_id": "g1", "game_number": "1", "opponent_name": "NRG",
             "map": "Lotus", "round_start_time": "2026-03-10T12:00:00Z", "round_number": number,
             "holders": ("p1",), "holder_names": ("OXY",), "snapshot_available": True}
            for number in (1, 2)
        ] + [{
            "series_id": "s1", "game_id": "g2", "game_number": "2", "opponent_name": "FURIA",
            "map": "Ascent", "round_start_time": "2026-04-12T12:00:00Z",
            "round_number": 1, "holders": ("p1",), "holder_names": ("OXY",), "snapshot_available": True,
        }])
        players = pd.DataFrame([{"player_id": "p1", "player_name": "OXY"}])

        figure = _operator_strip_figure(rounds, players)

        self.assertEqual(list(figure.layout.yaxis.ticktext), [
            "2026-04-12 · vs FURIA · Ascent · G2",
            "2026-03-10 · vs NRG · Lotus · G1",
        ])
        marker = figure.data[0]
        self.assertEqual(len(set(marker.y)), 2)
        self.assertGreater(abs(marker.y[-1] - marker.y[1]), 1)

    def test_operator_strip_limits_to_latest_fifteen_unless_expanded(self):
        rounds = pd.DataFrame([
            {
                "series_id": f"s{index}", "game_id": f"g{index}", "game_number": "1",
                "opponent_name": f"Team {index}", "map": "Lotus",
                "round_start_time": f"2026-04-{index + 1:02d}T12:00:00Z", "round_number": 1,
                "holders": ("p1",), "holder_names": ("OXY",), "snapshot_available": True,
            }
            for index in range(17)
        ])
        players = pd.DataFrame([{"player_id": "p1", "player_name": "OXY"}])

        latest = _operator_strip_figure(rounds, players)
        all_games = _operator_strip_figure(rounds, players, show_all=True)

        self.assertEqual(len(latest.layout.yaxis.ticktext), 15)
        self.assertTrue(str(latest.layout.yaxis.ticktext[0]).startswith("2026-04-17"))
        self.assertEqual(len(latest.data[0].x), 15)
        self.assertEqual(len(all_games.layout.yaxis.ticktext), 17)

    def test_operator_headline_uses_player_share_and_side_buy_evidence(self):
        small = pd.DataFrame({"seconds_into_round": [5, 25, 45, 50], "site_label": ["B site"] * 4,
                              "aggressiveness": ["aggressive"] * 4})
        headline = _operator_headline(
            "G2 Esports", {"player_name": "OXY", "rounds": 93, "share": 96.875},
            289, small, "Lotus", "defense", operator_rounds=96,
            side_buy_rates={
                "defender": {"held_rounds": 20, "observed_rounds": 100, "rate": 20},
                "attacker": {"held_rounds": 1, "observed_rounds": 100, "rate": 1},
            },
        )

        self.assertIn("G2 Esports'", headline)
        self.assertIn("OXY holds G2 Esports' Operator in 97%", headline)
        self.assertIn("(93/96)", headline)
        self.assertIn("20% (20/100) of defense rounds", headline)
        self.assertIn("1% (1/100) on attack", headline)
        self.assertIn("Too few player engagements to profile angles", headline)
        self.assertIn("try All maps", headline)
        self.assertNotIn("0–20 s", headline)

        enough = pd.DataFrame({
            "seconds_into_round": list(range(10)),
            "site_label": ["B site"] * 10,
            "aggressiveness": ["aggressive"] * 10,
        })
        profiled = _operator_headline(
            "Cloud9", {"player_name": "OXY", "rounds": 20, "share": 100},
            100, enough, "Lotus", "defense", operator_rounds=20,
        )
        self.assertIn("Cloud9's", profiled)
        self.assertIn("0–20 s", profiled)
        self.assertIn("100% of OXY's defense Operator engagements", profiled)

    def test_counter_card_navigation_preserves_matching_filters(self):
        state = {}
        with patch("app.st.session_state", state):
            _navigate_counter_card("Utility Map", "Lotus", "defense", "team-a", "placeable", "cypher")

        self.assertEqual(state["scout_tabs"], "Utility Map")
        self.assertEqual(state["utility_map_widget_team-a"], "Lotus")
        self.assertEqual(state["utility_map_selection_team-a"], "Lotus")
        self.assertEqual(state["shared_side"], "defense")
        self.assertEqual(state["ability_category"], "placeable")
        self.assertEqual(state["utility_agent_widget_team-a"], "cypher")
        self.assertEqual(state["utility_position_team-a_placeable"], "Caster")


if __name__ == "__main__":
    unittest.main()