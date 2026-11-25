import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.extract import extract_series
from build_cache import select_series_ids


def _game(game_id, round_number, start_time, team1_side, map_name):
    teams = [
        {
            "id": "team-a", "name": "Alpha", "side": team1_side,
            "players": [{
                "id": f"{game_id}-p1", "name": "Player One", "character": {"id": "jett", "name": "jett"},
                "inventory": {"items": [{"id": "melee"}, {"id": "operator"}]},
            }],
        },
        {
            "id": "team-b", "name": "Bravo", "side": "attacker" if team1_side == "defender" else "defender",
            "players": [{
                "id": f"{game_id}-p2", "name": "Player Two", "character": {"id": "sova", "name": "sova"},
                "inventory": {"items": [{"id": "melee"}, {"id": "vandal"}]},
            }],
        },
    ]
    return {
        "id": game_id,
        "map": {"name": map_name},
        "teams": teams,
        "segments": [{"id": f"round-{round_number}", "type": "round", "startedAt": start_time}],
    }


def _envelope(event, occurred_at="2025-01-01T00:00:00Z"):
    return {"occurredAt": occurred_at, "events": [event]}


def _round_started(game):
    return {
        "id": f"start-{game['id']}",
        "type": "game-started-round",
        "actor": {"id": game["id"], "type": "ESPORTS"},
        "target": {"id": game["segments"][-1]["id"], "type": "round"},
        "seriesState": {"games": [game]},
        "includesFullState": True,
    }


def _meta_game(series_id, game_id, session="1", valid="True", map_name="ascent"):
    return {
        "title": "val", "series_id": series_id, "game_id": game_id, "map": map_name,
        "session": session, "session_valid": valid,
    }


class ExtractSeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "meta").mkdir()
        (self.root / "val-ability-supplements").mkdir()
        (self.root / "val").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def _write_games(self, rows):
        path = self.root / "meta" / "games.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["title", "series_id", "game_id", "map", "session", "session_valid"])
            writer.writeheader()
            writer.writerows(rows)

    def test_extracts_round_scoped_tables_and_filters_rollback_and_invalid_session(self):
        self._write_games([
            _meta_game("42", "game-1", map_name="ascent"),
            _meta_game("42", "game-2", map_name="lotus"),
            _meta_game("42", "rolled-game", map_name="bind"),
            _meta_game("42", "invalid-game", session="2", valid="False", map_name="sunset"),
        ])
        start = "2025-01-01T00:00:00Z"
        game1 = _game("game-1", 1, start, "defender", "ascent")
        game2 = _game("game-2", 1, "2025-01-01T00:10:00Z", "attacker", "lotus")
        rolled_game = _game("rolled-game", 1, "2025-01-01T00:20:00Z", "defender", "bind")
        invalid_game = _game("invalid-game", 1, "2025-01-01T00:30:00Z", "defender", "sunset")
        ability_event = {
            "id": "cast-gap", "type": "player-used-ability",
            "actor": {
                "id": "game-1-p1", "state": {
                    "id": "game-1-p1", "name": "Player One", "teamId": "team-a", "side": "defender",
                    "game": {"position": {"x": 1, "y": 2}},
                },
            },
            "target": {"id": "dash", "state": {"id": "dash", "name": "dash"}},
            "seriesState": {"games": [{"id": "game-1"}]},
        }
        missing_supplement_cast = {
            **ability_event,
            "id": "cast-no-supplement",
            "target": {"id": "smoke", "state": {"id": "smoke", "name": "smoke"}},
        }
        kill_event = {
            "id": "kill-1", "type": "player-killed-player",
            "actor": {
                "id": "game-1-p1", "state": {
                    "id": "game-1-p1", "name": "Player One", "teamId": "team-a", "side": "defender",
                    "round": {"id": "game-1-p1", "segmentType": "round"},
                    "game": {"position": {"x": 100, "y": 200}, "weaponKills": {"operator": 1}},
                },
                "stateDelta": {"game": {"weaponKills": {"operator": 1}}},
            },
            "target": {
                "id": "game-1-p2", "state": {
                    "id": "game-1-p2", "name": "Player Two", "teamId": "team-b", "side": "attacker",
                    "game": {"position": {"x": 300, "y": 400}},
                },
            },
            "seriesState": {"games": [{"id": "game-1", "segments": [{"id": "round-1", "type": "round"}]}]},
        }
        won_event = {
            "type": "team-won-round", "actor": {"id": "team-b", "type": "team"},
            "target": {"id": "round-1", "type": "round"},
            "seriesState": {"games": [{"id": "game-1", "segments": [{"id": "round-1", "type": "round"}]}]},
        }
        freeze_event = {
            "type": "round-ended-freezetime", "includesFullState": True,
            "seriesState": {"games": [game1]},
        }
        envelopes = [
            _envelope({"type": "grid-started-feed"}),
            _envelope(_round_started(game1), start),
            _envelope(ability_event, "2025-01-01T00:00:10Z"),
            _envelope(missing_supplement_cast, "2025-01-01T00:00:11Z"),
            _envelope(kill_event, "2025-01-01T00:00:12Z"),
            _envelope(won_event, "2025-01-01T00:00:15Z"),
            _envelope(freeze_event, "2025-01-01T00:00:20Z"),
            _envelope(_round_started(game2), "2025-01-01T00:10:00Z"),
            _envelope(_round_started(rolled_game), "2025-01-01T00:20:00Z"),
            _envelope({"type": "series-!started-game", "target": {"id": "rolled-game", "type": "game"}}),
            _envelope({"type": "grid-started-feed"}),
            _envelope(_round_started(invalid_game), "2025-01-01T00:30:00Z"),
        ]
        supplements = [{
            "grid_event_id": "cast-gap", "in_grid": True, "ability": "dash", "slot": "ABILITY_1",
            "quality": "gap_exceeded",
            "frame_before": {"offset_ms": -100, "position": {"x": 10, "y": 20, "z": 30}, "view": {"x": 1, "y": 0, "z": 0}},
            "frame_after": {"offset_ms": 900, "position": {"x": 90, "y": 80, "z": 70}, "view": {"x": 0, "y": 1, "z": 0}},
            "interpolated": {"position": {"x": 999, "y": 999, "z": 999}, "view": {"x": 0, "y": 0, "z": 1}},
        }]
        supplement_path = self.root / "val-ability-supplements" / "42.jsonl.zst"
        supplement_path.touch()

        def fake_read(path):
            return iter(supplements if "val-ability-supplements" in str(path) else envelopes)

        with patch("analysis.extract.read_jsonl_zst", side_effect=fake_read):
            tables = extract_series("42", self.root)

        rounds = tables["rounds"]
        self.assertEqual(set(rounds["game_id"]), {"game-1", "game-2"})
        self.assertEqual(rounds[rounds["game_id"] == "game-1"]["round_number"].tolist(), [1, 1])
        self.assertEqual(rounds[rounds["game_id"] == "game-1"]["side"].tolist(), ["defender", "attacker"])
        self.assertEqual(rounds[rounds["game_id"] == "game-1"]["won_round"].tolist(), [False, True])

        casts = tables["casts"]
        self.assertEqual(len(casts), 2)
        self.assertEqual(casts.iloc[0]["x"], 10)
        self.assertEqual(casts.iloc[0]["quality"], "gap_exceeded")
        self.assertTrue(casts.iloc[1]["x"] != casts.iloc[1]["x"])  # nullable supplement data

        kill = tables["kills"].iloc[0]
        self.assertEqual(kill["weapon"], "operator")
        self.assertEqual((kill["killer_x"], kill["killer_y"], kill["victim_x"], kill["victim_y"]), (100, 200, 300, 400))
        buy = tables["buys"].iloc[0]
        self.assertEqual(buy["primary_weapon"], "operator")
        self.assertTrue(buy["has_operator"])


class SeriesSelectionTests(unittest.TestCase):
    def test_selects_direct_opponents_only_within_the_same_event(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "val").mkdir()
            (root / "meta").mkdir()
            with (root / "val" / "series.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["series_id", "event", "teams"])
                writer.writeheader()
                writer.writerows([
                    {"series_id": "10", "event": "Event A", "teams": "Cloud9 vs Alpha"},
                    {"series_id": "11", "event": "Event A", "teams": "Alpha vs Beta"},
                    {"series_id": "12", "event": "Event B", "teams": "Alpha vs Gamma"},
                ])
            fields = ["title", "series_id", "team1_id", "team1_name", "team2_id", "team2_name"]
            with (root / "meta" / "games.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {"title": "val", "series_id": "10", "team1_id": "c9", "team1_name": "Cloud9", "team2_id": "a", "team2_name": "Alpha"},
                    {"title": "val", "series_id": "11", "team1_id": "a", "team1_name": "Alpha", "team2_id": "b", "team2_name": "Beta"},
                    {"title": "val", "series_id": "12", "team1_id": "a", "team1_name": "Alpha", "team2_id": "g", "team2_name": "Gamma"},
                ])
            self.assertEqual(select_series_ids(root), ["10", "11"])


if __name__ == "__main__":
    unittest.main()