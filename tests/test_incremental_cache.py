import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.extract import TABLE_COLUMNS
from analysis.operator import load_spawn_estimates
from analysis.incremental_cache import (
    bootstrap_existing_cache,
    combine_series_shards,
    read_series_table,
    series_is_complete,
    write_series_shard,
)


def _tables(series_id):
    tables = {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}
    tables["rounds"] = pd.DataFrame([{
        "series_id": series_id, "game_id": "game-1", "game_number": 1,
        "map": "Lotus", "round_number": 1, "round_start_time": "2025-01-01T00:00:00Z",
        "round_live_time": "2025-01-01T00:00:02Z", "team_id": 42,
        "team_name": "Test Team", "opponent_name": "Opponent", "side": "defender",
        "won_round": True,
    }])
    return tables


class IncrementalCacheTests(unittest.TestCase):
    def test_series_shards_publish_completion_then_stream_merge_with_stable_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shard_root, cache_dir = root / "series_shards", root / "cache"
            write_series_shard(shard_root, "series-1", _tables("series-1"))

            self.assertTrue(series_is_complete(shard_root, "series-1"))
            stored = read_series_table(shard_root, "series-1", "rounds")
            self.assertEqual(stored.loc[0, "team_id"], "42")
            counts, _ = combine_series_shards(["series-1"], shard_root, cache_dir)

            self.assertEqual(counts["rounds"], 1)
            self.assertEqual(counts["casts"], 0)
            merged = pd.read_parquet(cache_dir / "rounds.parquet")
            self.assertEqual(merged.loc[0, "team_name"], "Test Team")
            self.assertEqual(merged.loc[0, "round_number"], 1)

    def test_bootstrap_existing_cache_marks_series_complete_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_dir, shard_root = root / "cache", root / "cache" / "series_shards"
            cache_dir.mkdir()
            _tables("series-2")["rounds"].to_parquet(cache_dir / "rounds.parquet", index=False)

            first = bootstrap_existing_cache(cache_dir, shard_root)
            second = bootstrap_existing_cache(cache_dir, shard_root)

            self.assertEqual(first, {"series-2"})
            self.assertEqual(second, {"series-2"})
            self.assertTrue(series_is_complete(shard_root, "series-2"))
            self.assertEqual(len(read_series_table(shard_root, "series-2", "rounds")), 1)
            self.assertEqual(len(read_series_table(shard_root, "series-2", "casts")), 0)

    def test_cache_builder_persists_global_spawn_estimates(self):
        from build_cache import _save_spawn_estimates

        positions = pd.DataFrame([
            {"map": "Lotus", "side": "defender", "x": 1.0, "y": 2.0},
            {"map": "Lotus", "side": "defender", "x": 3.0, "y": 4.0},
            {"map": "Lotus", "side": "attacker", "x": 9.0, "y": 8.0},
        ])
        with tempfile.TemporaryDirectory() as temporary_directory:
            anchors = _save_spawn_estimates(positions, temporary_directory)
            path = Path(temporary_directory) / "spawn_estimates.json"

            self.assertEqual(anchors["lotus"]["defender"], (2.0, 3.0))
            self.assertEqual(
                load_spawn_estimates(path),
                {"lotus": {"defender": (2.0, 3.0), "attacker": (9.0, 8.0)}},
            )


if __name__ == "__main__":
    unittest.main()