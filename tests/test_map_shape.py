import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.map_shape import load_cached_map_shapes, load_map_shapes


class MapShapeTests(unittest.TestCase):
    def test_shape_uses_cast_and_kill_positions_and_is_cached(self):
        casts = pd.DataFrame([{"map": "Lotus", "x": 0, "y": 10}, {"map": "Lotus", "x": 10, "y": 0}])
        kills = pd.DataFrame([{
            "map": "Lotus", "killer_x": 5, "killer_y": 5, "victim_x": -5, "victim_y": 15,
        }])
        with tempfile.TemporaryDirectory() as directory:
            first = load_map_shapes(casts, kills, directory, bins=8)
            second = load_map_shapes(casts, kills, directory, bins=8)
            self.assertEqual(first["Lotus"]["position_count"], 4)
            self.assertEqual(first["Lotus"]["density"].shape, (8, 8))
            self.assertGreater(np.count_nonzero(first["Lotus"]["density"]), 4)
            self.assertAlmostEqual(float(first["Lotus"]["density"].max()), 1.0)
            np.testing.assert_array_equal(first["Lotus"]["density"], second["Lotus"]["density"])
            self.assertTrue((Path(directory) / "map_shapes.npz").is_file())
            cached = load_cached_map_shapes(directory)
            np.testing.assert_array_equal(first["Lotus"]["density"], cached["Lotus"]["density"])


if __name__ == "__main__":
    unittest.main()