import unittest

import numpy as np
import pandas as pd

from analysis.landing import (
    add_landing_estimates,
    build_walkable_mask,
    estimate_cast_landing,
    ray_march_landing,
)


def _shape(density):
    return {
        "density": np.asarray(density, dtype=float),
        "x": np.arange(len(density[0]), dtype=float),
        "y": np.arange(len(density), dtype=float),
    }


class LandingTests(unittest.TestCase):
    def test_ray_march_stops_at_first_wall(self):
        shape = _shape([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
        mask = build_walkable_mask(shape, threshold=0.5, dilation_cells=0)

        x, y, method = ray_march_landing(
            1, 0, 1, 0, shape, max_range=8, step=0.5, mask=mask,
        )

        self.assertEqual(method, "wall_hit")
        self.assertLess(x, 5)
        self.assertGreaterEqual(x, 4)
        self.assertEqual(y, 0)

    def test_ray_march_respects_max_range(self):
        shape = _shape([[1] * 11])
        mask = build_walkable_mask(shape, threshold=0.5, dilation_cells=0)

        x, y, method = ray_march_landing(
            1, 0, 2, 0, shape, max_range=3, step=0.5, mask=mask,
        )

        self.assertEqual(method, "max_range")
        self.assertAlmostEqual(x, 4)
        self.assertEqual(y, 0)

    def test_self_centered_and_remote_abilities_keep_caster_position(self):
        cast = {"map": "Lotus", "agent": "jett", "ability": "tailwind", "x": 4, "y": 5}
        self.assertEqual(
            estimate_cast_landing(cast, {}),
            {"landing_x": 4.0, "landing_y": 5.0, "landing_method": "caster_position", "estimated": False},
        )
        remote = {**cast, "agent": "astra", "ability": "stars"}
        self.assertEqual(estimate_cast_landing(remote, {}), estimate_cast_landing(cast, {}))

    def test_thrown_range_is_shortened_by_upward_aim(self):
        shape = {
            "density": np.ones((3, 201)),
            "x": np.arange(201, dtype=float) * 50,
            "y": np.arange(3, dtype=float),
        }
        horizontal = {
            "map": "Lotus", "agent": "killjoy", "ability": "nanoswarm",
            "x": 500, "y": 1, "view_x": 1, "view_y": 0, "view_z": 0,
        }
        upward = {**horizontal, "view_z": 1}

        flat_landing = estimate_cast_landing(horizontal, {"Lotus": shape})
        high_landing = estimate_cast_landing(upward, {"Lotus": shape})

        self.assertTrue(flat_landing["estimated"])
        self.assertTrue(high_landing["estimated"])
        self.assertLess(high_landing["landing_x"] - 500, flat_landing["landing_x"] - 500)

    def test_add_estimates_preserves_caster_coordinates_and_adds_required_fields(self):
        shape = _shape([[1] * 20 for _ in range(3)])
        casts = pd.DataFrame([{
            "map": "Lotus", "agent": "cypher", "ability": "trapwire", "x": 2.0, "y": 1.0,
            "view_x": 1.0, "view_y": 0.0, "view_z": 0.0,
        }])

        result = add_landing_estimates(casts, {"Lotus": shape})

        self.assertEqual((result.loc[0, "x"], result.loc[0, "y"]), (2.0, 1.0))
        self.assertEqual(result.loc[0, "landing_method"], "wall_hit")
        self.assertTrue(result.loc[0, "estimated"])
        self.assertIn("landing_x", result.columns)
        self.assertIn("landing_y", result.columns)
        self.assertIn("landing_method", result.columns)
        self.assertIn("estimated", result.columns)


if __name__ == "__main__":
    unittest.main()