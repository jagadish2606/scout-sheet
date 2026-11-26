import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.sites import (
    build_site_centres,
    disambiguate_zone_labels,
    label_zones,
    load_site_centres,
    site_label,
)


class SiteCentreTests(unittest.TestCase):
    def setUp(self):
        self.plants = pd.DataFrame([
            {"map": "Lotus", "x": 7800, "y": 5500},
            {"map": "Lotus", "x": 7700, "y": 5600},
            {"map": "Lotus", "x": 6700, "y": -4200},
            {"map": "Lotus", "x": 6600, "y": -4300},
            {"map": "Unknown", "x": 100, "y": 100},
        ])

    def test_plants_cluster_around_named_sites_and_label_zones(self):
        centres = build_site_centres(self.plants)
        self.assertEqual(set(centres["lotus"]), {"A", "C"})
        self.assertEqual(centres["lotus"]["A"]["plant_count"], 2)
        self.assertEqual(site_label("Lotus", 7750, 5550, centres), "A site")
        self.assertEqual(site_label("Lotus", 0, 0, centres), "mid")
        self.assertEqual(site_label("Unknown", 0, 0, centres), "Site unknown")
        self.assertEqual(site_label("Summit", 0, 0, centres), "Unlabeled area")
        zones = label_zones(pd.DataFrame([{"x": 7750, "y": 5550}]), "Lotus", centres)
        self.assertEqual(zones.iloc[0]["site_label"], "A site")

    def test_centres_are_persisted_and_loaded_from_plant_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            first = load_site_centres(self.plants, directory)
            second = load_site_centres(self.plants, directory)
            self.assertEqual(first, second)
            self.assertTrue((Path(directory) / "site_centres.json").is_file())

    def test_only_duplicate_site_and_ability_labels_in_top_five_get_spot_numbers(self):
        zones = pd.DataFrame([
            {"site_label": "A site", "ability": "trapwire"},
            {"site_label": "A site", "ability": "trapwire"},
            {"site_label": "A site", "ability": "spycam"},
            {"site_label": "near B site", "ability": "trapwire"},
            {"site_label": "A site", "ability": "trapwire"},
            {"site_label": "A site", "ability": "trapwire"},
        ])

        labeled = disambiguate_zone_labels(zones)

        self.assertEqual(labeled["site_label"].tolist(), [
            "A site", "A site (2nd spot)", "A site", "near B site", "A site (3rd spot)", "A site",
        ])


if __name__ == "__main__":
    unittest.main()