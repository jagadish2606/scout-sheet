import unittest

import pandas as pd

from analysis.abilities import ability_category, ability_display_name, ability_mapping


class AbilityMappingTests(unittest.TestCase):
    def test_observed_pairs_are_categorized_and_shared_ids_are_flagged(self):
        casts = pd.DataFrame([
            {"agent": "cypher", "ability": "trapwire"},
            {"agent": "killjoy", "ability": "trapwire"},
            {"agent": "omen", "ability": "dark-cover"},
        ])
        mapping = ability_mapping(casts).set_index(["agent", "ability"])
        self.assertEqual(mapping.loc[("cypher", "trapwire"), "category"], "placeable")
        self.assertEqual(mapping.loc[("omen", "dark-cover"), "display_name"], "Dark Cover")
        self.assertEqual(mapping.loc[("cypher", "trapwire"), "agent_pair_flag"], "Shared ability ID across agents")

    def test_unknown_ability_has_safe_fallbacks(self):
        self.assertEqual(ability_category("new-grid-ability"), "other")
        self.assertEqual(ability_display_name("new-grid-ability"), "New Grid Ability")


if __name__ == "__main__":
    unittest.main()