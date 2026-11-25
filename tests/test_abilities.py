import unittest

import pandas as pd

from analysis.abilities import ability_category, ability_display_name, ability_is_remote_cast, ability_mapping


class AbilityMappingTests(unittest.TestCase):
    def test_observed_pairs_are_categorized_and_shared_ids_are_flagged(self):
        casts = pd.DataFrame([
            {"agent": "cypher", "ability": "trapwire"},
            {"agent": "killjoy", "ability": "trapwire"},
            {"agent": "omen", "ability": "dark-cover"},
            {"agent": "astra", "ability": "stars"},
            {"agent": "killjoy", "ability": "nanoswarm"},
        ])
        mapping = ability_mapping(casts).set_index(["agent", "ability"])
        self.assertEqual(mapping.loc[("cypher", "trapwire"), "category"], "placeable")
        self.assertEqual(mapping.loc[("omen", "dark-cover"), "display_name"], "Dark Cover")
        self.assertEqual(mapping.loc[("astra", "stars"), "category"], "smoke")
        self.assertTrue(mapping.loc[("astra", "stars"), "remote_cast"])
        self.assertEqual(mapping.loc[("killjoy", "nanoswarm"), "category"], "damage")
        self.assertEqual(mapping.loc[("cypher", "trapwire"), "agent_pair_flag"], "Shared ability ID across agents")

    def test_unknown_ability_has_safe_fallbacks(self):
        self.assertEqual(ability_category("new-grid-ability"), "other")
        self.assertEqual(ability_display_name("new-grid-ability"), "New Grid Ability")

    def test_astra_star_activations_are_remote_but_thrown_smokes_are_not(self):
        for ability in ("stars", "nebula-dissipate", "gravity-well", "nova-pulse"):
            self.assertTrue(ability_is_remote_cast(ability, "astra"))
        self.assertFalse(ability_is_remote_cast("cove", "harbor"))
        self.assertFalse(ability_is_remote_cast("poison-cloud", "viper"))

    def test_damaging_utility_uses_damage_category_without_losing_short_placements(self):
        damaging = (
            ("sova", "shock-bolt"), ("viper", "snake-bite"), ("tejo", "armageddon"),
            ("tejo", "guided-salvo"), ("brimstone", "incendiary"),
            ("raze", "paint-shells"), ("raze", "blast-pack"), ("raze", "showstopper"),
            ("kay/o", "frag-ment"),
            ("gekko", "mosh-pit"),
        )
        for agent, ability in damaging:
            with self.subTest(agent=agent, ability=ability):
                self.assertEqual(ability_category(ability, agent), "damage")
        self.assertEqual(ability_category("nanoswarm", "killjoy"), "damage")
        self.assertEqual(ability_category("turret", "killjoy"), "damage")
        self.assertEqual(ability_category("alarmbot", "killjoy"), "placeable")
        self.assertEqual(ability_category("shear", "vyse"), "placeable")
        self.assertEqual(ability_category("special-delivery", "neon"), "stall/wall")
        self.assertEqual(ability_category("storm-surge", "waylay"), "stall/wall")


if __name__ == "__main__":
    unittest.main()