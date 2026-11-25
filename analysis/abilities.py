"""Ability labels and categories used by the Utility Map."""

import pandas as pd


CATEGORIES = ("placeable", "recon", "smoke", "stall/wall", "flash", "movement", "other")

_CATEGORY_BY_ABILITY = {}
for _category, _abilities in {
    "recon": {
        "boom-bot", "haunt", "hunter's-fury", "neural-theft", "nightfall", "owl-drone",
        "prowler", "recon-bolt", "seekers", "stealth-drone", "trailblazer", "zero-point",
    },
    "smoke": {
        "cloudburst", "cove", "dark-cover", "nebula-dissipate", "poison-cloud", "ruse",
        "sky-smoke", "stars", "toxic-screen", "viper's-pit",
    },
    "stall/wall": {
        "aftershock", "annihilation", "armageddon", "barrier-mesh", "barrier-orb", "blaze",
        "cascade", "contingency", "fault-line", "fast-lane", "frag-ment", "gravnet",
        "gravity-well", "guided-salvo", "high-tide", "hot-hands", "incendiary", "meddle",
        "lockdown", "mosh-pit", "nova-pulse", "orbital-strike", "paint-shells", "reckoning",
        "relay-bolt", "rolling-thunder", "seize", "shock-bolt", "slow-orb", "snake-bite",
        "special-delivery", "storm-surge", "steel-garden", "saturate", "thrash", "undercut",
        "wingman",
    },
    "flash": {
        "blindside", "curveball", "dizzy", "flash-drive", "flashpoint", "guiding-light",
        "leer", "paranoia",
    },
    "movement": {
        "blast-pack", "convergent-paths", "dimensional-drift", "from-the-shadows", "gatecrash",
        "high-gear", "light-speed", "pick-me-up", "refract", "rendezvous", "shrouded-step",
        "tailwind", "updraft",
    },
}.items():
    for _ability in _abilities:
        _CATEGORY_BY_ABILITY[_ability] = _category

_PLACEABLE_BY_AGENT = {
    "cypher": {"spycam", "trapwire"},
    "killjoy": {"alarmbot", "nanoswarm", "turret"},
    "chamber": {"trademark"},
    "vyse": {"arc-rose", "razorvine", "shear"},
    "deadlock": {"sonic-sensor"},
}

_REMOTE_CAST_BY_AGENT = {
    "astra": {"stars", "nebula-dissipate", "gravity-well", "nova-pulse"},
    "brimstone": {"sky-smoke", "orbital-strike"},
    "clove": {"ruse"},
    "omen": {"dark-cover"},
}

_DISPLAY_NAMES = {
    "frag-ment": "FRAG/ment",
    "null-cmd": "NULL/cmd",
    "viper's-pit": "Viper's Pit",
    "hunter's-fury": "Hunter's Fury",
    "razorvine": "Razorvine",
    "kill-contract": "Kill Contract",
    "tour-de-force": "Tour de Force",
}


def _normalize(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower().replace("_", "-")


def ability_category(ability, agent=None):
    """Return the agent-aware category for an ability ID."""
    if ability is None or pd.isna(ability):
        return "other"
    ability = _normalize(ability)
    agent = _normalize(agent)
    if ability in _PLACEABLE_BY_AGENT.get(agent, set()):
        return "placeable"
    if ability == "cyber-cage":
        return "smoke"
    return _CATEGORY_BY_ABILITY.get(ability, "other")


def ability_is_remote_cast(ability, agent=None):
    """Return whether the caster's location is a poor proxy for utility location."""
    ability = _normalize(ability)
    agent = _normalize(agent)
    return ability in _REMOTE_CAST_BY_AGENT.get(agent, set())


def ability_display_name(ability):
    """Convert a GRID ability ID into a readable label."""
    if ability is None or pd.isna(ability) or not str(ability).strip():
        return "Unknown ability"
    ability = str(ability).strip().lower()
    if ability in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[ability]
    return " ".join(part.capitalize() for part in ability.replace("_", "-").split("-"))


def ability_mapping(casts):
    """Return one categorized, readable row for each observed agent/ability pair."""
    observed = casts[["agent", "ability"]].drop_duplicates().copy()
    observed["agent"] = observed["agent"].fillna("Unknown agent")
    observed["ability"] = observed["ability"].fillna("unknown")
    observed["category"] = [
        ability_category(ability, agent) for agent, ability in zip(observed["agent"], observed["ability"])
    ]
    observed["display_name"] = observed["ability"].map(ability_display_name)
    observed["remote_cast"] = [
        ability_is_remote_cast(ability, agent)
        for agent, ability in zip(observed["agent"], observed["ability"])
    ]

    agents_per_ability = observed.groupby("ability")["agent"].transform("nunique")
    observed["agent_pair_flag"] = agents_per_ability.gt(1).map(
        {True: "Shared ability ID across agents", False: ""},
    )
    category_order = {category: index for index, category in enumerate(CATEGORIES)}
    observed["_category_order"] = observed["category"].map(category_order)
    return observed.sort_values(
        ["_category_order", "agent", "ability"], kind="stable",
    ).drop(columns="_category_order").reset_index(drop=True)