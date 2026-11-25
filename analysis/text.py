"""Shared wording helpers for analysis summaries."""


def team_possessive(team_name):
    name = str(team_name).strip()
    return f"{name}'" if name.casefold().endswith("s") else f"{name}'s"