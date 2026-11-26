"""Circuit labels and team counts derived from the GRID series index."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


GAME_CHANGERS_CIRCUITS = (
    "Game Changers – North America",
    "Game Changers – Brazil",
    "Game Changers – Latin America",
    "Game Changers – EMEA",
    "Game Changers – Japan",
    "Game Changers – Korea",
    "Game Changers – Southeast Asia/Oceania",
    "Game Changers – Championship",
)
CIRCUITS = ("VCT Americas", "Masters", *GAME_CHANGERS_CIRCUITS)


def _contains_region(event, region_aliases):
    normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    padded = f" {normalize(event)} "
    return any(f" {normalize(alias)} " in padded for alias in region_aliases)


def circuit_for_event(event_name):
    event = str(event_name or "").strip()
    normalized = event.casefold()
    if "game changer" in normalized or "gamechanger" in normalized:
        if "championship" in normalized:
            return "Game Changers – Championship"
        if _contains_region(event, ("north america", "north american", "na", "gcna")):
            return "Game Changers – North America"
        if _contains_region(event, ("brazil", "brasil", " br ", "gcbr")):
            return "Game Changers – Brazil"
        if _contains_region(event, ("latin america", "latam", "latinoamerica", "gc latam")):
            return "Game Changers – Latin America"
        if _contains_region(event, ("emea", "europe", "middle east", "africa")):
            return "Game Changers – EMEA"
        if _contains_region(event, ("japan", "jpn", "jp", "gcjp")):
            return "Game Changers – Japan"
        if _contains_region(event, ("korea", "kr", "gckr")):
            return "Game Changers – Korea"
        if _contains_region(event, ("southeast asia", "sea", "oceania", "anz", "apac", "pacific")):
            return "Game Changers – Southeast Asia/Oceania"
        return "Other events"
    if "masters" in normalized:
        return "Masters"
    if "vct americas" in normalized:
        return "VCT Americas"
    for token, label in (("vct emea", "VCT EMEA"), ("vct pacific", "VCT Pacific"), ("vct china", "VCT China")):
        if token in normalized:
            return label
    if "challengers" in normalized:
        return "Other events"
    if not event:
        return "Other events"
    return f"Other · {event}"


def load_series_event_map(series_csv):
    with Path(series_csv).open(newline="", encoding="utf-8") as file:
        return {
            str(row["series_id"]): row.get("event", "")
            for row in csv.DictReader(file)
            if row.get("series_id")
        }


def team_circuit_labels(rounds, event_by_series):
    """Return one team choice per circuit, with distinct cached-series counts."""
    required = {"team_id", "team_name", "series_id"}
    if rounds.empty or not required.issubset(rounds.columns):
        return pd.DataFrame(columns=["team_id", "team_name", "circuit", "series_count", "label"])
    memberships = rounds.dropna(subset=["team_id", "series_id"])[
        ["team_id", "team_name", "series_id"]
    ].copy()
    memberships["team_id"] = memberships["team_id"].astype(str)
    memberships["series_id"] = memberships["series_id"].astype(str)
    memberships = memberships.drop_duplicates(["team_id", "series_id"])
    memberships["circuit"] = memberships["series_id"].map(
        lambda series_id: circuit_for_event(event_by_series.get(series_id, "")),
    )
    teams = memberships.groupby(["team_id", "circuit"], as_index=False, sort=False).agg(
        team_name=("team_name", lambda values: values.dropna().astype(str).value_counts().index[0] if values.notna().any() else "Unknown team"),
        series_count=("series_id", "nunique"),
    )
    teams["label"] = teams.apply(
        lambda row: f"{row['team_name']} · {int(row['series_count']):,} series",
        axis=1,
    )
    circuit_order = {circuit: index for index, circuit in enumerate(CIRCUITS)}
    teams["_circuit_order"] = teams["circuit"].map(circuit_order).fillna(len(circuit_order))
    return teams.sort_values(
        ["_circuit_order", "series_count", "team_name", "team_id"],
        ascending=[True, False, True, True],
        key=lambda values: values.str.casefold() if values.name in {"team_name", "team_id"} else values,
        kind="stable",
    ).drop(columns="_circuit_order").reset_index(drop=True)