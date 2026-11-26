"""Circuit labels and team counts derived from the GRID series index."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


_REGIONS = (
    ("North America", ("north america", "na", "north-america")),
    ("Latin America North", ("latam north", "latin america north", "latam norte")),
    ("Latin America South", ("latam south", "latin america south", "latam sur")),
    ("Brazil", ("brazil", "brasil")),
    ("EMEA", ("emea", "europe")),
    ("Pacific", ("pacific", "apac", "asia pacific")),
    ("Southeast Asia", ("southeast asia", "sea")),
    ("Japan", ("japan", "jp")),
    ("Korea", ("korea", "kr")),
    ("China", ("china", "cn")),
    ("Oceania", ("oceania", "anz")),
)


def _contains_region(event, region_aliases):
    padded = f" {re.sub(r'[^a-z0-9]+', ' ', event.casefold()).strip()} "
    return any(f" {alias} " in padded for alias in region_aliases)


def circuit_for_event(event_name):
    event = str(event_name or "").strip()
    normalized = event.casefold()
    if "game changer" in normalized or "gamechanger" in normalized:
        region = next(
            (label for label, aliases in _REGIONS if _contains_region(event, aliases)),
            "Region not specified",
        )
        return f"Game Changers · {region}"
    if "masters" in normalized:
        return "Masters"
    if "vct americas" in normalized:
        return "VCT Americas"
    for token, label in (("vct emea", "VCT EMEA"), ("vct pacific", "VCT Pacific"), ("vct china", "VCT China")):
        if token in normalized:
            return label
    if "challengers" in normalized:
        region = next(
            (label for label, aliases in _REGIONS if _contains_region(event, aliases)),
            "Region not specified",
        )
        return f"Challengers · {region}"
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
        lambda row: f"{row['circuit']} · {row['team_name']} [{row['team_id']}] · "
        f"{int(row['series_count'])} series",
        axis=1,
    )
    priority = {"VCT Americas": 0, "Masters": 1}
    teams["_circuit_order"] = teams["circuit"].map(priority).fillna(2)
    return teams.sort_values(
        ["_circuit_order", "circuit", "team_name", "team_id"],
        key=lambda values: values.str.casefold() if values.name in {"circuit", "team_name", "team_id"} else values,
        kind="stable",
    ).drop(columns="_circuit_order").reset_index(drop=True)