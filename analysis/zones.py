"""Round-matched utility filtering and 4 m cell evidence calculations."""

import math

import pandas as pd

from analysis.abilities import ability_category, ability_display_name


ROUND_KEYS = ["series_id", "game_id", "round_number"]
ZONE_COLUMNS = [
    "cell_x", "cell_y", "x", "y", "rounds_used", "total_rounds", "percentage",
    "series_count", "ability", "ability_name", "agent", "player_name", "casts",
]


def eligible_rounds(rounds, team_id, map_name, side):
    """Return distinct denominator rounds for the selected team, map and side."""
    selected = rounds[
        rounds["team_id"].astype(str).eq(str(team_id))
        & rounds["map"].astype(str).str.casefold().eq(str(map_name).casefold())
        & rounds["side"].astype(str).str.casefold().eq(str(side).casefold())
    ].copy()
    return selected.drop_duplicates(ROUND_KEYS).reset_index(drop=True)


def match_casts_to_rounds(casts, rounds_for_team):
    """Restrict casts to actual selected-team round records; return excluded count."""
    cast_scope = casts[
        casts["team_id"].astype(str).isin(rounds_for_team["team_id"].astype(str).unique())
        & casts["map"].astype(str).str.casefold().isin(
            rounds_for_team["map"].astype(str).str.casefold().unique(),
        )
        & casts["side"].astype(str).str.casefold().isin(
            rounds_for_team["side"].astype(str).str.casefold().unique(),
        )
    ].copy()
    if rounds_for_team.empty:
        return cast_scope.iloc[0:0].copy(), len(cast_scope)

    keys = ROUND_KEYS + ["team_id", "map", "side"]
    left = cast_scope.copy()
    right = rounds_for_team[keys].drop_duplicates().copy()
    for frame in (left, right):
        for key in keys:
            frame[key] = frame[key].astype("string")
        frame["map"] = frame["map"].str.casefold()
        frame["side"] = frame["side"].str.casefold()
    joined = left.merge(right, on=keys, how="left", indicator=True, validate="many_to_one")
    unmatched = int(joined["_merge"].eq("left_only").sum())
    return joined.loc[joined["_merge"].eq("both")].drop(columns="_merge").copy(), unmatched


def filter_casts(casts, category=None, player_id=None, agent=None, seconds_range=(0, 100)):
    """Apply cast-only filters; these do not change the round denominator."""
    selected = casts.copy()
    selected["category"] = selected["ability"].map(ability_category)
    selected["ability_name"] = selected["ability"].map(ability_display_name)
    if category and category != "all":
        selected = selected[selected["category"].eq(category)]
    if player_id and player_id != "all":
        selected = selected[selected["player_id"].astype(str).eq(str(player_id))]
    if agent and agent != "all":
        selected = selected[selected["agent"].astype(str).str.casefold().eq(str(agent).casefold())]
    start, end = seconds_range
    times = pd.to_numeric(selected["seconds_into_round"], errors="coerce")
    selected = selected[times.between(start, end, inclusive="both")].copy()
    selected["seconds_into_round"] = times.loc[selected.index]
    return selected


def aggregate_zone_cells(casts, denominator_rounds, cell_size=400):
    """Aggregate casts into cells and count distinct rounds used in each cell."""
    total_rounds = len(denominator_rounds.drop_duplicates(ROUND_KEYS))
    valid = casts.dropna(subset=["x", "y"]).copy()
    if valid.empty or total_rounds == 0:
        return pd.DataFrame(columns=ZONE_COLUMNS)
    valid["x"] = pd.to_numeric(valid["x"], errors="coerce")
    valid["y"] = pd.to_numeric(valid["y"], errors="coerce")
    valid = valid.dropna(subset=["x", "y"])
    if valid.empty:
        return pd.DataFrame(columns=ZONE_COLUMNS)
    valid["cell_x"] = (valid["x"] / cell_size).map(math.floor)
    valid["cell_y"] = (valid["y"] / cell_size).map(math.floor)
    valid["round_key"] = list(zip(*(valid[column].astype(str) for column in ROUND_KEYS)))

    rows = []
    for (cell_x, cell_y), group in valid.groupby(["cell_x", "cell_y"], sort=False):
        rounds_used = int(group["round_key"].nunique())
        ability = _mode(group["ability"])
        rows.append({
            "cell_x": int(cell_x),
            "cell_y": int(cell_y),
            "x": (cell_x + 0.5) * cell_size,
            "y": (cell_y + 0.5) * cell_size,
            "rounds_used": rounds_used,
            "total_rounds": total_rounds,
            "percentage": 100 * rounds_used / total_rounds,
            "series_count": int(group["series_id"].nunique()),
            "ability": ability,
            "ability_name": ability_display_name(ability),
            "agent": _mode(group["agent"]),
            "player_name": _mode(group["player_name"]),
            "casts": int(len(group)),
        })
    return pd.DataFrame(rows, columns=ZONE_COLUMNS).sort_values(
        ["rounds_used", "casts", "cell_x", "cell_y"], ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _mode(values):
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return "Unknown"
    counts = non_null.value_counts()
    return str(counts.index[0])


def calculate_zone_percentage(
    rounds,
    casts,
    team_id,
    map_name,
    side,
    category=None,
    player_id=None,
    agent=None,
    seconds_range=(0, 100),
    cell_size=400,
):
    """Calculate per-zone distinct-round use and its fixed team/map/side denominator."""
    denominator = eligible_rounds(rounds, team_id, map_name, side)
    matched, _ = match_casts_to_rounds(casts, denominator)
    filtered = filter_casts(matched, category, player_id, agent, seconds_range)
    return aggregate_zone_cells(filtered, denominator, cell_size)