"""Round-matched utility filtering and 4 m cell evidence calculations."""

import math

import pandas as pd

from analysis.abilities import ability_category, ability_display_name, ability_is_remote_cast


ROUND_KEYS = ["series_id", "game_id", "round_number"]
ZONE_COLUMNS = [
    "cell_x", "cell_y", "x", "y", "rounds_used", "total_rounds", "percentage",
    "series_count", "ability", "ability_name", "agent", "player_name", "casts",
    "cells", "round_keys", "source_counts",
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


def filter_casts(
    casts, category=None, player_id=None, agent=None, seconds_range=(0, 100), include_remote=False,
    include_buy_phase=False,
):
    """Apply cast-only filters; these do not change the round denominator."""
    selected = casts.copy()
    selected["category"] = [
        ability_category(ability, cast_agent)
        for ability, cast_agent in zip(selected["ability"], selected["agent"])
    ]
    selected["ability_name"] = selected["ability"].map(ability_display_name)
    selected["remote_cast"] = pd.Series([
        ability_is_remote_cast(ability, cast_agent)
        for ability, cast_agent in zip(selected["ability"], selected["agent"])
    ], index=selected.index, dtype=bool)
    if not include_remote:
        selected = selected[~selected["remote_cast"]]
    if category and category != "all":
        selected = selected[selected["category"].eq(category)]
    if player_id and player_id != "all":
        selected = selected[selected["player_id"].astype(str).eq(str(player_id))]
    if agent and agent != "all":
        selected = selected[selected["agent"].astype(str).str.casefold().eq(str(agent).casefold())]
    start, end = seconds_range
    times = pd.to_numeric(selected["seconds_into_round"], errors="coerce")
    time_matches = times.between(start, end, inclusive="both")
    if include_buy_phase:
        time_matches |= times.lt(0)
    selected = selected[time_matches].copy()
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
    for (cell_x, cell_y, ability_value), group in valid.groupby(
        ["cell_x", "cell_y", "ability"], dropna=False, sort=False,
    ):
        rounds_used = int(group["round_key"].nunique())
        source_counts = tuple(
            ("Unknown" if pd.isna(agent) else str(agent),
             "Unknown" if pd.isna(player) else str(player), int(count))
            for (agent, player), count in group.groupby(
                ["agent", "player_name"], dropna=False, sort=False,
            ).size().items()
        )
        agent, player_name, _ = max(source_counts, key=lambda source: source[2])
        ability = "Unknown" if pd.isna(ability_value) else str(ability_value)
        round_keys = tuple(sorted(set(group["round_key"])))
        rows.append({
            "cell_x": int(cell_x),
            "cell_y": int(cell_y),
            "x": (cell_x + 0.5) * cell_size,
            "y": (cell_y + 0.5) * cell_size,
            "rounds_used": rounds_used,
            "total_rounds": total_rounds,
            "percentage": 100 * rounds_used / total_rounds,
            "series_count": len({round_key[0] for round_key in round_keys}),
            "ability": ability,
            "ability_name": ability_display_name(ability),
            "agent": agent,
            "player_name": player_name,
            "casts": int(len(group)),
            "cells": ((int(cell_x), int(cell_y)),),
            "round_keys": round_keys,
            "source_counts": source_counts,
        })
    return pd.DataFrame(rows, columns=ZONE_COLUMNS).sort_values(
        ["rounds_used", "casts", "cell_x", "cell_y"], ascending=[False, False, True, True],
    ).reset_index(drop=True)


def merge_zone_cells(zones, cell_size=400, max_distance=600):
    """Merge connected neighbouring cells with the same ability and site label."""
    if zones.empty:
        return zones.reindex(columns=ZONE_COLUMNS).copy()

    merged = []
    for _, group in zones.groupby(["site_label", "ability"], dropna=False, sort=False):
        rows = {index: row for index, row in group.iterrows()}
        cell_to_index = {
            tuple(cell): index
            for index, row in rows.items()
            for cell in (row.get("cells") or ((int(row.cell_x), int(row.cell_y)),))
        }
        unvisited = set(rows)
        neighbour_steps = math.ceil(max_distance / cell_size) + 1
        while unvisited:
            start = min(unvisited)
            unvisited.remove(start)
            component = [start]
            queue = [start]
            while queue:
                current = queue.pop()
                current_cells = rows[current].get("cells") or ((int(rows[current].cell_x), int(rows[current].cell_y)),)
                for cell_x, cell_y in current_cells:
                    for dx in range(-neighbour_steps, neighbour_steps + 1):
                        for dy in range(-neighbour_steps, neighbour_steps + 1):
                            gap_x = max(abs(dx) - 1, 0) * cell_size
                            gap_y = max(abs(dy) - 1, 0) * cell_size
                            if math.hypot(gap_x, gap_y) > max_distance:
                                continue
                            neighbour = cell_to_index.get((cell_x + dx, cell_y + dy))
                            if neighbour in unvisited:
                                unvisited.remove(neighbour)
                                component.append(neighbour)
                                queue.append(neighbour)

            component_rows = [rows[index] for index in component]
            round_keys = tuple(sorted({key for row in component_rows for key in row.round_keys}))
            cells = tuple(sorted({cell for row in component_rows for cell in row.cells}))
            source_totals = {}
            for row in component_rows:
                for agent, player_name, count in row.source_counts:
                    key = (agent, player_name)
                    source_totals[key] = source_totals.get(key, 0) + int(count)
            agent, player_name = max(source_totals, key=source_totals.get)
            cast_count = sum(int(row.casts) for row in component_rows)
            merged_x = sum(float(row.x) * int(row.casts) for row in component_rows) / cast_count
            merged_y = sum(float(row.y) * int(row.casts) for row in component_rows) / cast_count
            first = component_rows[0]
            rounds_used = len(round_keys)
            merged.append({
                "cell_x": cells[0][0], "cell_y": cells[0][1], "x": merged_x, "y": merged_y,
                "rounds_used": rounds_used, "total_rounds": int(first.total_rounds),
                "percentage": 100 * rounds_used / int(first.total_rounds),
                "series_count": len({key[0] for key in round_keys}),
                "ability": str(first.ability), "ability_name": str(first.ability_name),
                "agent": agent, "player_name": player_name, "casts": cast_count,
                "cells": cells, "round_keys": round_keys,
                "source_counts": tuple((a, p, count) for (a, p), count in source_totals.items()),
                "site_label": str(first.site_label),
            })
    result = pd.DataFrame(merged)
    return result.sort_values(
        ["rounds_used", "casts", "cell_x", "cell_y"],
        ascending=[False, False, True, True], kind="stable",
    ).reset_index(drop=True)


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
    """Calculate merged per-zone use and its fixed team/map/side denominator."""
    denominator = eligible_rounds(rounds, team_id, map_name, side)
    if denominator.empty:
        return pd.DataFrame(columns=ZONE_COLUMNS)
    matched, _ = match_casts_to_rounds(casts, denominator)
    filtered = filter_casts(matched, category, player_id, agent, seconds_range)
    cells = aggregate_zone_cells(filtered, denominator, cell_size)
    if not cells.empty:
        cells["site_label"] = "Site unknown"
    return merge_zone_cells(cells, cell_size=cell_size)