"""Round-based Operator possession and engagement analysis."""

import json
import math
from pathlib import Path

import pandas as pd


ROUND_KEYS = ["series_id", "game_id", "round_number"]


def _map_mask(frame, map_name):
    if str(map_name).casefold() == "all maps":
        return pd.Series(True, index=frame.index)
    return frame["map"].astype(str).str.casefold().eq(str(map_name).casefold())


def _round_key_columns(frame):
    result = frame.copy()
    for column in ROUND_KEYS:
        result[column] = result[column].astype("string")
    return result


def _sort_rounds(frame):
    result = frame.copy()
    result["_round_order"] = pd.to_numeric(result["round_number"], errors="coerce")
    result = result.sort_values(["series_id", "game_id", "_round_order"], kind="stable")
    return result.drop(columns="_round_order")


def _rate_summary(eligible_rounds, observed_rounds, held_rounds):
    return {
        "eligible_rounds": int(eligible_rounds),
        "observed_rounds": int(observed_rounds),
        "held_rounds": int(held_rounds),
        "rate": 100 * held_rounds / observed_rounds if observed_rounds else 0.0,
        "coverage": 100 * observed_rounds / eligible_rounds if eligible_rounds else 0.0,
    }


def calculate_buy_rates(rounds, buys, team_id, map_name, side=None):
    """Calculate Operator rates from rounds with a recorded freeze-time snapshot.

    Returns the full round coverage as well as observed-snapshot denominators, so
    absent buy records are never interpreted as confirmed no-Operator rounds.
    """
    selected_rounds = rounds[
        rounds["team_id"].astype(str).eq(str(team_id))
        & _map_mask(rounds, map_name)
    ].drop_duplicates(ROUND_KEYS + ["side"]).copy()
    if side is not None:
        selected_rounds = selected_rounds[
            selected_rounds["side"].astype(str).str.casefold().eq(str(side).casefold())
        ]
    if selected_rounds.empty:
        empty_rounds = pd.DataFrame(columns=ROUND_KEYS + ["side", "holders", "holder_names", "snapshot_available"])
        empty_shares = pd.DataFrame(columns=["player_id", "player_name", "agents", "rounds", "share"])
        return {
            "overall": _rate_summary(0, 0, 0), "by_side": {}, "rounds": empty_rounds,
            "strip_rounds": empty_rounds, "players": empty_shares,
            "agents": pd.DataFrame(columns=["agent", "rounds", "share"]),
        }

    selected_buys = buys[
        buys["team_id"].astype(str).eq(str(team_id))
        & _map_mask(buys, map_name)
    ].copy()
    if side is not None:
        selected_buys = selected_buys[
            selected_buys["side"].astype(str).str.casefold().eq(str(side).casefold())
        ]
    selected_rounds = _round_key_columns(selected_rounds)
    selected_buys = _round_key_columns(selected_buys)
    observed_keys = selected_buys[ROUND_KEYS].drop_duplicates()
    round_rows = selected_rounds.merge(observed_keys, on=ROUND_KEYS, how="inner", validate="one_to_one")

    selected_buys["has_operator"] = selected_buys["has_operator"].fillna(False).astype(bool)
    holders = selected_buys[selected_buys["has_operator"]].copy()
    holder_groups = {}
    name_groups = {}
    for key, group in holders.groupby(ROUND_KEYS, sort=False, dropna=False):
        key = tuple(str(value) for value in key)
        holder_groups[key] = tuple(dict.fromkeys(group["player_id"].astype(str)))
        name_groups[key] = tuple(dict.fromkeys(group["player_name"].dropna().astype(str)))
    observed_set = set(
        tuple(row) for row in observed_keys.astype(str).itertuples(index=False, name=None)
    )

    def attach_holders(frame):
        frame["_key"] = list(zip(*(frame[column].astype(str) for column in ROUND_KEYS)))
        frame["snapshot_available"] = frame["_key"].isin(observed_set)
        frame["holders"] = frame["_key"].map(lambda key: holder_groups.get(key, ()))
        frame["holder_names"] = frame["_key"].map(lambda key: name_groups.get(key, ()))
        return _sort_rounds(frame.drop(columns="_key")).reset_index(drop=True)

    round_rows = attach_holders(round_rows)
    strip_rounds = attach_holders(selected_rounds)

    held_round_rows = round_rows[round_rows["holders"].map(bool)]
    held_count = len(held_round_rows)
    observed_count = len(round_rows)
    overall = _rate_summary(len(selected_rounds), observed_count, held_count)
    by_side = {}
    for side, eligible in selected_rounds.groupby("side", dropna=False, sort=False):
        group = round_rows[round_rows["side"].astype(str).str.casefold().eq(str(side).casefold())]
        by_side[str(side)] = _rate_summary(
            len(eligible), len(group), int(group["holders"].map(bool).sum()),
        )

    if holders.empty:
        player_rows = pd.DataFrame(columns=["player_id", "player_name", "agents", "rounds"])
        agent_rows = pd.DataFrame(columns=["agent", "rounds"])
    else:
        player_rows = holders.groupby("player_id", dropna=False).apply(
            lambda group: pd.Series({
                "player_name": (
                    group["player_name"].dropna().astype(str).value_counts().index[0]
                    if group["player_name"].notna().any() else "Unknown player"
                ),
                "agents": ", ".join(sorted(group["agent"].dropna().astype(str).unique(), key=str.casefold))
                if group["agent"].notna().any() else "Unknown",
                "rounds": group[ROUND_KEYS].drop_duplicates().shape[0],
            }),
            include_groups=False,
        ).reset_index()
        agent_rows = holders.groupby("agent", dropna=False).apply(
            lambda group: group[ROUND_KEYS].drop_duplicates().shape[0], include_groups=False,
        ).rename("rounds").reset_index()
    player_rows["share"] = 100 * player_rows["rounds"] / held_count if held_count else 0.0
    player_rows = player_rows.sort_values(["rounds", "player_name"], ascending=[False, True], kind="stable")

    agent_rows["share"] = 100 * agent_rows["rounds"] / held_count if held_count else 0.0
    agent_rows = agent_rows.sort_values(["rounds", "agent"], ascending=[False, True], kind="stable")

    return {
        "overall": overall,
        "by_side": by_side,
        "rounds": round_rows,
        "strip_rounds": strip_rounds,
        "players": player_rows.reset_index(drop=True),
        "agents": agent_rows.reset_index(drop=True),
    }


def calculate_handoff_rate(round_holders):
    """Compare holder sets across successive known Operator rounds within each game."""
    if round_holders.empty:
        return {"handoffs": 0, "comparisons": 0, "rate": 0.0}
    selected = _sort_rounds(round_holders.copy())
    handoffs = comparisons = 0
    for _, game in selected.groupby(["series_id", "game_id"], sort=False):
        previous = None
        for row in game.itertuples(index=False):
            if hasattr(row, "snapshot_available") and not row.snapshot_available:
                previous = None
                continue
            current = frozenset(row.holders)
            if not current:
                continue
            if previous is not None:
                comparisons += 1
                handoffs += previous != current
            previous = current
    return {
        "handoffs": int(handoffs),
        "comparisons": int(comparisons),
        "rate": 100 * handoffs / comparisons if comparisons else 0.0,
    }


def estimate_spawn_anchors(spawn_positions):
    """Use per-map, per-side median freeze-end positions as spawn proxies."""
    if spawn_positions.empty:
        return {}
    points = spawn_positions.copy()
    points["x"] = pd.to_numeric(points["x"], errors="coerce")
    points["y"] = pd.to_numeric(points["y"], errors="coerce")
    points = points.dropna(subset=["map", "side", "x", "y"])
    result = {}
    for (map_name, side), group in points.groupby(
        [points["map"].astype(str).str.casefold(), points["side"].astype(str).str.casefold()],
        sort=False,
    ):
        result.setdefault(str(map_name), {})[str(side)] = (
            float(group["x"].median()), float(group["y"].median()),
        )
    return result


def load_spawn_estimates(path):
    """Load persisted full-data anchors as case-folded map/side coordinate tuples."""
    try:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return {}
    maps = data.get("maps", data) if isinstance(data, dict) else {}
    result = {}
    for map_name, sides in maps.items():
        if not isinstance(sides, dict):
            continue
        anchors = {}
        for side, point in sides.items():
            try:
                if isinstance(point, dict):
                    coordinate = (float(point["x"]), float(point["y"]))
                else:
                    coordinate = (float(point[0]), float(point[1]))
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if all(math.isfinite(value) for value in coordinate):
                anchors[str(side).casefold()] = coordinate
        if anchors:
            result[str(map_name).casefold()] = anchors
    return result


def aggressiveness_score(position, defender_spawn, attacker_spawn):
    """Return normalized progress from defender spawn toward attacker spawn."""
    try:
        x, y = float(position[0]), float(position[1])
        defender = (float(defender_spawn[0]), float(defender_spawn[1]))
        attacker = (float(attacker_spawn[0]), float(attacker_spawn[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not all(math.isfinite(value) for value in (*[x, y], *defender, *attacker)):
        return None
    if math.dist(defender, attacker) == 0:
        return None
    distance_defender = math.dist((x, y), defender)
    distance_attacker = math.dist((x, y), attacker)
    total = distance_defender + distance_attacker
    return distance_defender / total if total else None


def aggressiveness_label(score):
    if score is None or pd.isna(score):
        return "unknown"
    if score > 0.55:
        return "aggressive"
    if score < 0.40:
        return "passive"
    return "standard"


def _event_rows(kills, damage, buys, team_id, map_name, side="defender"):
    possession = buys[
        buys["team_id"].astype(str).eq(str(team_id))
        & _map_mask(buys, map_name)
        & buys["has_operator"].fillna(False).astype(bool)
    ].copy()
    possession = _round_key_columns(possession)
    holder_keys = set(
        tuple(row) for row in possession[ROUND_KEYS + ["player_id"]].astype(str).itertuples(index=False, name=None)
    )
    events = []
    for frame, kind, id_column, name_column, team_column, agent_column, x_column, y_column in (
        (kills, "kill", "killer_id", "killer_name", "killer_team_id", "killer_agent", "killer_x", "killer_y"),
        (damage, "damage", "player_id", "player_name", "team_id", "agent", "x", "y"),
    ):
        if frame.empty:
            continue
        selected = frame[
            frame[team_column].astype(str).eq(str(team_id))
            & _map_mask(frame, map_name)
            & frame["side" if kind == "damage" else "killer_side"].astype(str).str.casefold().eq(str(side).casefold())
            & frame["weapon"].astype(str).str.casefold().eq("operator")
        ]
        for row in selected.to_dict("records"):
            player_id = str(row.get(id_column))
            key = tuple(str(row.get(column)) for column in ROUND_KEYS) + (player_id,)
            if key not in holder_keys:
                continue
            events.append({
                "series_id": row.get("series_id"),
                "game_id": row.get("game_id"),
                "map": row.get("map"),
                "round_number": row.get("round_number"),
                "player_id": player_id,
                "player_name": row.get(name_column),
                "team_id": row.get(team_column),
                "side": row.get("side" if kind == "damage" else "killer_side"),
                "agent": row.get(agent_column),
                "victim_id": row.get("victim_id"),
                "seconds_into_round": row.get("seconds_into_round"),
                "event_time": row.get("event_time"),
                "x": row.get(x_column),
                "y": row.get(y_column),
                "event_type": kind,
            })
    return pd.DataFrame(events)


def operator_engagements(kills, damage, buys, team_id, map_name, side="defender", deduplicate=True):
    """Collect Operator kill/damage events by freeze-time holder on a selected side."""
    events = _event_rows(kills, damage, buys, team_id, map_name, side)
    if events.empty or not deduplicate:
        return events
    return deduplicate_engagements(events)


def deduplicate_engagements(events, gap_seconds=1.5):
    """Collapse rapid damage ticks and a resulting kill into one opponent encounter."""
    if events.empty:
        return events.copy()
    rows = []
    keys = ROUND_KEYS + ["player_id", "victim_id"]
    for _, group in events.groupby(keys, sort=False, dropna=False):
        group = group.copy()
        group["seconds_into_round"] = pd.to_numeric(group["seconds_into_round"], errors="coerce")
        group = group.sort_values("seconds_into_round", kind="stable", na_position="last")
        cluster = []
        cluster_end = None
        for row in group.to_dict("records"):
            timestamp = row.get("seconds_into_round")
            timestamp = float(timestamp) if pd.notna(timestamp) else None
            if cluster and (timestamp is None or cluster_end is None or timestamp - cluster_end > gap_seconds):
                rows.append(_representative_engagement(cluster))
                cluster = []
            cluster.append(row)
            cluster_end = timestamp
        if cluster:
            rows.append(_representative_engagement(cluster))
    return pd.DataFrame(rows)


def _representative_engagement(cluster):
    representative = next((row for row in cluster if row["event_type"] == "kill"), cluster[0]).copy()
    representative["source_events"] = len(cluster)
    return representative