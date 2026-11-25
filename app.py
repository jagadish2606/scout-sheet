import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.abilities import CATEGORIES, ability_mapping
from analysis.map_shape import density_for_points, load_map_shapes
from analysis.operator import (
    aggressiveness_label,
    aggressiveness_score,
    calculate_buy_rates,
    calculate_handoff_rate,
    estimate_spawn_anchors,
    operator_engagements,
)
from analysis.sites import label_zones, load_site_centres, site_label
from analysis.zones import aggregate_zone_cells, eligible_rounds, filter_casts, match_casts_to_rounds


PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
TABLE_NAMES = ("rounds", "casts", "kills", "damage", "buys", "spawn_positions", "plants")
SENTINEL_AGENTS = {"chamber", "cypher", "deadlock", "killjoy", "sage", "vyse"}


st.set_page_config(page_title="Scout Sheet", layout="wide")


@st.cache_data(show_spinner=False)
def load_tables(signature):
    return {
        name: pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        for name in TABLE_NAMES
    }


@st.cache_data(show_spinner="Building map position densities…")
def load_shapes(casts, kills):
    return load_map_shapes(casts, kills, CACHE_DIR)


def _most_common(values):
    values = values.dropna().astype(str)
    if values.empty:
        return None
    return str(values.value_counts().index[0])


def _cast_hover(row):
    seconds = row.get("seconds_into_round")
    if pd.isna(seconds):
        phase = "time unknown"
    elif seconds < 0:
        phase = "buy phase"
    else:
        phase = f"{seconds:.1f}s"
    ability = row.get("ability_name", row.get("ability", "Ability"))
    return f"{ability} · {row.get('player_name', 'Unknown player')} · R{row.get('round_number', '?')} · {phase}"


def _padded_axis_range(shape, axis, padding=0.05):
    lower = float(shape[f"{axis}_min"])
    upper = float(shape[f"{axis}_max"])
    margin = max(upper - lower, 1.0) * padding
    return [lower - margin, upper + margin]


def _separate_zone_markers(zones, shape=None, marker_gap_pixels=54, plot_pixels=600):
    """Separate zone markers by their approximate screen-space footprint."""
    positions = []
    angles = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    if shape:
        map_span = max(
            float(shape["x_max"]) - float(shape["x_min"]),
            float(shape["y_max"]) - float(shape["y_min"]),
        )
    else:
        map_span = 14000.0
    min_distance = map_span * marker_gap_pixels / plot_pixels
    for row in zones.itertuples(index=False):
        location = (float(row.x), float(row.y))
        if positions and any(math.dist(location, previous) < min_distance for previous in positions):
            for ring in range(1, 16):
                radius = min_distance * (1.15 + ring * 0.65)
                candidates = [
                    (location[0] + radius * math.cos(angle), location[1] + radius * math.sin(angle))
                    for angle in angles
                ]
                available = [
                    candidate for candidate in candidates
                    if all(math.dist(candidate, previous) >= min_distance for previous in positions)
                ]
                if available:
                    location = min(available, key=lambda point: math.dist(point, (float(row.x), float(row.y))))
                    break
        positions.append(location)
    return [point[0] for point in positions], [point[1] for point in positions]


def _category_counts(casts):
    return casts["category"].value_counts().reindex(CATEGORIES, fill_value=0).astype(int).to_dict()


def _most_common_category(counts, current_category):
    alternatives = [category for category in CATEGORIES if category != current_category and counts[category] > 0]
    return max(alternatives, key=lambda category: counts[category]) if alternatives else None


def _set_ability_category(category, widget_key=None):
    st.session_state["ability_category"] = category
    if widget_key:
        st.session_state[widget_key] = category


def _persist_filter_value(widget_key, persistent_key):
    value = st.session_state.get(widget_key)
    if value is not None:
        st.session_state[persistent_key] = value


def _sentinel_summary(buys, team_id, map_name):
    selected = buys[
        buys["team_id"].astype(str).eq(str(team_id))
        & buys["map"].astype(str).str.casefold().eq(str(map_name).casefold())
    ]
    if selected.empty:
        return "Sentinel lineup data is unavailable for this map."
    agents = selected["agent"].dropna().astype(str).str.casefold().unique().tolist()
    sentinels = sorted(SENTINEL_AGENTS.intersection(agents))
    if sentinels:
        return "Sentinel agents on this map: " + ", ".join(agent.title() for agent in sentinels) + "."
    if not agents:
        return "Sentinel lineup data is unavailable for this map."
    return "No sentinel played on this map."


def _team_labels(rounds):
    teams = rounds.groupby("team_id", dropna=True).agg(
        team_name=("team_name", _most_common),
        series_count=("series_id", "nunique"),
    ).reset_index()
    teams["label"] = teams.apply(
        lambda row: f"{row['team_name'] or 'Unknown team'} [{row['team_id']}]", axis=1,
    )
    return teams.sort_values(
        ["series_count", "label"],
        ascending=[False, True],
        key=lambda values: values.str.casefold() if values.name == "label" else values,
    ).reset_index(drop=True)


def _map_figure(shape, casts, zones, show_dots, map_name, site_centres):
    figure = go.Figure()
    figure.add_trace(go.Heatmap(
        x=shape["x"], y=shape["y"], z=shape["density"],
        colorscale=[[0, "rgba(0,0,0,0)"], [0.2, "#343b46"], [1, "#a4abb5"]],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip", name="All-team position density",
    ))

    located = casts.dropna(subset=["x", "y"])
    if not located.empty:
        points = located[["x", "y"]].to_numpy(dtype=float)
        cast_density = density_for_points(points, shape, sigma=12)
        if cast_density.max() > 0:
            figure.add_trace(go.Heatmap(
                x=shape["x"], y=shape["y"], z=cast_density,
                colorscale=[
                    [0, "rgba(251,191,36,0)"], [0.12, "rgba(251,146,60,0.65)"],
                    [0.5, "#f97316"], [1, "#ef4444"],
                ],
                zmin=0, zmax=1, opacity=0.95, showscale=False, hoverinfo="skip",
                name="Filtered cast density",
            ))
        if show_dots:
            figure.add_trace(go.Scattergl(
                x=located["x"], y=located["y"], mode="markers",
                marker={"size": 5, "color": "#f8fafc", "opacity": 0.62, "line": {"width": 0}},
                text=located.apply(_cast_hover, axis=1),
                hovertemplate="%{text}<extra></extra>", name="Individual casts",
            ))

    map_sites = site_centres.get(str(map_name).casefold(), {})
    if map_sites:
        figure.add_trace(go.Scatter(
            x=[site["x"] for site in map_sites.values()],
            y=[site["y"] for site in map_sites.values()],
            text=list(map_sites), mode="markers+text", textposition="middle center",
            textfont={"size": 18, "color": "#e0f2fe", "family": "Arial Black"},
            marker={"size": 30, "color": "rgba(14,165,233,0.35)", "line": {"color": "#bae6fd", "width": 2}},
            hovertemplate="%{text} site plant centre<extra></extra>", name="Plant-derived site centres",
        ))

    top_zones = zones.head(5)
    if not top_zones.empty:
        marker_x, marker_y = _separate_zone_markers(top_zones, shape)
        figure.add_trace(go.Scatter(
            x=marker_x, y=marker_y, mode="markers+text",
            text=[str(index) for index in range(1, len(top_zones) + 1)],
            textposition="middle center",
            textfont={"size": 9, "color": "#111827", "family": "Arial Black"},
            marker={"size": 40, "color": "#f59e0b", "line": {"color": "#fff7ed", "width": 2}},
            customdata=top_zones[["ability_name", "rounds_used", "total_rounds", "percentage"]],
            hovertemplate=(
                "%{customdata[0]}<br>Used in %{customdata[1]} of %{customdata[2]} rounds "
                "(%{customdata[3]:.1f}%)<extra></extra>"
            ),
            name="Top zones",
        ))

    x_range = _padded_axis_range(shape, "x")
    y_range = _padded_axis_range(shape, "y")
    figure.update_layout(
        template="plotly_dark", height=650, margin={"l": 8, "r": 8, "t": 16, "b": 8},
        plot_bgcolor="#111827", paper_bgcolor="#111827", showlegend=False, autosize=True,
        xaxis={"title": None, "range": x_range, "showgrid": False, "zeroline": False,
               "showticklabels": False, "ticks": "", "constrain": "range"},
        yaxis={"title": None, "range": y_range, "showgrid": False, "zeroline": False,
               "showticklabels": False, "ticks": "", "constrain": "range"},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1, constrain="range")
    return figure


def _operator_strip_figure(rounds, player_shares):
    ordered = rounds.copy()
    ordered["_round_order"] = pd.to_numeric(ordered["round_number"], errors="coerce")
    ordered["_game_order"] = pd.to_numeric(ordered.get("game_number"), errors="coerce")
    ordered = ordered.sort_values(
        ["series_id", "_game_order", "game_id", "_round_order"], kind="stable",
    ).reset_index(drop=True)
    palette = ["#38bdf8", "#f97316", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#fb7185"]
    player_info = {}
    for row in player_shares.drop_duplicates("player_id").itertuples(index=False):
        player_info[str(row.player_id)] = {
            "name": row.player_name,
            "color": palette[len(player_info) % len(palette)],
        }
    points = {"No Operator": [], "Snapshot unavailable": [], "Multiple holders": []}
    game_labels = []
    game_ticks = []
    row_positions = {}
    game_groups = list(ordered.groupby(["series_id", "game_id"], sort=False))
    for game_index, ((_, _), game) in enumerate(game_groups):
        game_row = game.iloc[0]
        game_number = game_row.get("game_number")
        game_number = str(game_number) if pd.notna(game_number) and str(game_number).strip() else str(game_index + 1)
        opponent = game_row.get("opponent_name")
        opponent = str(opponent) if pd.notna(opponent) and str(opponent).strip() else "Unknown opponent"
        game_labels.append(f"vs {opponent} · G{game_number}")
        game_ticks.append(float(game_index) * 1.3)
        for row in game.itertuples(index=False):
            row_positions[(str(row.series_id), str(row.game_id), str(row.round_number))] = float(game_index) * 1.3

    for row in ordered.itertuples(index=False):
        holders = tuple(row.holders)
        names = tuple(row.holder_names)
        holder_label = ", ".join(names) if names else ("No Operator recorded" if not holders else "Holder name unavailable")
        game_label = game_labels[next(
            index for index, ((series_id, game_id), _) in enumerate(game_groups)
            if str(series_id) == str(row.series_id) and str(game_id) == str(row.game_id)
        )]
        x = float(row.round_number) if pd.notna(row.round_number) else 0.0
        y = row_positions[(str(row.series_id), str(row.game_id), str(row.round_number))]
        hover = f"{game_label} · Round {row.round_number}<br>{holder_label}"
        if not row.snapshot_available:
            points["Snapshot unavailable"].append((x, y, hover))
        elif not holders:
            points["No Operator"].append((x, y, hover))
        elif len(holders) > 1:
            points["Multiple holders"].append((x, y, hover))
        else:
            player_id = str(holders[0])
            player_info.setdefault(player_id, {"name": names[0] if names else "Unknown player", "color": "#f59e0b"})
            points.setdefault(player_id, []).append((x, y, hover))

    figure = go.Figure()
    for player_id, selections in points.items():
        if not selections:
            continue
        if player_id == "No Operator":
            label, color = player_id, "#9ca3af"
        elif player_id == "Snapshot unavailable":
            label, color = player_id, "#4b5563"
        elif player_id == "Multiple holders":
            label, color = player_id, "#f59e0b"
        else:
            label, color = player_info[player_id]["name"], player_info[player_id]["color"]
        figure.add_trace(go.Scatter(
            x=[x for x, _, _ in selections], y=[y for _, y, _ in selections], mode="markers",
            marker={"symbol": "square", "size": 18, "color": color, "line": {"color": "#111827", "width": 1}},
            text=[hover for _, _, hover in selections], hovertemplate="%{text}<extra></extra>", name=label,
        ))
    max_round = int(ordered["_round_order"].max()) if not ordered.empty and ordered["_round_order"].notna().any() else 1
    chart_height = max(180, 38 * len(game_labels) + 70)
    figure.update_layout(
        template="plotly_dark", height=chart_height, margin={"l": 8, "r": 8, "t": 8, "b": 24},
        plot_bgcolor="#111827", paper_bgcolor="#111827", showlegend=True,
        legend={"orientation": "h", "y": 1.15, "x": 0},
        xaxis={"title": "Round number", "range": [0.5, max_round + 0.5], "dtick": 1,
               "showgrid": False, "zeroline": False},
        yaxis={"tickmode": "array", "tickvals": game_ticks, "ticktext": game_labels,
               "range": [-0.5, max(0.5, len(game_labels) * 1.3 - 0.8)],
               "showgrid": False, "zeroline": False},
    )
    return figure


def _team_possessive(team_name):
    return f"{team_name}'" if str(team_name).rstrip().casefold().endswith("s") else f"{team_name}'s"


def _operator_headline(team_name, primary, total_rounds, engagements, map_name, selected_side):
    if primary is None:
        return None
    player_name = primary["player_name"]
    held_rounds = int(primary["rounds"])
    possessive = _team_possessive(team_name)
    if selected_side != "defense":
        scope = "attack rounds across all maps" if map_name == "All maps" else f"{map_name.title()} attack rounds"
        return f"{possessive} primary Operator player {player_name} held it in {held_rounds} of {total_rounds} {scope}; the angle profile below uses defense engagements."
    if len(engagements) < 10:
        suggestion = "; try All maps" if map_name != "All maps" else ""
        scope = "defense rounds across all maps" if map_name == "All maps" else f"{map_name.title()} defense rounds"
        return (
            f"{possessive} primary Operator player {player_name} held it in {held_rounds} of {total_rounds} "
            f"{scope} — too few engagements to profile angles here{suggestion}."
        )

    seconds = pd.to_numeric(engagements["seconds_into_round"], errors="coerce")
    timed = engagements[seconds.notna()]
    early = timed[pd.to_numeric(timed["seconds_into_round"], errors="coerce").between(0, 20, inclusive="left")]
    if map_name == "All maps":
        if timed.empty:
            return f"{possessive} primary Operator player {player_name} has {len(engagements)} defense engagements across all maps; event timing is unavailable."
        return (
            f"{possessive} primary Operator player {player_name} has {100 * len(early) / len(timed):.0f}% of "
            f"their defense Operator engagements in 0–20 s ({len(early)}/{len(timed)}) across all maps."
        )

    known = engagements[~engagements["aggressiveness"].eq("unknown")]
    known_counts = known["aggressiveness"].value_counts()
    if not known.empty and known_counts.iloc[0] / len(known) >= 0.5:
        angle_style = {
            "aggressive": "holds aggressive angles",
            "passive": "holds passive angles",
            "standard": "takes standard angles",
        }.get(known_counts.index[0], "engages")
    else:
        angle_style = "engages"
    site = _most_common(engagements["site_label"]) or "an unknown area"
    timing = (
        f"{100 * len(early) / len(timed):.0f}% of their Op engagements happen 0–20 s into the round "
        f"({len(early)}/{len(timed)})"
        if not timed.empty else "engagement timing is unavailable"
    )
    return f"{possessive} primary Operator player {player_name} {angle_style} at {site} — {timing} on {map_name.title()}."


def _operator_map_figure(shape, engagements, map_name, site_centres):
    empty_zones = pd.DataFrame(columns=["x", "y", "site_label", "ability_name", "rounds_used", "total_rounds", "percentage"])
    figure = _map_figure(shape, engagements.iloc[:0], empty_zones, False, map_name, site_centres)
    for trace in figure.data:
        trace.showlegend = False
    buckets = ("0–20 s", "20–40 s", "40 s+", "Buy phase", "Time unknown")
    colors = {"0–20 s": "#fb7185", "20–40 s": "#f59e0b", "40 s+": "#38bdf8", "Buy phase": "#a78bfa", "Time unknown": "#d1d5db"}
    if not engagements.empty:
        for bucket in buckets:
            selected = engagements[engagements["time_bucket"].eq(bucket)]
            if selected.empty:
                continue
            figure.add_trace(go.Scatter(
                x=selected["x"], y=selected["y"], mode="markers",
                marker={"size": 11, "color": colors[bucket], "line": {"color": "#111827", "width": 1}},
                customdata=selected[["player_name", "seconds_into_round", "site_label", "event_type", "aggressiveness"]],
                hovertemplate=(
                    "%{customdata[0]} · %{customdata[3]}<br>%{customdata[1]:.1f}s · %{customdata[2]}"
                    "<br>%{customdata[4]}<extra></extra>"
                ), name=bucket,
            ))
    figure.update_layout(showlegend=True, legend={"orientation": "h", "y": 1.02, "x": 0})
    return figure


def _time_bucket(seconds):
    if pd.isna(seconds):
        return "Time unknown"
    seconds = float(seconds)
    if seconds < 0:
        return "Buy phase"
    if seconds < 20:
        return "0–20 s"
    if seconds < 40:
        return "20–40 s"
    return "40 s+"


def _profile_engagements(kills, damage, buys, spawn_positions, team_id, map_name, site_centres):
    engagements = operator_engagements(kills, damage, buys, team_id, map_name, "defender")
    if engagements.empty:
        return engagements, pd.DataFrame(columns=["player_name", "engagements", "aggressive", "standard", "passive", "unknown", "aggressive_share"])
    anchors = estimate_spawn_anchors(spawn_positions)
    engagements["seconds_into_round"] = pd.to_numeric(engagements["seconds_into_round"], errors="coerce")
    engagements["time_bucket"] = engagements["seconds_into_round"].map(_time_bucket)
    aggression_scores = []
    site_labels = []
    for row in engagements.itertuples(index=False):
        map_anchors = anchors.get(str(row.map).casefold(), {})
        defender_spawn = map_anchors.get("defender")
        attacker_spawn = map_anchors.get("attacker")
        aggression_scores.append(
            aggressiveness_score((row.x, row.y), defender_spawn, attacker_spawn)
            if defender_spawn is not None and attacker_spawn is not None else None
        )
        site_labels.append(site_label(row.map, row.x, row.y, site_centres))
    engagements["aggression_score"] = aggression_scores
    engagements["aggressiveness"] = engagements["aggression_score"].map(aggressiveness_label)
    engagements["site_label"] = site_labels
    player_rows = []
    for player_id, group in engagements.groupby("player_id", dropna=False, sort=False):
        counts = group["aggressiveness"].value_counts()
        player_rows.append({
            "player_name": _most_common(group["player_name"]),
            "player_id": player_id,
            "engagements": len(group),
            "aggressive": int(counts.get("aggressive", 0)),
            "standard": int(counts.get("standard", 0)),
            "passive": int(counts.get("passive", 0)),
            "unknown": int(counts.get("unknown", 0)),
            "aggressive_share": 100 * int(counts.get("aggressive", 0)) / len(group),
        })
    player_summary = pd.DataFrame(player_rows).sort_values(
        ["engagements", "player_name"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    return engagements, player_summary


def _render_utility_tab(tables, team_name, team_id, map_name, selected_side, side, shapes, site_centres):
    rounds, casts, buys = tables["rounds"], tables["casts"], tables["buys"]
    denominator = eligible_rounds(rounds, team_id, map_name, side)
    matched, unmatched = match_casts_to_rounds(casts, denominator)
    player_rows = matched.dropna(subset=["player_id", "player_name"])[["player_id", "player_name"]].drop_duplicates()
    player_rows["label"] = player_rows["player_name"].astype(str) + " [" + player_rows["player_id"].astype(str) + "]"
    player_rows = player_rows.sort_values("label", key=lambda values: values.str.casefold())
    player_ids = ["all", *player_rows["player_id"].astype(str).tolist()]
    player_labels = {"all": "All players"}
    player_labels.update(dict(zip(player_rows["player_id"].astype(str), player_rows["label"])))
    player_id = st.sidebar.selectbox(
        "Player (optional)", player_ids, format_func=lambda value: player_labels[value],
    )
    agent_options = sorted(matched["agent"].dropna().astype(str).unique(), key=str.casefold)
    agent = st.sidebar.selectbox(
        "Agent (optional)", ["all", *agent_options],
        format_func=lambda value: "All agents" if value == "all" else value.title(),
    )
    time_window = st.sidebar.slider(
        "Live round time window (seconds)", min_value=0, max_value=100, value=(0, 100), step=5,
    )
    include_buy_phase = st.sidebar.toggle(
        "Include buy phase", value=True,
        help="Include casts with negative times, before the round-ended-freezetime event.",
    )
    include_remote = st.sidebar.toggle(
        "Include remote casts", value=False,
        help="Remote abilities are excluded by default because caster position is not the utility location.",
    )
    category_base_casts = filter_casts(
        matched, player_id=player_id, agent=agent, seconds_range=time_window,
        include_remote=include_remote, include_buy_phase=include_buy_phase,
    )
    category_counts = _category_counts(category_base_casts)
    category_options = ["all", *CATEGORIES]
    if st.session_state.get("ability_category") not in category_options:
        st.session_state["ability_category"] = "placeable"
    filter_signature = repr((team_id, map_name, side, player_id, agent, time_window, include_buy_phase, include_remote))
    key_suffix = hashlib.blake2b(filter_signature.encode(), digest_size=8).hexdigest()
    category_widget_key = f"ability_category_{key_suffix}"
    category = st.sidebar.selectbox(
        "Ability category", category_options,
        index=category_options.index(st.session_state["ability_category"]),
        format_func=lambda value: (
            f"All categories ({len(category_base_casts):,})" if value == "all"
            else f"{value.title()} ({category_counts[value]:,})"
        ),
        key=category_widget_key,
    )
    st.session_state["ability_category"] = category
    show_dots = st.sidebar.toggle("Show individual cast dots", value=False)

    filtered_casts = (
        category_base_casts if category == "all"
        else category_base_casts[category_base_casts["category"].eq(category)].copy()
    )
    zones = label_zones(aggregate_zone_cells(filtered_casts, denominator), map_name, site_centres)
    shape = shapes.get(map_name)
    series_count = int(denominator["series_id"].nunique())
    game_count = int(denominator[["series_id", "game_id"]].drop_duplicates().shape[0])
    round_count = int(denominator[["series_id", "game_id", "round_number"]].drop_duplicates().shape[0])
    if not zones.empty:
        strongest = zones.iloc[0]
        label = strongest["site_label"]
        location = label if label.startswith("near ") or label == "Site unknown" else f"at {label}"
        st.markdown(
            f"**{team_name} use utility {location} in {strongest['percentage']:.0f}% of {selected_side} rounds "
            f"on {map_name.title()} ({strongest['rounds_used']}/{strongest['total_rounds']}, "
            f"{strongest['series_count']} series).** Most common: "
            f"{strongest['agent'].title()} · {strongest['ability_name']}."
        )
    elif filtered_casts.empty:
        summary = _sentinel_summary(buys, team_id, map_name)
        label = "all categories" if category == "all" else category.title()
        st.info(f"No {label} casts match these filters for {team_name} on {map_name.title()}. {summary}")
        alternative = _most_common_category(category_counts, category)
        if alternative:
            st.button(
                f"Show {alternative.title()} ({category_counts[alternative]:,}) casts",
                on_click=_set_ability_category, args=(alternative, category_widget_key), key="switch_empty_category",
            )
    else:
        st.info("No hot zone is strong enough for the current filters yet.")

    metrics = st.columns(4)
    metrics[0].metric("Series", f"{series_count:,}")
    metrics[1].metric("Games", f"{game_count:,}")
    metrics[2].metric("Rounds", f"{round_count:,}")
    metrics[3].metric("Matching casts", f"{len(filtered_casts):,}")
    if round_count < 20:
        st.warning(f"Small sample: only {round_count} {selected_side} rounds match this team/map/side selection.")
    if unmatched:
        st.info(f"Excluded {unmatched:,} cast rows that did not match a team/map/side round record.")
    with st.expander("How this is calculated"):
        st.markdown(
            "Zone evidence is per distinct round. The denominator is all rounds for this team, map and side; "
            "player, agent, category, remote-cast and time filters affect casts, not that denominator. "
            "Round time starts at `round-ended-freezetime`; negative seconds are buy-phase casts. Site labels "
            "are approximate, based on plant positions clustered around named site callouts. The grey background "
            "uses all-team cast and kill positions from the local cache rather than Riot map artwork. Ability "
            "positions mark caster locations, not utility landing points."
        )

    map_column, zones_column = st.columns([1.7, 1])
    with map_column:
        st.subheader(f"{map_name.title()} · {selected_side.title()}")
        if shape is None:
            st.warning("No cached positions are available for this map.")
        else:
            st.plotly_chart(
                _map_figure(shape, filtered_casts, zones, show_dots, map_name, site_centres),
                width="stretch",
            )
        if not filtered_casts.empty:
            missing_positions = int(filtered_casts[["x", "y"]].isna().any(axis=1).sum())
            if missing_positions:
                st.caption(f"{missing_positions:,} matching casts have no usable position and are omitted from zones.")
    with zones_column:
        st.subheader("Hot zones")
        st.caption("Cells are approximately 4 m × 4 m; ranking is by distinct rounds used.")
        if zones.empty:
            if filtered_casts.empty:
                st.caption("No hot zones to show for this empty result.")
            else:
                st.info("Matching casts are present, but none have usable positions for a hot zone.")
        else:
            for index, zone in enumerate(zones.head(5).itertuples(index=False), start=1):
                st.markdown(f"#### {index}. {zone.site_label} · {zone.ability_name}")
                st.write(f"Most common cast: **{zone.agent.title()}** · {zone.player_name}")
                st.caption(
                    f"Used in {zone.rounds_used} of {zone.total_rounds} {selected_side} rounds "
                    f"({zone.percentage:.0f}%) on {map_name.title()} across {zone.series_count} series."
                )
                st.divider()
    with st.expander("Observed agent / ability mapping"):
        st.dataframe(ability_mapping(casts), width="stretch", hide_index=True)
        st.caption(
            "Shared ability IDs are flagged because the same GRID ability ID appeared under multiple agents. "
            "Placeable is restricted to local Cypher, Killjoy, Chamber, Vyse and Deadlock utility; remote casts "
            "are identified and excluded by default."
        )


def _render_operator_tab(tables, team_name, team_id, map_name, selected_side, shapes, site_centres):
    rounds, buys = tables["rounds"], tables["buys"]
    rates = calculate_buy_rates(rounds, buys, team_id, map_name)
    selected_side_key = "attacker" if selected_side == "attack" else "defender"
    selected_rates = calculate_buy_rates(rounds, buys, team_id, map_name, side=selected_side_key)
    overall = rates["overall"]
    strip_rounds = selected_rates["strip_rounds"]
    handoff = calculate_handoff_rate(strip_rounds)
    map_mask = pd.Series(True, index=rounds.index) if map_name == "All maps" else rounds["map"].astype(str).str.casefold().eq(map_name.casefold())
    series_count = int(rounds[rounds["team_id"].astype(str).eq(str(team_id)) & map_mask]["series_id"].nunique())
    eligible_count = overall["eligible_rounds"]
    observed_count = overall["observed_rounds"]
    selected_eligible_count = selected_rates["overall"]["eligible_rounds"]
    display_map = map_name if map_name == "All maps" else map_name.title()
    engagements, player_aggression = _profile_engagements(
        tables["kills"], tables["damage"], buys, tables["spawn_positions"],
        team_id, map_name, site_centres,
    )
    primary = selected_rates["players"].iloc[0].to_dict() if not selected_rates["players"].empty else None
    primary_engagements = pd.DataFrame()
    if primary is not None and selected_side == "defense" and not engagements.empty:
        primary_engagements = engagements[engagements["player_id"].astype(str).eq(str(primary["player_id"]))]
        if primary_engagements.empty:
            primary_engagements = engagements[engagements["player_name"].eq(primary["player_name"])]
    headline = _operator_headline(
        team_name, primary, selected_rates["overall"]["observed_rounds"],
        primary_engagements, map_name, selected_side,
    )
    st.markdown(f"**{team_name} Operator profile · {display_map}**")
    if headline:
        st.markdown(f"**{headline}**")
    metrics = st.columns(4)
    metrics[0].metric("Series" if map_name == "All maps" else "Map series", f"{series_count:,}")
    metrics[1].metric("Team rounds", f"{eligible_count:,}")
    metrics[2].metric("Buy snapshots", f"{observed_count:,}")
    metrics[3].metric("Operator rounds", f"{overall['held_rounds']:,}")
    st.caption(f"Buy snapshot coverage: {observed_count:,} of {eligible_count:,} team rounds.")
    if observed_count < 20:
        st.warning(f"Small sample: only {observed_count} rounds have a recorded freeze-time buy snapshot.")
    if selected_eligible_count < 20:
        st.warning(f"Small sample: only {selected_eligible_count} team rounds are in the selected {selected_side} filter.")

    rate_columns = st.columns(3)
    for column, label, summary in (
        (rate_columns[0], "Overall buy rate", overall),
        (rate_columns[1], "Attack buy rate", rates["by_side"].get("attacker", {"held_rounds": 0, "observed_rounds": 0, "eligible_rounds": 0, "rate": 0})),
        (rate_columns[2], "Defense buy rate", rates["by_side"].get("defender", {"held_rounds": 0, "observed_rounds": 0, "eligible_rounds": 0, "rate": 0})),
    ):
        column.metric(label, f"{summary['held_rounds']} / {summary['observed_rounds']} ({summary['rate']:.0f}%)")
        column.caption(f"Buy snapshots: {summary['observed_rounds']} / {summary['eligible_rounds']} rounds")

    st.subheader("Who buys the Operator")
    player_column, agent_column = st.columns(2)
    player_shares = selected_rates["players"].drop(columns=["player_id"], errors="ignore").rename(
        columns={"player_name": "Player", "agents": "Agent(s)", "rounds": "Rounds", "share": "Share"},
    )
    agent_shares = selected_rates["agents"].rename(
        columns={"agent": "Agent", "rounds": "Rounds", "share": "Share"},
    )
    with player_column:
        st.markdown(f"**Per player · {selected_side} Operator rounds**")
        st.dataframe(
            player_shares, width="stretch", hide_index=True,
            column_config={"Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100)},
        )
    with agent_column:
        st.markdown(f"**Per agent · {selected_side} Operator rounds**")
        st.dataframe(
            agent_shares, width="stretch", hide_index=True,
            column_config={"Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100)},
        )

    st.subheader(f"Possession · {selected_side.title()} rounds")
    st.plotly_chart(_operator_strip_figure(strip_rounds, selected_rates["players"]), width="stretch")
    st.metric(
        "Handoff rate between consecutive Operator rounds",
        f"{handoff['handoffs']} / {handoff['comparisons']} ({handoff['rate']:.0f}%)",
    )
    st.caption("A handoff is a different holder set between successive Operator rounds; comparisons reset each game and across missing snapshots.")

    st.subheader("Defense setup positions and engagements")
    st.caption("Operator-holder kill/damage encounters only; times are measured from round-ended-freezetime.")
    if len(engagements) < 20:
        st.warning(f"Small sample: {len(engagements)} deduplicated Operator engagements on defense.")
    if map_name == "All maps":
        st.info("Pick one map in the Map filter to view the engagement map; All maps is used for aggregate timing evidence only.")
    elif shapes.get(map_name) is None:
        st.warning("No cached positions are available for this map.")
    elif engagements.empty:
        st.info("No recorded defense Operator engagements match this team and map.")
    else:
        st.plotly_chart(_operator_map_figure(shapes[map_name], engagements, map_name, site_centres), width="stretch")

    st.subheader("Aggressive vs passive angles")
    st.caption("Spawn anchors are medians of freeze-end player positions by map and side; this is a rough directional proxy.")
    profiled_players = player_aggression[player_aggression["engagements"].ge(10)]
    if profiled_players.empty:
        st.info("No player has 10 or more defense engagements in this filter; per-player angle profiles are withheld.")
    else:
        display_aggression = profiled_players.drop(columns=["player_id"], errors="ignore")
        st.dataframe(display_aggression, width="stretch", hide_index=True)
    st.caption(f"Engagement sample: {len(engagements)} defense engagements; time buckets omit unknown times.")
    st.caption(
        "Possession is sampled at round-ended-freezetime and does not track later pickups or drops. "
        "Rapid damage ticks against one player are grouped with a resulting kill when within 1.5 seconds."
    )


def main():
    try:
        signature = tuple((CACHE_DIR / f"{name}.parquet").stat().st_mtime_ns for name in TABLE_NAMES)
        tables = load_tables(signature)
    except FileNotFoundError as error:
        st.error(f"Cache file missing: {error.filename or error}. Build it with `uv run build_cache.py` first.")
        st.stop()

    rounds, casts, kills = (tables[name] for name in ("rounds", "casts", "kills"))
    teams = _team_labels(rounds)
    if teams.empty:
        st.error("The rounds cache has no teams to analyze.")
        st.stop()

    default_team = next(
        (index for index, name in enumerate(teams["team_name"]) if name == "Cloud9"), 0,
    )
    st.sidebar.header("Filters")
    team_index = st.sidebar.selectbox(
        "Opponent team", range(len(teams)), index=default_team,
        format_func=lambda index: teams.iloc[index]["label"], key="shared_team",
    )
    selected_team = teams.iloc[team_index]
    team_id = str(selected_team["team_id"])

    team_rounds = rounds[rounds["team_id"].astype(str).eq(team_id)]
    map_counts = team_rounds.drop_duplicates(["series_id", "game_id"]).groupby("map").size()
    maps = map_counts.sort_values(ascending=False, kind="stable").index.astype(str).tolist()
    if not maps:
        st.warning("No rounds are available for this team.")
        st.stop()
    active_tab = st.session_state.get("scout_tabs", "Utility Map")
    if active_tab == "Operator Profiler":
        map_options = ["All maps", *maps]
        map_widget_key = f"operator_map_widget_{team_id}"
        persistent_map_key = f"operator_map_selection_{team_id}"
    else:
        map_options = maps
        map_widget_key = f"utility_map_widget_{team_id}"
        persistent_map_key = f"utility_map_selection_{team_id}"
    if persistent_map_key not in st.session_state or st.session_state[persistent_map_key] not in map_options:
        st.session_state[persistent_map_key] = map_options[0]
    if map_widget_key not in st.session_state or st.session_state[map_widget_key] not in map_options:
        st.session_state[map_widget_key] = st.session_state[persistent_map_key]
    map_name = st.sidebar.selectbox(
        "Map", map_options, key=map_widget_key,
        on_change=_persist_filter_value, args=(map_widget_key, persistent_map_key),
    )
    selected_side = st.sidebar.selectbox("Side", ("attack", "defense"), index=1, key="shared_side")
    side = "attacker" if selected_side == "attack" else "defender"
    shapes = load_shapes(casts, kills)
    plants = tables["plants"]
    site_centres = load_site_centres(plants, CACHE_DIR)
    title_col, badge_col = st.columns([5, 1])
    title_col.title("Scout Sheet")
    if selected_team["team_name"] in {"Cloud9", "Cloud9 GC"}:
        badge_col.markdown(
            '<div style="margin-top: 18px; padding: 6px 12px; border-radius: 999px; '
            'background: #164e63; color: #cffafe; font-weight: 700; text-align: center;">Self-scout</div>',
            unsafe_allow_html=True,
        )
    st.caption("VALORANT opponent scouting from local GRID-derived Parquet caches")
    utility_tab, operator_tab = st.tabs(
        ["Utility Map", "Operator Profiler"], key="scout_tabs", on_change="rerun",
    )
    with utility_tab:
        if utility_tab.open:
            _render_utility_tab(
                tables, selected_team["team_name"], team_id, map_name, selected_side, side, shapes, site_centres,
            )
    with operator_tab:
        if operator_tab.open:
            _render_operator_tab(
                tables, selected_team["team_name"], team_id, map_name, selected_side, shapes, site_centres,
            )


if __name__ == "__main__":
    main()