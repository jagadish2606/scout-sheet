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
    ordered = ordered.sort_values(["series_id", "game_id", "_round_order"], kind="stable").reset_index(drop=True)
    palette = ["#38bdf8", "#f97316", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#fb7185"]
    player_info = {}
    for row in player_shares.drop_duplicates("player_id").itertuples(index=False):
        player_info[str(row.player_id)] = {
            "name": row.player_name,
            "color": palette[len(player_info) % len(palette)],
        }
    points = {"No Operator": [], "Snapshot unavailable": [], "Multiple holders": []}
    for index, row in enumerate(ordered.itertuples(index=False)):
        holders = tuple(row.holders)
        names = tuple(row.holder_names)
        holder_label = ", ".join(names) if names else (", ".join(holders) if holders else "No Operator recorded")
        hover = f"{row.series_id} · {row.game_id} · Round {row.round_number}<br>{holder_label}"
        if not row.snapshot_available:
            points["Snapshot unavailable"].append((index, hover))
        elif not holders:
            points["No Operator"].append((index, hover))
        elif len(holders) > 1:
            points["Multiple holders"].append((index, hover))
        else:
            player_id = str(holders[0])
            player_info.setdefault(player_id, {"name": names[0] if names else player_id, "color": "#f59e0b"})
            points.setdefault(player_id, []).append((index, hover))

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
            x=[index for index, _ in selections], y=[0] * len(selections), mode="markers",
            marker={"symbol": "square", "size": 18, "color": color, "line": {"color": "#111827", "width": 1}},
            text=[hover for _, hover in selections], hovertemplate="%{text}<extra></extra>", name=label,
        ))
    figure.update_layout(
        template="plotly_dark", height=150, margin={"l": 8, "r": 8, "t": 8, "b": 24},
        plot_bgcolor="#111827", paper_bgcolor="#111827", showlegend=True,
        legend={"orientation": "h", "y": 1.15, "x": 0},
        xaxis={"title": "Rounds · hover a cell for game and round", "showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"visible": False, "range": [-1, 1]},
    )
    return figure


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
    anchors = estimate_spawn_anchors(spawn_positions).get(str(map_name).casefold(), {})
    defender_spawn = anchors.get("defender")
    attacker_spawn = anchors.get("attacker")
    engagements["seconds_into_round"] = pd.to_numeric(engagements["seconds_into_round"], errors="coerce")
    engagements["time_bucket"] = engagements["seconds_into_round"].map(_time_bucket)
    engagements["aggression_score"] = [
        aggressiveness_score((row.x, row.y), defender_spawn, attacker_spawn)
        if defender_spawn is not None and attacker_spawn is not None else None
        for row in engagements.itertuples(index=False)
    ]
    engagements["aggressiveness"] = engagements["aggression_score"].map(aggressiveness_label)
    engagements["site_label"] = [
        site_label(map_name, row.x, row.y, site_centres)
        for row in engagements.itertuples(index=False)
    ]
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
    overall = rates["overall"]
    strip_rounds = rates["strip_rounds"][rates["strip_rounds"]["side"].astype(str).str.casefold().eq(
        "attacker" if selected_side == "attack" else "defender",
    )]
    handoff = calculate_handoff_rate(strip_rounds)
    series_count = int(rounds[
        rounds["team_id"].astype(str).eq(str(team_id))
        & rounds["map"].astype(str).str.casefold().eq(map_name.casefold())
    ]["series_id"].nunique())
    eligible_count = overall["eligible_rounds"]
    observed_count = overall["observed_rounds"]
    st.markdown(f"**{team_name} Operator profile · {map_name.title()}**")
    metrics = st.columns(4)
    metrics[0].metric("Map series", f"{series_count:,}")
    metrics[1].metric("Team rounds", f"{eligible_count:,}")
    metrics[2].metric("Buy snapshots", f"{observed_count:,}")
    metrics[3].metric("Operator rounds", f"{overall['held_rounds']:,}")
    st.caption(f"Buy snapshot coverage: {observed_count:,} of {eligible_count:,} team rounds.")
    if observed_count < 20:
        st.warning(f"Small sample: only {observed_count} rounds have a recorded freeze-time buy snapshot.")

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
    player_shares = rates["players"].copy()
    agent_shares = rates["agents"].copy()
    if not player_shares.empty:
        player_shares["share"] = player_shares["share"].map(lambda value: f"{value:.0f}%")
    if not agent_shares.empty:
        agent_shares["share"] = agent_shares["share"].map(lambda value: f"{value:.0f}%")
    with player_column:
        st.markdown("**Per player · share of Operator rounds**")
        st.dataframe(player_shares, width="stretch", hide_index=True)
    with agent_column:
        st.markdown("**Per agent · share of Operator rounds**")
        st.dataframe(agent_shares, width="stretch", hide_index=True)

    st.subheader(f"Possession · {selected_side.title()} rounds")
    st.plotly_chart(_operator_strip_figure(strip_rounds, rates["players"]), width="stretch")
    st.metric(
        "Handoff rate between consecutive Operator rounds",
        f"{handoff['handoffs']} / {handoff['comparisons']} ({handoff['rate']:.0f}%)",
    )
    st.caption("A handoff is a different holder set between successive Operator rounds; comparisons reset each game and across missing snapshots.")

    engagements, player_aggression = _profile_engagements(
        tables["kills"], tables["damage"], buys, tables["spawn_positions"],
        team_id, map_name, site_centres,
    )
    st.subheader("Defense setup positions and engagements")
    st.caption("Operator-holder kill/damage encounters only; colors show elapsed time from round start.")
    if len(engagements) < 20:
        st.warning(f"Small sample: {len(engagements)} deduplicated Operator engagements on defense.")
    shape = shapes.get(map_name)
    if shape is None:
        st.warning("No cached positions are available for this map.")
    elif engagements.empty:
        st.info("No recorded defense Operator engagements match this team and map.")
    else:
        st.plotly_chart(_operator_map_figure(shape, engagements, map_name, site_centres), width="stretch")

    st.subheader("Aggressive vs passive angles")
    st.caption("Spawn anchors are medians of freeze-end player positions by map and side; this is a rough directional proxy.")
    if player_aggression.empty:
        st.info("No engagements are available to score.")
    else:
        st.dataframe(player_aggression, width="stretch", hide_index=True)
        primary = rates["players"].iloc[0] if not rates["players"].empty else None
        if primary is not None:
            primary_engagements = engagements[engagements["player_id"].astype(str).eq(str(primary["player_id"]))]
            if primary_engagements.empty:
                primary_engagements = engagements[engagements["player_name"].eq(primary["player_name"])]
            if primary_engagements.empty:
                st.info(f"{primary['player_name']} held the most Operators, but has no recorded defense engagements in this sample.")
            else:
                timed = primary_engagements[pd.to_numeric(primary_engagements["seconds_into_round"], errors="coerce").notna()]
                early = timed[timed["seconds_into_round"].between(0, 20, inclusive="left")]
                site = _most_common(primary_engagements["site_label"]) or "an unknown area"
                aggressive_count = int(primary_engagements["aggressiveness"].eq("aggressive").sum())
                if aggressive_count / len(primary_engagements) >= 0.5:
                    angle_style = "holds aggressive angles"
                elif primary_engagements["aggressiveness"].eq("passive").mean() >= 0.5:
                    angle_style = "holds passive angles"
                else:
                    angle_style = "takes standard angles"
                if len(timed):
                    early_percent = 100 * len(early) / len(timed)
                    headline = (
                        f"{team_name}'s primary Operator player {primary['player_name']} {angle_style} at {site} — "
                        f"{early_percent:.0f}% of his Op engagements happen 0–20 s into the round "
                        f"({len(early)}/{len(timed)})."
                    )
                else:
                    headline = f"{team_name}'s primary Operator player {primary['player_name']} {angle_style} at {site}; engagement timing is unavailable."
                st.markdown(f"**{headline}**")
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
    map_name = st.sidebar.selectbox("Map", maps, index=0, key=f"map_{team_id}")
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