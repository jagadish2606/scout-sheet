import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.abilities import CATEGORIES, ability_mapping
from analysis.map_shape import density_for_points, load_map_shapes
from analysis.sites import label_zones, load_site_centres
from analysis.zones import aggregate_zone_cells, eligible_rounds, filter_casts, match_casts_to_rounds


PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
TABLE_NAMES = ("rounds", "casts", "kills", "buys", "plants")
SENTINEL_AGENTS = {"chamber", "cypher", "deadlock", "killjoy", "sage", "vyse"}


st.set_page_config(page_title="Scout Sheet · Utility Map", layout="wide")


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


def _separate_zone_markers(zones, min_distance=300, offset_distance=800):
    """Nudge close zone markers apart while leaving their card data untouched."""
    positions = []
    angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    for row in zones.itertuples(index=False):
        original = (float(row.x), float(row.y))
        if any(math.dist(original, previous) < min_distance for previous in positions):
            candidates = [
                (original[0] + offset_distance * math.cos(angle),
                 original[1] + offset_distance * math.sin(angle))
                for angle in angles
            ]
            original = max(
                candidates,
                key=lambda candidate: min(math.dist(candidate, previous) for previous in positions),
            )
        positions.append(original)
    return [point[0] for point in positions], [point[1] for point in positions]


def _category_counts(casts):
    return casts["category"].value_counts().reindex(CATEGORIES, fill_value=0).astype(int).to_dict()


def _most_common_category(counts, current_category):
    alternatives = [category for category in CATEGORIES if category != current_category and counts[category] > 0]
    return max(alternatives, key=lambda category: counts[category]) if alternatives else None


def _set_ability_category(category):
    st.session_state["ability_category"] = category


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
        marker_x, marker_y = _separate_zone_markers(top_zones)
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


def main():
    st.title("Scout Sheet")
    st.caption("Utility Map · Opponent scouting from local GRID-derived Parquet caches")
    try:
        signature = tuple((CACHE_DIR / f"{name}.parquet").stat().st_mtime_ns for name in TABLE_NAMES)
        tables = load_tables(signature)
    except FileNotFoundError as error:
        st.error(f"Cache file missing: {error.filename or error}. Build it with `uv run build_cache.py` first.")
        st.stop()

    rounds, casts, kills, buys, plants = (tables[name] for name in TABLE_NAMES)
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
        format_func=lambda index: teams.iloc[index]["label"],
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
    selected_side = st.sidebar.selectbox("Side", ("attack", "defense"), index=1)
    side = "attacker" if selected_side == "attack" else "defender"

    time_window = st.sidebar.slider(
        "Live round time window (seconds)", min_value=0, max_value=100, value=(0, 100), step=5,
    )
    include_buy_phase = st.sidebar.toggle(
        "Include buy phase", value=True,
        help="Include casts with negative times, before the round-ended-freezetime event.",
    )

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
    agent = st.sidebar.selectbox("Agent (optional)", ["all", *agent_options], format_func=lambda value: "All agents" if value == "all" else value.title())
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
    category = st.sidebar.selectbox(
        "Ability category", category_options, index=None,
        format_func=lambda value: (
            f"All categories ({len(category_base_casts):,})" if value == "all"
            else f"{value.title()} ({category_counts[value]:,})"
        ),
        key="ability_category",
    )
    show_dots = st.sidebar.toggle("Show individual cast dots", value=False)

    filtered_casts = (
        category_base_casts if category == "all"
        else category_base_casts[category_base_casts["category"].eq(category)].copy()
    )
    category_counts = _category_counts(category_base_casts)
    site_centres = load_site_centres(plants, CACHE_DIR)
    zones = label_zones(aggregate_zone_cells(filtered_casts, denominator), map_name, site_centres)
    shapes = load_shapes(casts, kills)
    shape = shapes.get(map_name)

    series_count = int(denominator["series_id"].nunique())
    game_count = int(denominator[["series_id", "game_id"]].drop_duplicates().shape[0])
    round_count = int(denominator[["series_id", "game_id", "round_number"]].drop_duplicates().shape[0])
    filtered_count = len(filtered_casts)
    if not zones.empty:
        strongest = zones.iloc[0]
        label = strongest["site_label"]
        if label == "Site unknown":
            location = "in an unlabelled area"
        elif label.startswith("near "):
            location = label
        else:
            location = f"at {label}"
        st.markdown(
            f"**{selected_team['team_name']} use utility {location} in "
            f"{strongest['percentage']:.0f}% of {selected_side} rounds "
            f"on {map_name.title()} ({strongest['rounds_used']}/{strongest['total_rounds']}, "
            f"{strongest['series_count']} series).** "
            f"Most common: {strongest['agent'].title()} · {strongest['ability_name']}."
        )
    elif filtered_casts.empty:
        summary = _sentinel_summary(buys, team_id, map_name)
        label = "all categories" if category == "all" else category.title()
        st.info(f"No {label} casts match these filters for {selected_team['team_name']} on {map_name.title()}. {summary}")
        alternative = _most_common_category(category_counts, category)
        if alternative:
            st.button(
                f"Show {alternative.title()} ({category_counts[alternative]:,}) casts",
                on_click=_set_ability_category, args=(alternative,), key="switch_empty_category",
            )
    else:
        st.info("No hot zone is strong enough for the current filters yet.")
    metrics = st.columns(4)
    metrics[0].metric("Series", f"{series_count:,}")
    metrics[1].metric("Games", f"{game_count:,}")
    metrics[2].metric("Rounds", f"{round_count:,}")
    metrics[3].metric("Matching casts", f"{filtered_count:,}")
    if round_count < 20:
        st.warning(f"Small sample: only {round_count} {selected_side} rounds match this team/map/side selection.")
    if unmatched:
        st.info(f"Excluded {unmatched:,} cast rows that did not match a team/map/side round record.")
    st.caption(
        "Zone evidence is per distinct round. The denominator is all rounds for this team, map and side; "
        "player, agent, category, remote-cast and time filters affect casts, not that denominator. "
        "Time starts at round-ended-freezetime; negative seconds are buy-phase casts. Site labels are approximate, "
        "based on plant positions clustered around named site callouts."
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
            st.caption(
                "Grey background: all-team cast and kill position density from the cache, not map artwork. "
                "Ability positions mark caster locations; landing points are not provided."
            )

    with zones_column:
        st.subheader("Hot zones")
        st.caption("Cells are approximately 4 m × 4 m; ranking is by distinct rounds used.")
        if zones.empty:
            if filtered_casts.empty:
                st.caption("No hot zones to show for this empty result.")
            else:
                st.info("Matching casts are present, but none have usable positions for a hot zone.")
        else:
            for index, zone in zones.head(5).iterrows():
                st.markdown(
                    f"#### {zones.index.get_loc(index) + 1}. {zone['site_label']} · {zone['ability_name']}"
                )
                st.write(f"Most common cast: **{zone['agent'].title()}** · {zone['player_name']}")
                st.caption(
                    f"Used in {zone['rounds_used']} of {zone['total_rounds']} {selected_side} rounds "
                    f"({zone['percentage']:.0f}%) on {map_name.title()} across {zone['series_count']} series."
                )
                st.divider()
        if not filtered_casts.empty:
            missing_positions = int(filtered_casts[["x", "y"]].isna().any(axis=1).sum())
            if missing_positions:
                st.caption(f"{missing_positions:,} matching casts have no usable position and are omitted from zones.")

    with st.expander("Observed agent / ability mapping"):
        mapping = ability_mapping(casts)
        st.dataframe(mapping, width="stretch", hide_index=True)
        st.caption(
            "Shared ability IDs are flagged because the same GRID ability ID appeared under multiple agents. "
            "Placeable is restricted to local Cypher, Killjoy, Chamber, Vyse and Deadlock utility; "
            "remote casts are identified and excluded by default."
        )


if __name__ == "__main__":
    main()