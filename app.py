from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.abilities import CATEGORIES, ability_mapping
from analysis.map_shape import density_for_points, load_map_shapes
from analysis.zones import aggregate_zone_cells, eligible_rounds, filter_casts, match_casts_to_rounds


PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
TABLE_NAMES = ("rounds", "casts", "kills")


st.set_page_config(page_title="Scout Sheet · Utility Map", layout="wide")


@st.cache_data(show_spinner=False)
def load_tables():
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


def _team_labels(rounds):
    teams = rounds.groupby("team_id", dropna=True).agg(
        team_name=("team_name", _most_common),
    ).reset_index()
    teams["label"] = teams.apply(
        lambda row: f"{row['team_name'] or 'Unknown team'} [{row['team_id']}]", axis=1,
    )
    return teams.sort_values("label", key=lambda values: values.str.casefold()).reset_index(drop=True)


def _map_figure(shape, casts, zones, show_dots):
    figure = go.Figure()
    figure.add_trace(go.Heatmap(
        x=shape["x"], y=shape["y"], z=shape["density"],
        colorscale=[[0, "rgba(0,0,0,0)"], [0.2, "#343b46"], [1, "#a4abb5"]],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip", name="All-team position density",
    ))

    located = casts.dropna(subset=["x", "y"])
    if not located.empty:
        points = located[["x", "y"]].to_numpy(dtype=float)
        cast_density = density_for_points(points, shape)
        if cast_density.max() > 0:
            figure.add_trace(go.Heatmap(
                x=shape["x"], y=shape["y"], z=cast_density,
                colorscale=[[0, "rgba(251,113,133,0)"], [0.35, "#fb7185"], [1, "#f43f5e"]],
                zmin=0, zmax=1, opacity=0.82, showscale=False, hoverinfo="skip",
                name="Filtered cast density",
            ))
        if show_dots:
            figure.add_trace(go.Scattergl(
                x=located["x"], y=located["y"], mode="markers",
                marker={"size": 5, "color": "#f8fafc", "opacity": 0.62, "line": {"width": 0}},
                text=located.apply(
                    lambda row: f"{row.get('ability_name', row.get('ability', 'Ability'))} · "
                    f"{row.get('player_name', 'Unknown player')} · "
                    f"R{row.get('round_number', '?')}", axis=1,
                ),
                hovertemplate="%{text}<extra></extra>", name="Individual casts",
            ))

    top_zones = zones.head(5)
    if not top_zones.empty:
        figure.add_trace(go.Scatter(
            x=top_zones["x"], y=top_zones["y"], mode="markers+text",
            text=[str(index) for index in range(1, len(top_zones) + 1)],
            textposition="middle center",
            marker={"size": 27, "color": "#f59e0b", "line": {"color": "#fff7ed", "width": 2}},
            customdata=top_zones[["ability_name", "rounds_used", "total_rounds", "percentage"]],
            hovertemplate=(
                "Zone %{text}<br>%{customdata[0]}<br>Used in %{customdata[1]} of "
                "%{customdata[2]} rounds (%{customdata[3]:.1f}%)<extra></extra>"
            ),
            name="Top zones",
        ))

    figure.update_layout(
        template="plotly_dark", height=650, margin={"l": 8, "r": 8, "t": 16, "b": 8},
        plot_bgcolor="#111827", paper_bgcolor="#111827", showlegend=False,
        xaxis={"title": "Map x (GRID units)", "showgrid": False, "zeroline": False},
        yaxis={"title": "Map y (GRID units)", "showgrid": False, "zeroline": False},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def main():
    st.title("Scout Sheet")
    st.caption("Utility Map · Opponent scouting from local GRID-derived Parquet caches")
    try:
        tables = load_tables()
    except FileNotFoundError as error:
        st.error(f"Cache file missing: {error.filename or error}. Build it with `uv run build_cache.py` first.")
        st.stop()

    rounds, casts, kills = (tables[name] for name in TABLE_NAMES)
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
    maps = sorted(team_rounds["map"].dropna().astype(str).unique(), key=str.casefold)
    if not maps:
        st.warning("No rounds are available for this team.")
        st.stop()
    map_name = st.sidebar.selectbox("Map", maps)
    selected_side = st.sidebar.selectbox("Side", ("attack", "defense"))
    side = "attacker" if selected_side == "attack" else "defender"

    category_options = ["all", *CATEGORIES]
    category = st.sidebar.selectbox(
        "Ability category", category_options, index=category_options.index("placeable"),
        format_func=lambda value: "All categories" if value == "all" else value.title(),
    )
    time_window = st.sidebar.slider(
        "Round time window (seconds)", min_value=0, max_value=100, value=(0, 100), step=5,
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
    show_dots = st.sidebar.toggle("Show individual cast dots", value=False)

    filtered_casts = filter_casts(
        matched, category=category, player_id=player_id, agent=agent, seconds_range=time_window,
    )
    zones = aggregate_zone_cells(filtered_casts, denominator)
    shapes = load_shapes(casts, kills)
    shape = shapes.get(map_name)

    series_count = int(denominator["series_id"].nunique())
    game_count = int(denominator[["series_id", "game_id"]].drop_duplicates().shape[0])
    round_count = int(denominator[["series_id", "game_id", "round_number"]].drop_duplicates().shape[0])
    filtered_count = len(filtered_casts)
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
        "player, agent, category and time filters affect casts, not that denominator. "
        "Round time is measured from GRID's recorded round start (including freeze time)."
    )

    map_column, zones_column = st.columns([1.7, 1])
    with map_column:
        st.subheader(f"{map_name.title()} · {selected_side.title()}")
        if shape is None:
            st.warning("No cached positions are available for this map.")
        else:
            st.plotly_chart(_map_figure(shape, filtered_casts, zones, show_dots), width="stretch")
            st.caption(
                "Grey background: all-team cast and kill position density from the cache, not map artwork. "
                "Ability positions mark caster locations; landing points are not provided."
            )

    with zones_column:
        st.subheader("Hot zones")
        st.caption("Cells are approximately 4 m × 4 m; ranking is by distinct rounds used.")
        if zones.empty:
            st.info("No located casts match these filters.")
        else:
            for index, zone in zones.head(5).iterrows():
                st.markdown(f"#### {zones.index.get_loc(index) + 1}. {zone['ability_name']}")
                st.write(f"**{zone['agent'].title()}** · {zone['player_name']}")
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
        st.caption("Shared ability IDs are flagged because the same GRID ability ID appeared under multiple agents.")


if __name__ == "__main__":
    main()