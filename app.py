import hashlib
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.abilities import CATEGORIES, ability_mapping
from analysis.catalog import CIRCUITS, load_series_event_map, team_circuit_labels
from analysis.counter_plan import generate_counter_plan
from analysis.landing import build_walkable_mask
from analysis.map_shape import density_for_points, load_cached_map_shapes
from analysis.operator import (
    aggressiveness_label,
    aggressiveness_score,
    calculate_buy_rates,
    calculate_handoff_rate,
    estimate_spawn_anchors,
    load_spawn_estimates,
    operator_engagements,
)
from analysis.report import generate_scouting_report
from analysis.sites import disambiguate_zone_labels, label_zones, load_cached_site_centres, site_label
from analysis.text import team_possessive
from analysis.zones import aggregate_zone_cells, eligible_rounds, filter_casts, match_casts_to_rounds, merge_zone_cells
from config import DATA_DIR


PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
TABLE_NAMES = ("rounds", "casts", "kills", "damage", "buys", "spawn_positions", "plants")
SENTINEL_AGENTS = {"chamber", "cypher", "deadlock", "killjoy", "sage", "vyse"}
ROUND_INDEX_COLUMNS = ("series_id", "team_id", "team_name", "map", "game_id", "round_number")
TABLE_READ_COLUMNS = {
    "rounds": ("series_id", "game_id", "game_number", "map", "round_number", "round_start_time", "round_live_time", "team_id", "team_name", "opponent_name", "side", "won_round"),
    "casts": ("series_id", "game_id", "map", "round_number", "seconds_into_round", "player_id", "player_name", "team_id", "team_name", "side", "agent", "ability", "slot", "x", "y", "z", "view_x", "view_y", "view_z", "quality", "landing_x", "landing_y", "landing_method", "estimated"),
    "kills": ("series_id", "game_id", "map", "round_number", "seconds_into_round", "event_time", "killer_id", "killer_name", "killer_team_id", "killer_team_name", "killer_side", "killer_agent", "victim_id", "victim_name", "victim_team_id", "victim_team_name", "victim_side", "victim_agent", "weapon", "killer_x", "killer_y", "victim_x", "victim_y"),
    "damage": ("series_id", "game_id", "map", "round_number", "seconds_into_round", "event_time", "player_id", "player_name", "team_id", "team_name", "side", "agent", "victim_id", "victim_name", "victim_team_id", "victim_team_name", "victim_side", "weapon", "damage_amount", "x", "y", "victim_x", "victim_y"),
    "buys": ("series_id", "game_id", "map", "round_number", "player_id", "player_name", "team_id", "side", "agent", "primary_weapon", "has_operator"),
    "spawn_positions": ("series_id", "game_id", "map", "round_number", "player_id", "player_name", "team_id", "team_name", "side", "agent", "x", "y"),
    "plants": ("series_id", "game_id", "map", "round_number", "planter_id", "team_id", "team_name", "x", "y"),
}


st.set_page_config(page_title="Scout Sheet", layout="wide")


@st.cache_data(show_spinner=False)
def load_round_index(signature):
    return pd.read_parquet(CACHE_DIR / "rounds.parquet", columns=list(ROUND_INDEX_COLUMNS))


@st.cache_data(show_spinner=False)
def load_tables(signature, team_id, map_name):
    """Read projected columns for the selected team and map from Parquet."""
    loaded = {}
    for name in TABLE_NAMES:
        path = CACHE_DIR / f"{name}.parquet"
        columns = list(TABLE_READ_COLUMNS[name])
        if name == "plants":
            loaded[name] = pd.DataFrame(columns=columns)
            continue
        if name in {"rounds", "casts", "damage", "buys", "spawn_positions"}:
            filters = [("team_id", "==", str(team_id))]
        elif name == "kills":
            filters = [
                [("killer_team_id", "==", str(team_id))],
                [("victim_team_id", "==", str(team_id))],
            ]
        else:
            filters = None
        if name == "casts" and str(map_name).casefold() != "all maps":
            filters = [*filters, ("map", "==", str(map_name))]
        loaded[name] = pd.read_parquet(path, columns=columns, filters=filters)
    return loaded


@st.cache_data(show_spinner="Building map position densities…")
def load_shapes(signature):
    return load_cached_map_shapes(CACHE_DIR)


@st.cache_data(show_spinner=False)
def load_site_shapes(signature):
    return load_cached_site_centres(CACHE_DIR)


@st.cache_data(show_spinner=False)
def load_global_spawn_estimates(signature):
    return load_spawn_estimates(CACHE_DIR / "spawn_estimates.json")


@st.cache_data(show_spinner=False)
def load_event_catalog(signature, series_csv):
    return load_series_event_map(series_csv)


@st.cache_data(show_spinner=False)
def _cached_scouting_report(
    cache_signature, generated_date, team_name, team_id, map_name, selected_side, side,
    _tables, _shapes, _site_centres, _spawn_estimates,
):
    return generate_scouting_report(
        _tables, team_name, team_id, map_name, selected_side, side, _shapes, _site_centres,
        spawn_estimates=_spawn_estimates, generated_on=date.fromisoformat(generated_date),
    )


def _scouting_report_download_data(*report_args):
    def prepare_report():
        return _cached_scouting_report(*report_args)

    return prepare_report


def _download_report(report_data, tab_key):
    st.download_button(
        "Download scouting report", report_data,
        file_name="scout-sheet-scouting-report.html", mime="text/html",
        key=f"download_scouting_report_{tab_key}",
    )


def _render_app_style():
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        h1 { font-size: clamp(2rem, 2.4vw, 2.8rem) !important; }
        h2, h3 { letter-spacing: -0.015em; }
        [data-testid="stMetricValue"] {
            font-size: clamp(1.35rem, 1.7vw, 2rem);
            line-height: 1.15;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(148, 163, 184, 0.35);
            border-radius: 0.85rem;
        }
        [data-testid="stPlotlyChart"] { width: 100%; }
        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_welcome_panel():
    if "welcome_panel" not in st.session_state:
        st.session_state["welcome_panel"] = True
    with st.expander(
        "Welcome to Scout Sheet", expanded=True, key="welcome_panel", on_change="rerun",
    ):
        st.write(
            "Scout Sheet helps coaches find opponent utility patterns, Operator habits, and repeatable tendencies."
        )
        st.markdown(
            "**Utility Map** — Locate recurring utility zones and round-by-round evidence.  \n"
            "**Operator Profiler** — Review Operator buys, holders, engagements, and angles.  \n"
            "**Counter Plan** — Turn measured patterns into evidence-backed responses."
        )
        st.caption("Pick a circuit and team on the left to start.")


def _missing_prerequisites(data_dir=DATA_DIR, cache_dir=CACHE_DIR):
    missing_cache = [name for name in TABLE_NAMES if not (Path(cache_dir) / f"{name}.parquet").is_file()]
    missing_cache.extend(
        name for name in ("map_shapes.npz", "site_centres.json")
        if not (Path(cache_dir) / name).is_file()
    )
    return not Path(data_dir).is_dir(), missing_cache


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


def _separate_zone_markers(
    zones, shape=None, site_centres=None, marker_gap_pixels=54, plot_pixels=600,
):
    """Separate zone markers from each other and the site-letter markers."""
    positions = []
    angles = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    if shape:
        map_span = max(
            float(shape["x_max"]) - float(shape["x_min"]),
            float(shape["y_max"]) - float(shape["y_min"]),
        )
    else:
        map_span = 14000.0
    min_distance = map_span * marker_gap_pixels / plot_pixels
    site_distance = map_span * (marker_gap_pixels + 18) / plot_pixels
    x_bounds = _padded_axis_range(shape, "x") if shape else [-math.inf, math.inf]
    y_bounds = _padded_axis_range(shape, "y") if shape else [-math.inf, math.inf]
    site_points = [
        (float(site["x"]), float(site["y"]))
        for site in (site_centres or {}).values()
    ]

    def is_available(candidate):
        return (
            x_bounds[0] <= candidate[0] <= x_bounds[1]
            and y_bounds[0] <= candidate[1] <= y_bounds[1]
            and all(math.dist(candidate, previous) >= min_distance for previous in positions)
            and all(math.dist(candidate, site) >= site_distance for site in site_points)
        )

    for row in zones.itertuples(index=False):
        original = (float(row.x), float(row.y))
        location = original
        collides = (
            any(math.dist(location, previous) < min_distance for previous in positions)
            or any(math.dist(location, site) < site_distance for site in site_points)
            or not is_available(location)
        )
        if collides:
            for ring in range(1, 18):
                radius = min_distance * (1 + ring * 0.55)
                candidates = [
                    (original[0] + radius * math.cos(angle), original[1] + radius * math.sin(angle))
                    for angle in angles
                ]
                available = [candidate for candidate in candidates if is_available(candidate)]
                if available:
                    location = min(available, key=lambda point: math.dist(point, original))
                    break
            else:
                candidates = [
                    (min(max(original[0], x_bounds[0]), x_bounds[1]), min(max(original[1], y_bounds[0]), y_bounds[1])),
                    (x_bounds[0], y_bounds[0]), (x_bounds[0], y_bounds[1]),
                    (x_bounds[1], y_bounds[0]), (x_bounds[1], y_bounds[1]),
                ]
                location = max(
                    candidates,
                    key=lambda point: min(
                        [math.dist(point, previous) for previous in positions]
                        + [math.dist(point, site) for site in site_points]
                        + [min_distance]
                    ),
                )
        positions.append(location)
    return [point[0] for point in positions], [point[1] for point in positions]


def _casts_for_arrow_zone(casts, zones, selected_zone=None, sample_size=80, random_state=42):
    """Select landing estimates for one 4 m zone or a bounded fallback sample."""
    if not {"estimated", "caster_x", "caster_y", "x", "y"}.issubset(casts.columns):
        return casts.iloc[0:0].copy()
    candidates = casts[
        casts["estimated"].fillna(False).astype(bool)
        & casts["caster_x"].notna() & casts["caster_y"].notna()
        & casts["x"].notna() & casts["y"].notna()
        & ((casts["x"] != casts["caster_x"]) | (casts["y"] != casts["caster_y"]))
    ].copy()
    if selected_zone is not None and 1 <= int(selected_zone) <= len(zones):
        zone = zones.iloc[int(selected_zone) - 1]
        cell_x = (pd.to_numeric(candidates["x"], errors="coerce") / 400).map(math.floor)
        cell_y = (pd.to_numeric(candidates["y"], errors="coerce") / 400).map(math.floor)
        cells = zone.get("cells")
        if not isinstance(cells, (tuple, list)) or not cells:
            cells = ((int(zone["cell_x"]), int(zone["cell_y"])),)
        selected = pd.Series(False, index=candidates.index)
        for selected_x, selected_y in cells:
            selected |= cell_x.eq(int(selected_x)) & cell_y.eq(int(selected_y))
        if "ability" in candidates and pd.notna(zone.get("ability")):
            selected &= candidates["ability"].astype(str).eq(str(zone["ability"]))
        return candidates[selected]
    return candidates.sample(n=min(int(sample_size), len(candidates)), random_state=random_state)


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


def _map_for_team(map_name, map_options, map_counts, default_map, team_changed):
    if map_name not in map_options:
        return default_map
    if team_changed and map_name != "All maps" and int(map_counts.get(map_name, 0)) < 20:
        return default_map
    return map_name


def _navigate_counter_card(target_tab, map_name, selected_side, team_id, category=None, agent=None):
    st.session_state["scout_tabs"] = target_tab
    st.session_state["shared_side"] = selected_side
    tab_key = "operator" if target_tab == "Operator Profiler" else "utility"
    map_widget_key = f"{tab_key}_map_widget_{team_id}"
    persistent_map_key = f"{tab_key}_map_selection_{team_id}"
    st.session_state[map_widget_key] = map_name
    st.session_state[persistent_map_key] = map_name
    if target_tab == "Utility Map":
        st.session_state["ability_category"] = category or "all"
        st.session_state[f"utility_agent_widget_{team_id}"] = agent or "all"
        st.session_state[f"utility_position_{team_id}_{category or 'all'}"] = "Caster"


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


def _team_labels(rounds, event_by_series=None):
    return team_circuit_labels(rounds, event_by_series or {})


def _map_figure(
    shape, casts, zones, show_dots, map_name, site_centres, position_mode="Caster", arrow_zone=1,
):
    figure = go.Figure()
    walkable_mask = build_walkable_mask(shape)
    figure.add_trace(go.Heatmap(
        x=shape["x"], y=shape["y"], z=shape["density"],
        colorscale=[[0, "rgba(0,0,0,0)"], [0.2, "#343b46"], [1, "#a4abb5"]],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip", name="All-team position density",
    ))

    located = casts.dropna(subset=["x", "y"])
    if not located.empty:
        points = located[["x", "y"]].to_numpy(dtype=float)
        cast_density = density_for_points(points, shape, sigma=12)
        cast_density[~walkable_mask] = np.nan
        finite_density = cast_density[np.isfinite(cast_density)]
        if finite_density.size and finite_density.max() > 0:
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
            if position_mode == "Estimated landing" and {"caster_x", "caster_y", "estimated"}.issubset(located.columns):
                estimated = _casts_for_arrow_zone(located, zones, arrow_zone)
                arrow_x, arrow_y, arrow_head_angles = [], [], []
                for cast in estimated.itertuples(index=False):
                    caster_x, caster_y = float(cast.caster_x), float(cast.caster_y)
                    landing_x, landing_y = float(cast.x), float(cast.y)
                    arrow_x.extend((caster_x, landing_x, None))
                    arrow_y.extend((caster_y, landing_y, None))
                    arrow_head_angles.append(
                        math.degrees(math.atan2(landing_y - caster_y, landing_x - caster_x)) + 90
                    )
                if arrow_head_angles:
                    figure.add_trace(go.Scatter(
                        x=arrow_x, y=arrow_y, mode="lines",
                        line={"width": 0.7, "color": "rgba(248,250,252,0.3)"},
                        hoverinfo="skip", showlegend=False, name="Caster-to-estimate arrows",
                    ))
                    figure.add_trace(go.Scatter(
                        x=estimated["x"], y=estimated["y"], mode="markers",
                        marker={"size": 5, "color": "rgba(248,250,252,0.42)",
                                "symbol": "triangle-up", "angle": arrow_head_angles},
                        hoverinfo="skip", showlegend=False, name="Estimated landing arrowheads",
                    ))
                if not estimated.empty:
                    figure.add_trace(go.Scattergl(
                        x=estimated["x"], y=estimated["y"], mode="markers",
                        marker={"size": 3, "color": "#f8fafc", "opacity": 0.72, "line": {"width": 0}},
                        hoverinfo="skip", name="Estimated landing points",
                    ))
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
        marker_x, marker_y = _separate_zone_markers(top_zones, shape, map_sites)
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
               "showticklabels": False, "ticks": "", "constrain": "domain"},
        yaxis={"title": None, "range": y_range, "showgrid": False, "zeroline": False,
               "showticklabels": False, "ticks": "", "constrain": "domain"},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    return figure


def _casts_for_position(casts, position_mode):
    """Return cast rows with x/y set to the requested display and zone position."""
    selected = casts.copy()
    if position_mode != "Estimated landing":
        return selected
    selected["caster_x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["caster_y"] = pd.to_numeric(selected["y"], errors="coerce")
    landing_x = pd.to_numeric(selected["landing_x"], errors="coerce") if "landing_x" in selected else selected["caster_x"]
    landing_y = pd.to_numeric(selected["landing_y"], errors="coerce") if "landing_y" in selected else selected["caster_y"]
    selected["x"] = landing_x.fillna(selected["caster_x"])
    selected["y"] = landing_y.fillna(selected["caster_y"])
    return selected


def _operator_strip_figure(rounds, player_shares, show_all=False):
    ordered = rounds.copy()
    figure = go.Figure()
    if ordered.empty:
        figure.update_layout(template="plotly_dark", height=150)
        return figure
    ordered["_round_order"] = pd.to_numeric(ordered["round_number"], errors="coerce")
    if "round_start_time" not in ordered:
        ordered["round_start_time"] = None
    ordered["_date_order"] = pd.to_datetime(ordered["round_start_time"], utc=True, errors="coerce")
    for column in ("game_number", "opponent_name", "map"):
        if column not in ordered:
            ordered[column] = None
    ordered["_game_order"] = pd.to_numeric(ordered["game_number"], errors="coerce")
    palette = ["#38bdf8", "#f97316", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#fb7185"]
    player_info = {}
    for row in player_shares.drop_duplicates("player_id").itertuples(index=False):
        player_info[str(row.player_id)] = {
            "name": row.player_name,
            "color": palette[len(player_info) % len(palette)],
        }
    game_rows = []
    for (series_id, game_id), game in ordered.groupby(["series_id", "game_id"], sort=False):
        first = game.iloc[0]
        game_number = first.get("game_number")
        opponent = first.get("opponent_name")
        map_name = first.get("map")
        game_date = game["_date_order"].dropna().max()
        game_rows.append({
            "series_id": str(series_id), "game_id": str(game_id),
            "game_number": str(game_number) if pd.notna(game_number) and str(game_number).strip() else "?",
            "opponent": str(opponent) if pd.notna(opponent) and str(opponent).strip() else "Unknown opponent",
            "map": str(map_name).title() if pd.notna(map_name) and str(map_name).strip() else "Unknown map",
            "date": game_date,
            "game_order": pd.to_numeric(game_number, errors="coerce"),
        })
    game_rows.sort(
        key=lambda item: (
            pd.isna(item["date"]),
            -item["date"].value if pd.notna(item["date"]) else 0,
            -(item["game_order"] if pd.notna(item["game_order"]) else -1),
            item["series_id"], item["game_id"],
        ),
    )
    if not show_all:
        game_rows = game_rows[:15]
    visible_games = {(row["series_id"], row["game_id"]) for row in game_rows}
    ordered = ordered[
        ordered.apply(lambda row: (str(row["series_id"]), str(row["game_id"])) in visible_games, axis=1)
    ].copy()
    game_labels = {}
    row_positions = {}
    game_ticks = []
    for game_index, game_row in enumerate(game_rows):
        date = game_row["date"].strftime("%Y-%m-%d") if pd.notna(game_row["date"]) else "Date unknown"
        label = f"{date} · vs {game_row['opponent']} · {game_row['map']} · G{game_row['game_number']}"
        game_key = (game_row["series_id"], game_row["game_id"])
        game_labels[game_key] = label
        y = float(len(game_rows) - game_index - 1) * 1.3
        game_ticks.append((y, label))
        game = ordered[
            ordered["series_id"].astype(str).eq(game_row["series_id"])
            & ordered["game_id"].astype(str).eq(game_row["game_id"])
        ]
        for row in game.itertuples(index=False):
            row_positions[(str(row.series_id), str(row.game_id), str(row.round_number))] = y

    points = {"No Operator": [], "Snapshot unavailable": [], "Multiple holders": []}

    for row in ordered.itertuples(index=False):
        holders = tuple(row.holders)
        names = tuple(row.holder_names)
        holder_label = ", ".join(names) if names else ("No Operator recorded" if not holders else "Holder name unavailable")
        game_label = game_labels[(str(row.series_id), str(row.game_id))]
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
    chart_height = max(150, 34 * len(game_rows) + 36)
    figure.update_layout(
        template="plotly_dark", height=chart_height, margin={"l": 8, "r": 8, "t": 4, "b": 24},
        plot_bgcolor="#111827", paper_bgcolor="#111827", showlegend=True,
        legend={"orientation": "h", "y": 1.0, "yanchor": "top", "x": 0},
        xaxis={"title": "Round number", "range": [0.5, max_round + 0.5], "dtick": 1,
               "showgrid": False, "zeroline": False},
        yaxis={"tickmode": "array", "tickvals": [tick for tick, _ in game_ticks],
               "ticktext": [label for _, label in game_ticks],
               "range": [-0.5, max(0.5, (len(game_rows) - 1) * 1.3 + 0.5)],
               "showgrid": False, "zeroline": False},
    )
    return figure


def _operator_headline(
    team_name, primary, total_rounds, engagements, map_name, selected_side,
    operator_rounds=None, side_buy_rates=None,
):
    facts = []
    if primary is not None:
        player_name = primary["player_name"]
        held_rounds = int(primary["rounds"])
        operator_rounds = int(operator_rounds or primary.get("operator_rounds", 0))
        if operator_rounds:
            share = float(primary.get("share", 100 * held_rounds / operator_rounds))
            facts.append(
                f"{player_name} holds {team_possessive(team_name)} Operator in {share:.0f}% of their "
                f"{selected_side} Operator rounds ({held_rounds:,}/{operator_rounds:,})."
            )
    side_buy_rates = side_buy_rates or {}
    attack = side_buy_rates.get("attacker", {})
    defense = side_buy_rates.get("defender", {})
    if attack.get("observed_rounds", 0) and defense.get("observed_rounds", 0):
        attack_text = f"{attack['rate']:.0f}% ({int(attack['held_rounds']):,}/{int(attack['observed_rounds']):,})"
        defense_text = f"{defense['rate']:.0f}% ({int(defense['held_rounds']):,}/{int(defense['observed_rounds']):,})"
        if defense["rate"] > attack["rate"]:
            facts.append(
                f"{team_name} buy the Operator in {defense_text} of defense rounds but only "
                f"{attack_text} on attack."
            )
        elif attack["rate"] > defense["rate"]:
            facts.append(
                f"{team_name} buy the Operator in {attack_text} of attack rounds but only "
                f"{defense_text} on defense."
            )
        else:
            facts.append(f"{team_name} buy the Operator in {attack_text} of attack and defense rounds.")

    if primary is not None and selected_side == "defense":
        if len(engagements) < 10:
            suggestion = "; try All maps" if map_name != "All maps" else ""
            facts.append(f"Too few player engagements to profile angles here{suggestion}.")
        else:
            timed = pd.to_numeric(engagements["seconds_into_round"], errors="coerce").dropna()
            if len(timed) >= 10:
                early = int(timed.between(0, 20, inclusive="left").sum())
                facts.append(
                    f"{100 * early / len(timed):.0f}% of {primary['player_name']}'s defense Operator engagements "
                    f"happen in 0–20 s ({early:,}/{len(timed):,})."
                )
            known = engagements.loc[
                ~engagements["aggressiveness"].eq("unknown"), "aggressiveness",
            ].value_counts()
            if int(known.sum()) >= 10 and known.iloc[0] / known.sum() >= 0.5:
                angle_style = {
                    "aggressive": "holds aggressive angles",
                    "passive": "holds passive angles",
                    "standard": "takes standard angles",
                }.get(known.index[0], "engages")
                site = _most_common(engagements["site_label"]) or "an unknown area"
                facts.append(f"{primary['player_name']} {angle_style} at {site}.")
    return " ".join(facts) if facts else None


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


def _profile_engagements(
    kills, damage, buys, spawn_positions, team_id, map_name, site_centres, spawn_estimates=None,
):
    engagements = operator_engagements(kills, damage, buys, team_id, map_name, "defender")
    if engagements.empty:
        return engagements, pd.DataFrame(columns=["player_name", "engagements", "aggressive", "standard", "passive", "unknown", "aggressive_share"])
    anchors = estimate_spawn_anchors(spawn_positions)
    full_anchors = spawn_estimates or {}
    engagements["seconds_into_round"] = pd.to_numeric(engagements["seconds_into_round"], errors="coerce")
    engagements["time_bucket"] = engagements["seconds_into_round"].map(_time_bucket)
    aggression_scores = []
    site_labels = []
    for row in engagements.itertuples(index=False):
        map_key = str(row.map).casefold()
        map_anchors = full_anchors.get(map_key, anchors.get(map_key, {}))
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


def _render_utility_tab(tables, team_name, team_id, map_name, selected_side, side, shapes, site_centres, report_data):
    _download_report(report_data, "utility")
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
    agent_widget_key = f"utility_agent_widget_{team_id}"
    if st.session_state.get(agent_widget_key, "all") not in ["all", *agent_options]:
        st.session_state[agent_widget_key] = "all"
    agent = st.sidebar.selectbox(
        "Agent (optional)", ["all", *agent_options],
        format_func=lambda value: "All agents" if value == "all" else value.title(),
        key=agent_widget_key,
    )
    time_window = st.sidebar.slider(
        "Live round time window (seconds)", min_value=0, max_value=100, value=(0, 100), step=5,
    )
    include_buy_phase = st.sidebar.toggle(
        "Include buy phase", value=True,
        help="Include casts with negative times, before the round-ended-freezetime event.",
        key=f"include_buy_phase_{team_id}",
    )
    include_remote = st.sidebar.toggle(
        "Include remote casts", value=False,
        help="Remote abilities are excluded by default because caster position is not the utility location.",
        key=f"include_remote_casts_{team_id}",
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
    show_dots = st.sidebar.toggle(
        "Show individual cast dots", value=False, key=f"show_cast_dots_{team_id}",
    )

    filtered_casts = (
        category_base_casts if category == "all"
        else category_base_casts[category_base_casts["category"].eq(category)].copy()
    )
    position_key = f"utility_position_{team_id}_{category}"
    has_estimates = filtered_casts.get(
        "estimated", pd.Series(False, index=filtered_casts.index),
    ).fillna(False).astype(bool).any()
    if position_key not in st.session_state:
        st.session_state[position_key] = "Estimated landing" if category == "placeable" or has_estimates else "Caster"
    position_mode = st.radio(
        "Position", ("Caster", "Estimated landing"), horizontal=True, key=position_key,
        help="Estimated landing projects aim direction to a wall or range limit using the density silhouette.",
    )
    position_casts = _casts_for_position(filtered_casts, position_mode)
    if position_mode == "Estimated landing":
        st.caption("Estimated landing points use aim direction and the walkability mask; they are not exact utility positions.")
    cell_zones = label_zones(
        aggregate_zone_cells(position_casts, denominator), map_name, site_centres,
    )
    zones = disambiguate_zone_labels(
        merge_zone_cells(cell_zones), map_name, site_centres,
    )
    shape = shapes.get(map_name)
    arrow_zone = 1 if not zones.empty else None
    if show_dots and position_mode == "Estimated landing":
        arrow_options = [*range(1, min(5, len(zones)) + 1), None]
        arrow_zone = st.selectbox(
            "Arrows for hot zone", arrow_options,
            format_func=lambda value: (
                "No selected zone · random sample (max 80)" if value is None
                else f"Zone {value} · {zones.iloc[value - 1]['site_label']}"
            ),
            key=f"arrow_zone_{key_suffix}_{category}_{position_mode}",
            help="Estimated-landing arrows are limited to this zone; choose no zone to see a random sample of at most 80.",
        )
    series_count = int(denominator["series_id"].nunique())
    game_count = int(denominator[["series_id", "game_id"]].drop_duplicates().shape[0])
    round_count = int(denominator[["series_id", "game_id", "round_number"]].drop_duplicates().shape[0])
    if not zones.empty:
        strongest = zones.iloc[0]
        label = strongest["site_label"]
        location = label if label.startswith("near ") or label == "Site unknown" else f"at {label}"
        with st.container(border=True):
            st.markdown(
                f"**{team_name} use utility {location} in {strongest['percentage']:.0f}% of {selected_side} rounds "
                f"on {map_name.title()} ({int(strongest['rounds_used']):,}/{int(strongest['total_rounds']):,}, "
                f"{int(strongest['series_count']):,} series).** Most common: "
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
        st.warning(f"Small sample: only {round_count:,} {selected_side} rounds match this team/map/side selection.")
    if unmatched:
        st.info(f"Excluded {unmatched:,} cast rows that did not match a team/map/side round record.")
    with st.expander("How this is calculated"):
        st.markdown(
            "Zone evidence is per distinct round. The denominator is all rounds for this team, map and side; "
            "player, agent, category, remote-cast and time filters affect casts, not that denominator. "
            "Round time starts at `round-ended-freezetime`; negative seconds are buy-phase casts. Site labels "
            "are approximate, based on plant positions clustered around named site callouts. The grey background "
            "uses all-team cast and kill positions from the local cache rather than Riot map artwork. Estimated "
            "landing points project the caster's aim direction to a density-derived wall or range limit; they "
            "are estimates, not exact utility landing positions or verified wall geometry. Operator aggression "
            "uses the saved full-data `cache/spawn_estimates.json` anchors when that map is present, and falls "
            "back to this team's freeze-end median positions only for maps without a saved estimate."
        )
    map_column, zones_column = st.columns([1.7, 1])
    with map_column:
        st.subheader(f"{map_name.title()} · {selected_side.title()}")
        if shape is None:
            st.warning("No cached positions are available for this map.")
        else:
            st.plotly_chart(
                _map_figure(
                    shape, position_casts, zones, show_dots, map_name, site_centres,
                    position_mode, arrow_zone,
                ),
                width="stretch",
            )
        if not filtered_casts.empty:
            missing_positions = int(filtered_casts[["x", "y"]].isna().any(axis=1).sum())
            if missing_positions:
                st.caption(f"{missing_positions:,} matching casts have no usable position and are omitted from zones.")
    with zones_column:
        st.subheader("Hot zones")
        st.caption("Neighbouring 4 m cells within 6 m are merged by ability and site; ranking counts distinct rounds.")
        if zones.empty:
            if filtered_casts.empty:
                st.caption("No hot zones to show for this empty result.")
            else:
                st.info("Matching casts are present, but none have usable positions for a hot zone.")
        else:
            for index, zone in enumerate(zones.head(5).itertuples(index=False), start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. {zone.site_label} · {zone.ability_name}**")
                    st.write(f"Most common cast: **{zone.agent.title()}** · {zone.player_name}")
                    st.caption(
                        f"Used in {zone.rounds_used:,} of {zone.total_rounds:,} {selected_side} rounds "
                        f"({zone.percentage:.0f}%) on {map_name.title()} across {zone.series_count:,} series."
                    )
    with st.expander("Observed agent / ability mapping"):
        st.dataframe(ability_mapping(casts), width="stretch", hide_index=True)
        st.caption(
            "Shared ability IDs are flagged because the same GRID ability ID appeared under multiple agents. "
            "Placeable is restricted to local Cypher, Killjoy, Chamber, Vyse and Deadlock utility; remote casts "
            "are identified and excluded by default."
        )


def _render_operator_tab(
    tables, team_name, team_id, map_name, selected_side, shapes, site_centres, report_data,
    spawn_estimates=None,
):
    _download_report(report_data, "operator")
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
        team_id, map_name, site_centres, spawn_estimates,
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
        operator_rounds=selected_rates["overall"]["held_rounds"],
        side_buy_rates=rates["by_side"],
    )
    st.markdown(f"**{team_name} Operator profile · {display_map}**")
    if headline:
        with st.container(border=True):
            st.markdown(f"**{headline}**")
    metrics = st.columns(4)
    metrics[0].metric("Series" if map_name == "All maps" else "Map series", f"{series_count:,}")
    metrics[1].metric("Team rounds", f"{eligible_count:,}")
    metrics[2].metric("Buy snapshots", f"{observed_count:,}")
    metrics[3].metric("Operator rounds", f"{overall['held_rounds']:,}")
    st.caption(f"Buy snapshot coverage: {observed_count:,} of {eligible_count:,} team rounds.")
    if observed_count < 20:
        st.warning(f"Small sample: only {observed_count:,} rounds have a recorded freeze-time buy snapshot.")
    if selected_eligible_count < 20:
        st.warning(f"Small sample: only {selected_eligible_count:,} team rounds are in the selected {selected_side} filter.")

    rate_columns = st.columns(3)
    for column, label, summary in (
        (rate_columns[0], "Overall buy rate", overall),
        (rate_columns[1], "Attack buy rate", rates["by_side"].get("attacker", {"held_rounds": 0, "observed_rounds": 0, "eligible_rounds": 0, "rate": 0})),
        (rate_columns[2], "Defense buy rate", rates["by_side"].get("defender", {"held_rounds": 0, "observed_rounds": 0, "eligible_rounds": 0, "rate": 0})),
    ):
        column.metric(
            label,
            f"{int(summary['held_rounds']):,} / {int(summary['observed_rounds']):,} ({summary['rate']:.0f}%)",
        )
        column.caption(
            f"Buy snapshots: {int(summary['observed_rounds']):,} / {int(summary['eligible_rounds']):,} rounds",
        )

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
            column_config={
                "Rounds": st.column_config.NumberColumn("Rounds", format="localized"),
                "Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100),
            },
        )
    with agent_column:
        st.markdown(f"**Per agent · {selected_side} Operator rounds**")
        st.dataframe(
            agent_shares, width="stretch", hide_index=True,
            column_config={
                "Rounds": st.column_config.NumberColumn("Rounds", format="localized"),
                "Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100),
            },
        )

    st.subheader(f"Possession · {selected_side.title()} rounds")
    show_all_games = st.toggle("Show all games", value=False, key="operator_show_all_games")
    if not show_all_games and strip_rounds[["series_id", "game_id"]].drop_duplicates().shape[0] > 15:
        st.caption("Showing the 15 most recent games.")
    st.plotly_chart(
        _operator_strip_figure(strip_rounds, selected_rates["players"], show_all=show_all_games),
        width="stretch",
    )
    st.metric(
        "Handoff rate between consecutive Operator rounds",
        f"{handoff['handoffs']:,} / {handoff['comparisons']:,} ({handoff['rate']:.0f}%)",
    )
    st.caption("A handoff is a different holder set between successive Operator rounds; comparisons reset each game and across missing snapshots.")

    st.subheader("Defense setup positions and engagements")
    st.caption("Operator-holder kill/damage encounters only; times are measured from round-ended-freezetime.")
    if len(engagements) < 20:
        st.warning(f"Small sample: {len(engagements):,} deduplicated Operator engagements on defense.")
    if map_name == "All maps":
        st.info("Pick one map in the Map filter to view the engagement map; All maps is used for aggregate timing evidence only.")
    elif shapes.get(map_name) is None:
        st.warning("No cached positions are available for this map.")
    elif engagements.empty:
        st.info("No recorded defense Operator engagements match this team and map.")
    else:
        st.plotly_chart(_operator_map_figure(shapes[map_name], engagements, map_name, site_centres), width="stretch")

    st.subheader("Aggressive vs passive angles")
    st.caption(
        "Aggression uses full-data median freeze-end anchors from cache/spawn_estimates.json when available; "
        "maps without a saved estimate fall back to this team's medians. It is a rough directional proxy."
    )
    profiled_players = player_aggression[player_aggression["engagements"].ge(10)]
    if profiled_players.empty:
        st.info("No player has 10 or more defense engagements in this filter; per-player angle profiles are withheld.")
    else:
        display_aggression = profiled_players.drop(columns=["player_id"], errors="ignore")
        st.dataframe(
            display_aggression, width="stretch", hide_index=True,
            column_config={
                "engagements": st.column_config.NumberColumn("Engagements", format="localized"),
                "aggressive": st.column_config.NumberColumn("Aggressive", format="localized"),
                "standard": st.column_config.NumberColumn("Standard", format="localized"),
                "passive": st.column_config.NumberColumn("Passive", format="localized"),
                "unknown": st.column_config.NumberColumn("Unknown", format="localized"),
                "aggressive_share": st.column_config.ProgressColumn(
                    "Aggressive share", format="%.0f%%", min_value=0, max_value=100,
                ),
            },
        )
    st.caption(f"Engagement sample: {len(engagements):,} defense engagements; time buckets omit unknown times.")
    st.caption(
        "Possession is sampled at round-ended-freezetime and does not track later pickups or drops. "
        "Rapid damage ticks against one player are grouped with a resulting kill when within 1.5 seconds."
    )


def _render_counter_plan_tab(
    tables, team_name, team_id, map_name, selected_side, side, site_centres, report_data,
    spawn_estimates=None,
):
    _download_report(report_data, "counter")
    denominator = eligible_rounds(tables["rounds"], team_id, map_name, side)
    round_count = denominator[["series_id", "game_id", "round_number"]].drop_duplicates().shape[0]
    series_count = int(denominator["series_id"].nunique())
    title = "Self-scout: predictable habits" if team_name in {"Cloud9", "Cloud9 GC"} else "Counter Plan"
    st.subheader(f"{title} · {map_name.title()} · {selected_side.title()}")
    evidence = st.columns(2)
    evidence[0].metric("Series analysed", f"{series_count:,}")
    evidence[1].metric("Rounds analysed", f"{round_count:,}")
    if round_count < 20:
        st.warning(f"Small sample: only {round_count:,} {selected_side} rounds match this team/map/side selection.")
    st.caption(
        "Utility tendencies use this map; Operator-holder and buy-rate evidence pools all maps for the selected side. "
        "All card templates are generated locally."
    )
    plan = generate_counter_plan(
        tables["rounds"], tables["casts"], tables["buys"], tables["kills"], tables["damage"],
        tables["spawn_positions"], team_id, team_name, map_name, side, site_centres,
        self_scout=team_name in {"Cloud9", "Cloud9 GC"},
        spawn_estimates=spawn_estimates,
    )
    cards = plan["cards"]
    if not cards:
        st.info("No strong tendencies at this sample size.")
    for index, card in enumerate(cards):
        with st.container(border=True):
            st.markdown(f"**{card['tendency']}**")
            st.markdown("\n".join(f"- {fact}" for fact in card["facts"]))
            st.markdown(_counter_confidence_badge(card["confidence"]), unsafe_allow_html=True)
            st.info(card["response"])
            st.button(
                "Show on map",
                key=f"counter_show_{index}_{card['type'].lower().replace(' ', '_')}",
                on_click=_navigate_counter_card,
                args=(
                    card["navigation_tab"], card["map"], selected_side, team_id,
                    card.get("category"), card.get("agent"),
                ),
            )
    if plan["watch_list"]:
        st.markdown("### Watch list")
        for card in plan["watch_list"]:
            facts = " · ".join(card["facts"])
            st.markdown(f"- **{card['title']}** — {facts} · Low confidence")


def _counter_confidence_badge(confidence):
    styles = {
        "High": "background:#166534;color:#dcfce7",
        "Medium": "background:#854d0e;color:#fef9c3",
    }
    style = styles.get(confidence, "background:#334155;color:#e2e8f0")
    return (
        f"<span style='display:inline-block;padding:3px 9px;border-radius:999px;"
        f"font-weight:700;{style}'>{confidence} confidence</span>"
    )


def main():
    missing_data, missing_cache = _missing_prerequisites()
    if missing_data or missing_cache:
        st.title("Scout Sheet setup")
        st.info("Scout Sheet needs the local GRID starter data and generated cache before it can show scouting views.")
        st.markdown(
            "1. Install `uv` using the [official installer](https://docs.astral.sh/uv/getting-started/installation/).\n"
            "2. Clone Scout Sheet and the GRID starter repository into sibling folders.\n"
            "3. From the Scout Sheet folder, run `uv sync`.\n"
            "4. Run `uv run python build_cache.py` to create the ignored local Parquet cache.\n"
            "5. Start the app with `uv run streamlit run app.py` and open `http://localhost:8501`."
        )
        if missing_data:
            st.caption(f"GRID starter data folder not found at `{DATA_DIR}`.")
        if missing_cache:
            st.caption(f"Cache tables missing: {', '.join(missing_cache)}.")
        st.stop()
    try:
        signature = tuple((CACHE_DIR / f"{name}.parquet").stat().st_mtime_ns for name in TABLE_NAMES)
        round_index = load_round_index(signature)
        series_csv = DATA_DIR / "val" / "series.csv"
        event_signature = series_csv.stat().st_mtime_ns
        event_by_series = load_event_catalog(event_signature, str(series_csv))
    except FileNotFoundError as error:
        st.error(f"Cache or GRID index file missing: {error.filename or error}. Build it with `uv run python build_cache.py` first.")
        st.stop()

    teams = _team_labels(round_index, event_by_series)
    if teams.empty:
        st.error("The rounds cache has no teams to analyze.")
        st.stop()

    st.sidebar.header("Filters")
    circuit_options = [circuit for circuit in CIRCUITS if circuit in set(teams["circuit"])]
    if not circuit_options:
        st.error("No cached teams match the supported circuit list.")
        st.stop()
    default_circuit = "VCT Americas" if "VCT Americas" in circuit_options else circuit_options[0]
    if st.session_state.get("shared_circuit") not in circuit_options:
        st.session_state["shared_circuit"] = default_circuit
    selected_circuit = st.sidebar.selectbox("Circuit", circuit_options, key="shared_circuit")
    circuit_teams = teams[teams["circuit"].eq(selected_circuit)].copy()
    circuit_teams = circuit_teams.sort_values(
        ["series_count", "team_name", "team_id"],
        ascending=[False, True, True], kind="stable",
    )
    team_by_id = {
        str(row["team_id"]): row for _, row in circuit_teams.iterrows()
    }
    team_options = list(team_by_id)
    default_team_id = next(
        (team_id for team_id, team in team_by_id.items()
         if selected_circuit == "VCT Americas" and team["team_name"] == "Cloud9"),
        team_options[0],
    )
    if st.session_state.get("shared_team") not in team_options:
        st.session_state["shared_team"] = default_team_id
    selected_team_id = st.sidebar.selectbox(
        "Team", team_options, key="shared_team",
        format_func=lambda value: team_by_id[str(value)]["label"],
    )
    selected_team = team_by_id[str(selected_team_id)]
    team_id = str(selected_team["team_id"])

    team_rounds = round_index[round_index["team_id"].astype(str).eq(team_id)]
    map_counts = team_rounds.drop_duplicates(["series_id", "game_id", "round_number"]).groupby("map").size()
    maps = map_counts.sort_values(ascending=False, kind="stable").index.astype(str).tolist()
    if not maps:
        st.warning("No rounds are available for this team.")
        st.stop()
    active_tab = st.session_state.get("scout_tabs", "Utility Map")
    counter_tab_label = (
        "Self-scout: predictable habits"
        if selected_team["team_name"] in {"Cloud9", "Cloud9 GC"} else "Counter Plan"
    )
    if active_tab in {"Counter Plan", "Self-scout: predictable habits"}:
        if active_tab != counter_tab_label:
            st.session_state["scout_tabs"] = counter_tab_label
            active_tab = counter_tab_label
    if active_tab == "Operator Profiler":
        map_options = ["All maps", *maps]
        map_widget_key = f"operator_map_widget_{team_id}"
        persistent_map_key = f"operator_map_selection_{team_id}"
    elif active_tab in {"Counter Plan", "Self-scout: predictable habits"}:
        map_options = maps
        map_widget_key = f"counter_map_widget_{team_id}"
        persistent_map_key = f"counter_map_selection_{team_id}"
    else:
        map_options = maps
        map_widget_key = f"utility_map_widget_{team_id}"
        persistent_map_key = f"utility_map_selection_{team_id}"
    default_map = map_options[0] if active_tab == "Operator Profiler" else maps[0]
    team_changed = st.session_state.get("last_map_team_id") != team_id
    map_tab = "Counter Plan" if active_tab in {"Counter Plan", "Self-scout: predictable habits"} else active_tab
    if team_changed and st.session_state.get("last_map_tab") == map_tab:
        selected_map = _map_for_team(
            st.session_state.get("last_visible_map"), map_options, map_counts, maps[0], True,
        )
        st.session_state[persistent_map_key] = selected_map
        st.session_state[map_widget_key] = selected_map
    else:
        st.session_state[persistent_map_key] = _map_for_team(
            st.session_state.get(persistent_map_key), map_options, map_counts, default_map, team_changed,
        )
        st.session_state[map_widget_key] = _map_for_team(
            st.session_state.get(map_widget_key, st.session_state[persistent_map_key]),
            map_options, map_counts, default_map, team_changed,
        )
    st.session_state["last_map_team_id"] = team_id
    map_name = st.sidebar.selectbox(
        "Map", map_options, key=map_widget_key,
        on_change=_persist_filter_value, args=(map_widget_key, persistent_map_key),
    )
    st.session_state["last_visible_map"] = map_name
    st.session_state["last_map_tab"] = map_tab
    if "shared_side" not in st.session_state:
        st.session_state["shared_side"] = "defense"
    selected_side = st.sidebar.selectbox("Side", ("attack", "defense"), key="shared_side")
    side = "attacker" if selected_side == "attack" else "defender"
    try:
        tables = load_tables(signature, team_id, map_name)
        shapes = load_shapes((CACHE_DIR / "map_shapes.npz").stat().st_mtime_ns)
        site_centres = load_site_shapes((CACHE_DIR / "site_centres.json").stat().st_mtime_ns)
        spawn_estimates_path = CACHE_DIR / "spawn_estimates.json"
        spawn_signature = spawn_estimates_path.stat().st_mtime_ns if spawn_estimates_path.is_file() else None
        spawn_estimates = load_global_spawn_estimates(spawn_signature)
    except FileNotFoundError as error:
        st.error(f"Cache file missing: {error.filename or error}. Rebuild it with `uv run python build_cache.py --all`.")
        st.stop()
    report_data = _scouting_report_download_data(
        (
            signature,
            (CACHE_DIR / "map_shapes.npz").stat().st_mtime_ns,
            (CACHE_DIR / "site_centres.json").stat().st_mtime_ns,
            spawn_signature,
        ),
        date.today().isoformat(), selected_team["team_name"], team_id, map_name,
        selected_side, side, tables, shapes, site_centres, spawn_estimates,
    )
    _render_app_style()
    title_col, badge_col = st.columns([5, 1])
    title_col.title("Scout Sheet")
    if selected_team["team_name"] in {"Cloud9", "Cloud9 GC"}:
        badge_col.markdown(
            '<div style="margin-top: 18px; padding: 6px 12px; border-radius: 999px; '
            'background: #164e63; color: #cffafe; font-weight: 700; text-align: center;">Self-scout</div>',
            unsafe_allow_html=True,
        )
    st.caption("VALORANT opponent scouting from official GRID match data")
    _render_welcome_panel()
    if str(map_name).casefold() == "summit" and not site_centres.get("summit"):
        st.caption("Site labels are hidden for Summit because no verified plant-derived site centres are available.")
    utility_tab, operator_tab, counter_tab = st.tabs(
        ["Utility Map", "Operator Profiler", counter_tab_label], key="scout_tabs", on_change="rerun",
    )
    with utility_tab:
        if utility_tab.open:
            _render_utility_tab(
                tables, selected_team["team_name"], team_id, map_name, selected_side, side,
                shapes, site_centres, report_data,
            )
    with operator_tab:
        if operator_tab.open:
            _render_operator_tab(
                tables, selected_team["team_name"], team_id, map_name, selected_side,
                shapes, site_centres, report_data, spawn_estimates,
            )
    with counter_tab:
        if counter_tab.open:
            _render_counter_plan_tab(
                tables, selected_team["team_name"], team_id, map_name, selected_side, side,
                site_centres, report_data, spawn_estimates,
            )
    st.markdown(
        "<div style='margin-top: 2rem; padding-top: 0.8rem; border-top: 1px solid rgba(148,163,184,.3); "
        "text-align: center; color: #94a3b8; font-size: .8rem;'>"
        "Built with JetBrains PyCharm, Air and Junie · Data: GRID</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()