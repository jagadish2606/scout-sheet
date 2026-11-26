"""Aggregate-only printable scouting report generation."""

from __future__ import annotations

import base64
from datetime import date
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis.counter_plan import generate_counter_plan
from analysis.landing import build_walkable_mask
from analysis.operator import calculate_buy_rates
from analysis.sites import disambiguate_zone_labels, label_zones
from analysis.zones import aggregate_zone_cells, eligible_rounds, filter_casts, match_casts_to_rounds, merge_zone_cells


def _report_map(rounds, team_id, map_name, side):
    """Choose a single map for spatial utility evidence without combining coordinates."""
    if str(map_name).casefold() != "all maps":
        return str(map_name)
    selected = rounds[
        rounds["team_id"].astype(str).eq(str(team_id))
        & rounds["side"].astype(str).str.casefold().eq(str(side).casefold())
    ]
    if selected.empty:
        return None
    counts = selected.drop_duplicates(["series_id", "game_id", "round_number"]).groupby("map").size()
    return str(counts.sort_values(ascending=False, kind="stable").index[0]) if not counts.empty else None


def _date_range(rounds):
    if "round_start_time" not in rounds:
        return "Date range unavailable"
    dates = pd.to_datetime(rounds["round_start_time"], utc=True, errors="coerce").dropna()
    if dates.empty:
        return "Date range unavailable"
    first, last = dates.min().date(), dates.max().date()
    return first.isoformat() if first == last else f"{first.isoformat()} – {last.isoformat()}"


def _map_png(shape, zones, site_centres, image_exporter=None):
    figure = go.Figure()
    density = np.asarray(shape["density"], dtype=float).copy()
    density[~build_walkable_mask(shape)] = np.nan
    figure.add_trace(go.Heatmap(
        x=shape["x"], y=shape["y"], z=density,
        colorscale=[[0, "#f1f5f9"], [0.35, "#cbd5e1"], [1, "#64748b"]],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip", name="Observed position density",
    ))
    map_sites = site_centres.get(str(shape.get("map", "")).casefold(), {})
    if map_sites:
        figure.add_trace(go.Scatter(
            x=[site["x"] for site in map_sites.values()], y=[site["y"] for site in map_sites.values()],
            text=list(map_sites), mode="markers+text", textposition="middle center",
            textfont={"size": 15, "color": "#0f172a", "family": "Arial Black"},
            marker={"size": 24, "color": "rgba(14,165,233,0.2)", "line": {"color": "#0284c7", "width": 1}},
            hoverinfo="skip", showlegend=False,
        ))
    top_zones = zones.head(3)
    if not top_zones.empty:
        figure.add_trace(go.Scatter(
            x=top_zones["x"], y=top_zones["y"], mode="markers+text",
            text=[str(index) for index in range(1, len(top_zones) + 1)], textposition="middle center",
            textfont={"size": 10, "color": "#111827", "family": "Arial Black"},
            marker={"size": 28, "color": "#fbbf24", "line": {"color": "#fff", "width": 1}},
            hoverinfo="skip", showlegend=False,
        ))
    x_span = float(shape["x_max"]) - float(shape["x_min"])
    y_span = float(shape["y_max"]) - float(shape["y_min"])
    figure.update_layout(
        width=900, height=540, margin={"l": 2, "r": 2, "t": 2, "b": 2},
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", showlegend=False,
        xaxis={"range": [float(shape["x_min"]) - x_span * 0.05, float(shape["x_max"]) + x_span * 0.05],
               "visible": False, "constrain": "domain"},
        yaxis={"range": [float(shape["y_min"]) - y_span * 0.05, float(shape["y_max"]) + y_span * 0.05],
               "visible": False, "constrain": "domain", "scaleanchor": "x", "scaleratio": 1},
    )
    if image_exporter is not None:
        return image_exporter(figure)
    return figure.to_image(format="png", width=900, height=540, scale=1)


def _zone_rows(zones, side, map_name):
    if zones.empty:
        return "<li class='muted'>No placeable utility zones were recorded for this sample.</li>"
    rows = []
    for zone in zones.head(3).itertuples(index=False):
        rows.append(
            f"<li><b>{escape(str(zone.site_label))}</b>"
            f"<span class='ability-name'>{escape(str(zone.ability_name))}</span>"
            f"<small><b>Agent:</b> {escape(str(zone.agent).title())} · "
            f"<b>Player:</b> {escape(str(zone.player_name))}; "
            f"{int(zone.rounds_used)}/{int(zone.total_rounds)} {escape(side)} rounds "
            f"({float(zone.percentage):.0f}%), {int(zone.series_count)} series · {escape(map_name)}</small></li>"
        )
    return "".join(rows)


def _operator_summary(rounds, buys, team_id, map_name, selected_side, team_name):
    rates = calculate_buy_rates(rounds, buys, team_id, map_name)
    side_key = "attacker" if selected_side == "attack" else "defender"
    side_rates = rates["by_side"].get(side_key, {})
    observed = int(side_rates.get("observed_rounds", 0))
    held = int(side_rates.get("held_rounds", 0))
    rate = float(side_rates.get("rate", 0))
    selected = calculate_buy_rates(rounds, buys, team_id, map_name, side=side_key)
    holder_rounds = int(selected["overall"]["held_rounds"])
    primary = selected["players"].iloc[0] if not selected["players"].empty else None
    if primary is None:
        holder = "No freeze-time Operator holder was recorded in the selected sample."
    else:
        holder = (
            f"{escape(str(primary['player_name']))} was the leading holder, appearing in "
            f"{int(primary['rounds'])}/{holder_rounds} Operator rounds ({float(primary['share']):.0f}%)."
        )
    return (
        f"<p><b>{escape(team_name)}</b> held an Operator in {held}/{observed} recorded {escape(selected_side)} "
        f"buy snapshots ({rate:.0f}%).</p><p>{holder}</p>"
        f"<p class='muted'>Snapshot coverage: {observed}/{int(side_rates.get('eligible_rounds', 0))} rounds; "
        "freeze-time possession does not track later pickups or drops.</p>"
    )


def _counter_cards(tables, team_name, team_id, map_name, utility_map, side, site_centres, spawn_estimates=None):
    plan = generate_counter_plan(
        tables["rounds"], tables["casts"], tables["buys"], tables["kills"], tables["damage"],
        tables["spawn_positions"], team_id, team_name, utility_map, side, site_centres,
        self_scout=team_name in {"Cloud9", "Cloud9 GC"},
        spawn_estimates=spawn_estimates,
    )
    cards = plan["cards"][:3]
    if not cards:
        return "<p class='muted'>No strong Counter Plan tendencies met the current evidence thresholds.</p>"
    blocks = []
    for card in cards:
        facts = "".join(f"<li>{escape(str(fact))}</li>" for fact in card["facts"][:3])
        blocks.append(
            f"<article class='counter'><b>{escape(str(card['tendency']))}</b> "
            f"<span class='badge {escape(str(card['confidence']).lower())}'>"
            f"{escape(str(card['confidence']))}</span><ul>{facts}</ul>"
            f"<p class='response'>{escape(str(card['response']))}</p></article>"
        )
    return "".join(blocks)


def generate_scouting_report(
    tables, team_name, team_id, map_name, selected_side, side, shapes, site_centres,
    image_exporter=None, generated_on=None, spawn_estimates=None,
):
    """Build a standalone, print-focused HTML report containing aggregates only."""
    rounds = tables["rounds"]
    team_scope = rounds[
        rounds["team_id"].astype(str).eq(str(team_id))
        & rounds["side"].astype(str).str.casefold().eq(str(side).casefold())
    ]
    if str(map_name).casefold() != "all maps":
        team_scope = team_scope[team_scope["map"].astype(str).str.casefold().eq(str(map_name).casefold())]
    dates = _date_range(team_scope)
    utility_map = _report_map(rounds, team_id, map_name, side)
    denominator = eligible_rounds(rounds, team_id, utility_map, side) if utility_map else rounds.iloc[:0]
    matched, _ = match_casts_to_rounds(tables["casts"], denominator)
    placeable = filter_casts(
        matched, category="placeable", seconds_range=(0, 100), include_remote=False, include_buy_phase=True,
    )
    if {"landing_x", "landing_y"}.issubset(placeable.columns):
        placeable["x"] = pd.to_numeric(placeable["landing_x"], errors="coerce").fillna(
            pd.to_numeric(placeable["x"], errors="coerce"),
        )
        placeable["y"] = pd.to_numeric(placeable["landing_y"], errors="coerce").fillna(
            pd.to_numeric(placeable["y"], errors="coerce"),
        )
    if utility_map:
        cell_zones = label_zones(aggregate_zone_cells(placeable, denominator), utility_map, site_centres)
        zones = disambiguate_zone_labels(
            merge_zone_cells(cell_zones), utility_map, site_centres,
        )
    else:
        zones = pd.DataFrame()
    map_title = "All maps" if str(map_name).casefold() == "all maps" else str(map_name).title()
    spatial_title = f"Most-played map: {utility_map.title()}" if str(map_name).casefold() == "all maps" and utility_map else map_title

    image_markup = "<div class='map-placeholder'>Map silhouette unavailable for this selection.</div>"
    if utility_map and utility_map.casefold() in {str(key).casefold() for key in shapes}:
        shape = next(value for key, value in shapes.items() if str(key).casefold() == utility_map.casefold())
        shape = {**shape, "map": utility_map}
        try:
            png = _map_png(shape, zones, site_centres, image_exporter)
            image_markup = f"<img alt='Static utility zone map' src='data:image/png;base64,{base64.b64encode(png).decode('ascii')}'>"
        except Exception:
            image_markup = (
                "<div class='map-placeholder'>Static map export unavailable. Install Kaleido and its Chrome runtime, "
                "then rebuild the report.</div>"
            )

    if zones.empty:
        utility_insight = f"No placeable hot-zone evidence for {escape(team_name)} on {escape(spatial_title)} yet."
    else:
        strongest = zones.iloc[0]
        location = str(strongest["site_label"])
        location_phrase = location if location.casefold().startswith("near ") else f"at {location}"
        if location.casefold() == "mid":
            location_phrase = "through mid"
        utility_insight = (
            f"{escape(team_name)} most often use {escape(str(strongest['ability_name']))} "
            f"{escape(location_phrase)} in {float(strongest['percentage']):.0f}% of "
            f"{escape(selected_side)} rounds ({int(strongest['rounds_used'])}/{int(strongest['total_rounds'])})."
        )
    operator_markup = _operator_summary(rounds, tables["buys"], team_id, map_name, selected_side, team_name)
    counters = _counter_cards(
        tables, team_name, team_id, map_name, utility_map, side, site_centres, spawn_estimates,
    ) if utility_map else ""
    report_date = (generated_on or date.today()).isoformat()
    scope_note = (
        f" Spatial utility uses the team's most-played map, {escape(spatial_title)}, because coordinates cannot be combined across maps."
        if str(map_name).casefold() == "all maps" and utility_map else ""
    )
    small_sample = len(denominator.drop_duplicates(["series_id", "game_id", "round_number"])) < 20
    sample_note = "Small sample: fewer than 20 rounds." if small_sample else ""
    css = """
    @page { size: A4 portrait; margin: 8mm; }
    * { box-sizing: border-box; }
    body { margin: 0; color: #172033; background: #fff; font: 9pt/1.28 Arial, sans-serif; }
    .sheet { width: 194mm; height: 280mm; overflow: hidden; margin: auto; padding: 2mm; }
    header { border-bottom: 2px solid #0f172a; padding-bottom: 3mm; margin-bottom: 3mm; }
    h1 { font-size: 19pt; margin: 0 0 1.5mm; }
    h2 { font-size: 11pt; margin: 0 0 2mm; color: #0f172a; }
    h3 { font-size: 9pt; margin: 0 0 1mm; }
    .meta, .muted { color: #475569; font-size: 8pt; }
    .insight { padding: 2.5mm 3mm; margin: 2mm 0 3mm; background: #eff6ff; border-left: 3px solid #2563eb; font-weight: 700; }
    .grid { display: grid; grid-template-columns: 1.15fr .85fr; gap: 3mm; }
    .panel { border: 1px solid #cbd5e1; border-radius: 2mm; padding: 2.5mm; margin-bottom: 3mm; }
    .map { min-height: 60mm; display: flex; align-items: center; }
    img { display: block; width: 100%; height: auto; max-height: 72mm; object-fit: contain; }
    ol, ul { margin: 1mm 0 0; padding-left: 5mm; }
    li { margin: 0 0 1.4mm; }
    .ability-name { display: block; font-weight: 700; }
    li small { display: block; color: #475569; }
    .operator p { margin: 1mm 0; }
    .counter { border-top: 1px solid #e2e8f0; padding: 1.5mm 0; }
    .counter:first-of-type { border-top: 0; }
    .badge { display: inline-block; font-size: 7pt; font-weight: bold; border-radius: 8px; padding: 1px 5px; background: #dcfce7; color: #166534; }
    .badge.medium { background: #fef3c7; color: #854d0e; }
    .response { margin: 1mm 0 0; color: #334155; font-style: italic; }
    .map-placeholder { padding: 10mm 4mm; text-align: center; background: #f1f5f9; color: #475569; }
    footer { border-top: 1px solid #cbd5e1; margin-top: 2mm; padding-top: 2mm; color: #475569; font-size: 7pt; }
    @media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } .sheet { margin: 0; } }
    """
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scout Sheet · {escape(str(team_name))} · {escape(map_title)} · {escape(selected_side.title())}</title>
<style>{css}</style></head><body><main class="sheet">
<header><h1>{escape(str(team_name))} · Scouting Report</h1>
<div class="meta">{escape(map_title)} · {escape(selected_side.title())} · Data: {escape(dates)} · Generated {escape(report_date)}</div></header>
<div class="insight">{utility_insight}{scope_note}{(' ' + escape(sample_note)) if sample_note else ''}</div>
<div class="grid"><section class="panel"><h2>Utility hot zones</h2><div class="meta">{escape(spatial_title)} · placeable utility · estimated locations use aim and the observed-position wall mask.</div>
<div class="map">{image_markup}</div></section>
<section class="panel"><h2>Top 3 zones</h2><ol>{_zone_rows(zones, selected_side, spatial_title)}</ol></section></div>
<div class="grid"><section class="panel operator"><h2>Operator summary</h2>{operator_markup}</section>
<section class="panel"><h2>Counter Plan · top tendencies</h2>{counters}</section></div>
<footer>Data: GRID official esports data · generated by Scout Sheet. Aggregate evidence only; landing points and site labels are estimates. Print at 100% scale with A4 paper.</footer>
</main></body></html>"""