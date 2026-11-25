"""Deterministic, evidence-gated scouting tendencies for Counter Plan."""

import math

import pandas as pd

from analysis.abilities import ability_display_name
from analysis.operator import (
    aggressiveness_label,
    aggressiveness_score,
    calculate_buy_rates,
    estimate_spawn_anchors,
    operator_engagements,
)
from analysis.sites import site_label
from analysis.zones import eligible_rounds, filter_casts, match_casts_to_rounds


ROUND_KEYS = ["series_id", "game_id", "round_number"]
CONFIDENCE_ORDER = {"High": 0, "Medium": 1}
CELL_SIZE = 400


def confidence_level(rounds, percentage):
    """Return confidence only when both the sample and prevalence qualify."""
    rounds = int(rounds)
    percentage = float(percentage)
    if rounds >= 40 and percentage >= 50:
        return "High"
    if rounds >= 20 and percentage >= 35:
        return "Medium"
    return None


def _percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0.0


def _time_window(seconds):
    seconds = pd.to_numeric(seconds, errors="coerce")
    if pd.isna(seconds):
        return "unknown"
    if seconds < 0:
        return "buy phase"
    if seconds < 20:
        return "0–20 s"
    if seconds < 40:
        return "20–40 s"
    return "40 s+"


def _dominant_cast_window(casts):
    buckets = casts["seconds_into_round"].map(_time_window)
    counts = buckets.value_counts()
    return str(counts.index[0]) if not counts.empty else "an unknown time"


def _response(team_name, self_scout, advice):
    advice = advice[0].lower() + advice[1:] if advice else advice
    if self_scout:
        return f"{team_name} is predictable here — consider varying the setup: {advice}"
    return advice


def _utility_cards(rounds, casts, team_id, team_name, map_name, side, site_centres, self_scout):
    display_side = "defense" if side == "defender" else "attack"
    if casts.empty:
        return []
    denominator = eligible_rounds(rounds, team_id, map_name, side)
    if denominator.empty:
        return []
    total_rounds = len(denominator.drop_duplicates(ROUND_KEYS))
    matched, _ = match_casts_to_rounds(casts, denominator)
    selected = filter_casts(
        matched, seconds_range=(0, 100), include_remote=False, include_buy_phase=True,
    )
    selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["y"] = pd.to_numeric(selected["y"], errors="coerce")
    selected = selected.dropna(subset=["x", "y"])
    if selected.empty:
        return []
    selected["cell_x"] = (selected["x"] / CELL_SIZE).map(math.floor)
    selected["cell_y"] = (selected["y"] / CELL_SIZE).map(math.floor)
    cards = []
    group_columns = ["ability", "agent", "cell_x", "cell_y"]
    for (ability, agent, cell_x, cell_y), group in selected.groupby(group_columns, dropna=False, sort=False):
        rounds_used = group[ROUND_KEYS].astype("string").drop_duplicates().shape[0]
        percentage = _percentage(rounds_used, total_rounds)
        if percentage < 35:
            continue
        confidence = confidence_level(total_rounds, percentage)
        if confidence is None:
            continue
        zone = site_label(map_name, (cell_x + 0.5) * CELL_SIZE, (cell_y + 0.5) * CELL_SIZE, site_centres)
        ability_name = ability_display_name(ability)
        window = _dominant_cast_window(group)
        series_count = int(group["series_id"].nunique())
        tendency = (
            f"Expect {ability_name} at {zone} in {percentage:.0f}% of {display_side} rounds — "
            f"most casts happen {window}."
        )
        advice = (
            f"Clear it with utility or avoid the {zone} lane before executing; "
            f"most casts happen {window}."
        )
        cards.append({
            "type": "Utility",
            "tendency": tendency,
            "evidence": f"{rounds_used}/{total_rounds} rounds ({percentage:.0f}%), {series_count} series.",
            "confidence": confidence,
            "confidence_order": CONFIDENCE_ORDER[confidence],
            "frequency": percentage,
            "response": _response(team_name, self_scout, advice),
            "rounds": rounds_used,
            "total_rounds": total_rounds,
            "series_count": series_count,
            "category": group.iloc[0]["category"],
            "agent": str(agent),
            "ability": str(ability),
            "map": map_name,
            "side": side,
            "navigation_tab": "Utility Map",
        })
    return cards


def _operator_cards(
    rounds, buys, kills, damage, spawn_positions, team_id, team_name, map_name, side,
    site_centres, self_scout,
):
    all_rates = calculate_buy_rates(rounds, buys, team_id, map_name)
    side_key = side
    display_side = "defense" if side == "defender" else "attack"
    side_rates = calculate_buy_rates(rounds, buys, team_id, map_name, side=side_key)
    cards = []
    team_map_rounds = rounds[
        rounds["team_id"].astype(str).eq(str(team_id))
        & rounds["map"].astype(str).str.casefold().eq(str(map_name).casefold())
    ]
    series_count = int(team_map_rounds["series_id"].nunique())

    if not side_rates["players"].empty:
        primary = side_rates["players"].iloc[0]
        op_rounds = side_rates["overall"]["held_rounds"]
        share = float(primary["share"])
        confidence = confidence_level(op_rounds, share)
        if confidence:
            player_id = str(primary["player_id"])
            engagements = operator_engagements(kills, damage, buys, team_id, map_name, "defender")
            if not engagements.empty:
                engagements = engagements[engagements["player_id"].astype(str).eq(player_id)].copy()
            site = "an unknown site"
            timing_detail = ""
            angle_detail = ""
            timed_engagement_count = 0
            if len(engagements) >= 10:
                sites = engagements.apply(
                    lambda row: site_label(row["map"], row.get("x"), row.get("y"), site_centres), axis=1,
                )
                known_sites = sites[~sites.isin(["Site unknown", "mid"])]
                if not known_sites.empty:
                    site = str(known_sites.value_counts().index[0])
                times = pd.to_numeric(engagements["seconds_into_round"], errors="coerce")
                timed = times.notna()
                timed_engagement_count = int(timed.sum())
                if int(timed.sum()) >= 10:
                    early = int(times.between(0, 20, inclusive="left").sum())
                    late = int((times >= 20).sum())
                    timing_detail = f" Early/late defense engagements: 0–20 s {early}/{int(timed.sum())}, 20 s+ {late}/{int(timed.sum())}."
                anchors = estimate_spawn_anchors(spawn_positions)
                labels = []
                for event in engagements.itertuples(index=False):
                    map_anchors = anchors.get(str(event.map).casefold(), {})
                    defender_spawn = map_anchors.get("defender")
                    attacker_spawn = map_anchors.get("attacker")
                    score = (
                        aggressiveness_score((event.x, event.y), defender_spawn, attacker_spawn)
                        if defender_spawn is not None and attacker_spawn is not None else None
                    )
                    labels.append(aggressiveness_label(score))
                known = pd.Series(labels).value_counts()
                known_count = int(known.sum() - known.get("unknown", 0))
                if known_count >= 10:
                    angle_detail = (
                        f" Aggression split: {int(known.get('aggressive', 0))} aggressive, "
                        f"{int(known.get('passive', 0))} passive of {known_count} classified engagements."
                    )

            buy_rates = all_rates["by_side"]
            attack = buy_rates.get("attacker", {})
            defense = buy_rates.get("defender", {})
            tendency = (
                f"{primary['player_name']} holds {team_name}'s Operator in {share:.0f}% of their "
                f"{display_side} Operator rounds ({int(primary['rounds'])}/{op_rounds}); buy rates are "
                f"{defense.get('rate', 0):.0f}% on defense and {attack.get('rate', 0):.0f}% on attack."
                f"{timing_detail}{angle_detail}"
            )
            if timed_engagement_count >= 10 and site != "an unknown site":
                advice = f"Don't dry-peek {site} in the first 20 s; use flashes or smokes on the Operator angle."
            else:
                advice = "Use flashes or smokes to clear likely Operator angles rather than dry-peeking."
            cards.append({
                "type": "Operator",
                "tendency": tendency,
                "evidence": (
                    f"{int(primary['rounds'])}/{op_rounds} Operator rounds ({share:.0f}% holder share); "
                    f"buy snapshots: defense {defense.get('held_rounds', 0)}/{defense.get('observed_rounds', 0)}, "
                    f"attack {attack.get('held_rounds', 0)}/{attack.get('observed_rounds', 0)}; "
                    f"{series_count} series."
                ),
                "confidence": confidence,
                "confidence_order": CONFIDENCE_ORDER[confidence],
                "frequency": share,
                "response": _response(team_name, self_scout, advice),
                "rounds": int(primary["rounds"]),
                "total_rounds": int(op_rounds),
                "series_count": series_count,
                "map": map_name,
                "side": display_side,
                "navigation_tab": "Operator Profiler",
            })

    attack = all_rates["by_side"].get("attacker", {})
    defense = all_rates["by_side"].get("defender", {})
    if attack.get("observed_rounds", 0) >= 20 and defense.get("observed_rounds", 0) >= 20:
        difference = abs(float(attack["rate"]) - float(defense["rate"]))
        if difference >= 10:
            higher_side, higher = ("attack", attack) if attack["rate"] > defense["rate"] else ("defense", defense)
            lower_side, lower = ("defense", defense) if higher_side == "attack" else ("attack", attack)
            confidence = confidence_level(higher["observed_rounds"], higher["rate"])
            if confidence:
                tendency = (
                    f"{team_name} buy the Operator in {higher['rate']:.0f}% of {higher_side} rounds "
                    f"versus {lower['rate']:.0f}% on {lower_side} (a {difference:.0f}-point gap)."
                )
                advice = (
                    f"Expect the Operator more on their {higher_side}; use flashes or smokes to clear angles "
                    "rather than dry-peeking."
                )
                cards.append({
                    "type": "Side economy",
                    "tendency": tendency,
                    "evidence": (
                        f"{higher['held_rounds']}/{higher['observed_rounds']} {higher_side} snapshots "
                        f"({higher['rate']:.0f}%), {lower['held_rounds']}/{lower['observed_rounds']} "
                        f"{lower_side} snapshots ({lower['rate']:.0f}%), {difference:.0f} points; "
                        f"{series_count} series."
                    ),
                    "confidence": confidence,
                    "confidence_order": CONFIDENCE_ORDER[confidence],
                    "frequency": float(higher["rate"]),
                    "response": _response(team_name, self_scout, advice),
                    "rounds": int(higher["observed_rounds"]),
                    "total_rounds": int(higher["eligible_rounds"]),
                    "series_count": series_count,
                    "map": map_name,
                    "side": display_side,
                    "navigation_tab": "Operator Profiler",
                })
    return cards


def generate_counter_cards(
    rounds, casts, buys, kills, damage, spawn_positions, team_id, team_name, map_name, side,
    site_centres=None, self_scout=False,
):
    """Build local rule-based cards, ordered by confidence and prevalence."""
    site_centres = site_centres or {}
    cards = _utility_cards(rounds, casts, team_id, team_name, map_name, side, site_centres, self_scout)
    cards.extend(_operator_cards(
        rounds, buys, kills, damage, spawn_positions, team_id, team_name, map_name, side,
        site_centres, self_scout,
    ))
    cards.sort(key=lambda card: (card["confidence_order"], -card["frequency"], card["type"], card["tendency"]))
    return cards[:8]