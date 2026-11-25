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
from analysis.text import team_possessive
from analysis.zones import eligible_rounds, filter_casts, match_casts_to_rounds


ROUND_KEYS = ["series_id", "game_id", "round_number"]
CONFIDENCE_ORDER = {"High": 0, "Medium": 1, "Low": 2}
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
    if counts.empty:
        return "an unknown time", 0, 0
    return str(counts.index[0]), int(counts.iloc[0]), int(len(buckets))


def _card_confidence(rounds, percentage):
    confidence = confidence_level(rounds, percentage)
    if confidence:
        return confidence
    if int(rounds) >= 20 and 25 <= float(percentage) < 35:
        return "Low"
    return None


def _buy_snapshot_fact(side, summary):
    observed = int(summary.get("observed_rounds", 0))
    if not observed:
        return f"No {side} buy snapshots recorded (0 observed)."
    return (
        f"{summary.get('held_rounds', 0)}/{observed} {side} buy snapshots "
        f"({summary.get('rate', 0):.0f}%)."
    )


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
        confidence = _card_confidence(total_rounds, percentage)
        if confidence is None:
            continue
        zone = site_label(map_name, (cell_x + 0.5) * CELL_SIZE, (cell_y + 0.5) * CELL_SIZE, site_centres)
        ability_name = ability_display_name(ability)
        window, window_count, cast_count = _dominant_cast_window(group)
        series_count = int(group["series_id"].nunique())
        if self_scout:
            tendency = (
                f"Opponents can expect {ability_name} at {zone} in {percentage:.0f}% of your "
                f"{display_side} rounds ({rounds_used}/{total_rounds}), mostly during {window}."
            )
            title = f"Your {ability_name} at {zone}"
            response = "Consider varying timing or position."
        else:
            tendency = (
                f"Expect {ability_name} at {zone} in {percentage:.0f}% of {display_side} rounds — "
                f"most casts happen {window}."
            )
            title = f"{ability_name} at {zone}"
            response = (
                f"Clear it with utility or avoid the {zone} lane before executing; "
                f"most casts happen {window}."
            )
        cards.append({
            "type": "Utility",
            "title": title,
            "tendency": tendency,
            "evidence": f"{rounds_used}/{total_rounds} rounds ({percentage:.0f}%), {series_count} series.",
            "facts": [
                f"{rounds_used}/{total_rounds} {display_side} rounds ({percentage:.0f}%).",
                f"{window_count}/{cast_count} casts during {window}; {series_count} series.",
            ],
            "confidence": confidence,
            "confidence_order": CONFIDENCE_ORDER[confidence],
            "frequency": percentage,
            "response": response,
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
    all_rates = calculate_buy_rates(rounds, buys, team_id, "All maps")
    side_key = side
    display_side = "defense" if side == "defender" else "attack"
    side_rates = calculate_buy_rates(rounds, buys, team_id, "All maps", side=side_key)
    cards = []
    team_rounds = rounds[
        rounds["team_id"].astype(str).eq(str(team_id))
        & rounds["side"].astype(str).str.casefold().eq(str(side_key).casefold())
    ]
    series_count = int(team_rounds["series_id"].nunique())

    if not side_rates["players"].empty:
        primary = side_rates["players"].iloc[0]
        op_rounds = side_rates["overall"]["held_rounds"]
        share = float(primary["share"])
        confidence = _card_confidence(op_rounds, share)
        if confidence:
            player_id = str(primary["player_id"])
            engagements = pd.DataFrame()
            if side_key == "defender":
                engagements = operator_engagements(kills, damage, buys, team_id, "All maps", "defender")
            if not engagements.empty:
                engagements = engagements[engagements["player_id"].astype(str).eq(player_id)].copy()
            times = pd.to_numeric(engagements.get("seconds_into_round", pd.Series(dtype=float)), errors="coerce")
            live_timed = times.notna() & times.ge(0)
            live_times = times[live_timed]
            timing_count = int(len(live_times))
            early_count = int(live_times.between(0, 20, inclusive="left").sum())
            late_count = int(live_times.ge(20).sum())

            anchors = estimate_spawn_anchors(spawn_positions)
            labels = []
            locations = []
            for event in engagements.itertuples(index=False):
                map_anchors = anchors.get(str(event.map).casefold(), {})
                defender_spawn = map_anchors.get("defender")
                attacker_spawn = map_anchors.get("attacker")
                score = (
                    aggressiveness_score((event.x, event.y), defender_spawn, attacker_spawn)
                    if defender_spawn is not None and attacker_spawn is not None else None
                )
                labels.append(aggressiveness_label(score))
                zone = site_label(event.map, event.x, event.y, site_centres)
                if zone not in {"Site unknown", "mid"}:
                    locations.append(f"{str(event.map).title()} {zone}")
            aggression = pd.Series(labels).value_counts()
            aggression_count = int(aggression.sum() - aggression.get("unknown", 0))
            aggressive_count = int(aggression.get("aggressive", 0))
            passive_count = int(aggression.get("passive", 0))
            location = (
                str(pd.Series(locations).value_counts().index[0]) if locations else "a deep angle across maps"
            )

            early_signal = timing_count >= 10 and early_count / timing_count >= 0.5
            late_signal = timing_count >= 10 and late_count / timing_count >= 0.6
            aggressive_signal = aggression_count >= 10 and aggressive_count / aggression_count >= 0.5
            passive_signal = aggression_count >= 10 and passive_count / aggression_count >= 0.5
            expects_early = early_signal or aggressive_signal
            expects_late = late_signal or passive_signal
            if expects_early and expects_late:
                pattern = "mixed"
            elif expects_early:
                pattern = "early"
            elif expects_late:
                pattern = "late"
            else:
                pattern = "mixed"

            buy_rates = all_rates["by_side"]
            attack = buy_rates.get("attacker", {})
            defense = buy_rates.get("defender", {})
            if self_scout:
                tendency = (
                    f"Opponents can expect {primary['player_name']} to hold your Operator in {share:.0f}% "
                    f"of your {display_side} Operator rounds ({int(primary['rounds'])}/{op_rounds}) across all maps."
                )
                advice = "Consider varying the Operator holder, timing, or angle across maps."
            else:
                tendency = (
                    f"{primary['player_name']} holds {team_possessive(team_name)} Operator in {share:.0f}% of their "
                    f"{display_side} Operator rounds ({int(primary['rounds'])}/{op_rounds}) across all maps."
                )
                if timing_count < 10 and aggression_count < 10:
                    advice = "Use flashes or smokes to clear likely Operator angles rather than dry-peeking."
                elif pattern == "early":
                    advice = (
                        f"Expect an early, aggressive Op at {location}: don't dry-peek in the first 20 s; "
                        "flash or smoke the angle, or bait with a second player."
                    )
                elif pattern == "late":
                    advice = (
                        f"Expect a passive Op holding deep at {location} later in the round: avoid wide swings late, "
                        "clear deep angles with utility, and force rotations early."
                    )
                else:
                    advice = (
                        f"Expect both early pressure and late deep holds around {location}: avoid isolated dry-peeks, "
                        "clear angles with utility, and vary your timing."
                    )

            facts = [
                f"Operator holder share: {int(primary['rounds'])}/{op_rounds} rounds ({share:.0f}%).",
                (
                    f"{_buy_snapshot_fact('defense', defense)} {_buy_snapshot_fact('attack', attack)}"
                ),
            ]
            if timing_count:
                facts.append(
                    f"Timing: {early_count}/{timing_count} engagements at 0–20 s; "
                    f"{late_count}/{timing_count} after 20 s."
                )
            if aggression_count:
                facts.append(
                    f"Position proxy: {aggressive_count}/{aggression_count} aggressive; "
                    f"{passive_count}/{aggression_count} passive."
                )
            if not timing_count and not aggression_count:
                facts.append(f"Defense engagement sample: {len(engagements)} events; angle profile unavailable.")
            cards.append({
                "type": "Operator",
                "title": f"{primary['player_name']} — {display_side} Operator holder · All maps",
                "tendency": tendency,
                "evidence": (
                    f"{int(primary['rounds'])}/{op_rounds} Operator rounds ({share:.0f}% holder share); "
                    f"buy snapshots: defense {defense.get('held_rounds', 0)}/{defense.get('observed_rounds', 0)}, "
                    f"attack {attack.get('held_rounds', 0)}/{attack.get('observed_rounds', 0)}; "
                    f"{series_count} series."
                ),
                "facts": facts[:4],
                "confidence": confidence,
                "confidence_order": CONFIDENCE_ORDER[confidence],
                "frequency": share,
                "response": advice,
                "rounds": int(primary["rounds"]),
                "total_rounds": int(op_rounds),
                "series_count": series_count,
                "map": "All maps",
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
            confidence = _card_confidence(higher["observed_rounds"], higher["rate"])
            if confidence:
                if self_scout:
                    tendency = (
                        f"Opponents can expect your Operator in {higher['rate']:.0f}% of your {higher_side} rounds "
                        f"versus {lower['rate']:.0f}% on {lower_side} across all maps."
                    )
                    advice = "Consider varying Operator purchase timing between sides."
                else:
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
                    "title": "Operator buy-rate split · All maps",
                    "tendency": tendency,
                    "evidence": (
                        f"{higher['held_rounds']}/{higher['observed_rounds']} {higher_side} snapshots "
                        f"({higher['rate']:.0f}%), {lower['held_rounds']}/{lower['observed_rounds']} "
                        f"{lower_side} snapshots ({lower['rate']:.0f}%), {difference:.0f} points; "
                        f"{series_count} series."
                    ),
                    "facts": [
                        f"{higher['held_rounds']}/{higher['observed_rounds']} {higher_side} buy snapshots ({higher['rate']:.0f}%).",
                        f"{lower['held_rounds']}/{lower['observed_rounds']} {lower_side} buy snapshots ({lower['rate']:.0f}%).",
                        f"{difference:.0f}-point gap across {series_count} series.",
                    ],
                    "confidence": confidence,
                    "confidence_order": CONFIDENCE_ORDER[confidence],
                    "frequency": float(higher["rate"]),
                    "response": advice,
                    "rounds": int(higher["observed_rounds"]),
                    "total_rounds": int(higher["eligible_rounds"]),
                    "series_count": series_count,
                    "map": "All maps",
                    "side": display_side,
                    "navigation_tab": "Operator Profiler",
                })
    return cards


def generate_counter_cards(
    rounds, casts, buys, kills, damage, spawn_positions, team_id, team_name, map_name, side,
    site_centres=None, self_scout=False,
):
    """Build strong local rule-based cards, ordered by confidence and prevalence."""
    return generate_counter_plan(
        rounds, casts, buys, kills, damage, spawn_positions, team_id, team_name, map_name,
        side, site_centres, self_scout,
    )["cards"]


def generate_counter_plan(
    rounds, casts, buys, kills, damage, spawn_positions, team_id, team_name, map_name, side,
    site_centres=None, self_scout=False,
):
    """Build ranked strong tendencies and lower-confidence watch-list entries."""
    site_centres = site_centres or {}
    cards = _utility_cards(rounds, casts, team_id, team_name, map_name, side, site_centres, self_scout)
    cards.extend(_operator_cards(
        rounds, buys, kills, damage, spawn_positions, team_id, team_name, map_name, side,
        site_centres, self_scout,
    ))
    sort_key = lambda card: (card["confidence_order"], -card["frequency"], card["type"], card["tendency"])
    strong = sorted((card for card in cards if card["confidence"] != "Low"), key=sort_key)
    watch_list = sorted((card for card in cards if card["confidence"] == "Low"), key=sort_key)
    return {"cards": strong[:8], "watch_list": watch_list}