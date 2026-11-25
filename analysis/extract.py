import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DATA_DIR, read_jsonl_zst


TABLE_COLUMNS = {
    "rounds": [
        "series_id", "game_id", "map", "round_number", "round_start_time",
        "team_id", "team_name", "side", "won_round",
    ],
    "casts": [
        "series_id", "game_id", "map", "round_number", "seconds_into_round",
        "player_id", "player_name", "team_id", "team_name", "side", "agent",
        "ability", "slot", "x", "y", "z", "view_x", "view_y", "view_z", "quality",
    ],
    "kills": [
        "series_id", "game_id", "map", "round_number", "seconds_into_round",
        "killer_id", "killer_name", "killer_team_id", "killer_team_name", "killer_side",
        "killer_agent", "victim_id", "victim_name", "victim_team_id", "victim_team_name",
        "victim_side", "victim_agent", "weapon", "killer_x", "killer_y", "victim_x", "victim_y",
    ],
    "buys": [
        "series_id", "game_id", "map", "round_number", "player_id", "player_name",
        "team_id", "side", "agent", "primary_weapon", "has_operator",
    ],
    "plants": [
        "series_id", "game_id", "map", "round_number", "planter_id", "team_id",
        "team_name", "x", "y",
    ],
}

PRIMARY_WEAPONS = {
    "ares", "bulldog", "bucky", "guardian", "judge", "marshal", "odin",
    "operator", "outlaw", "phantom", "spectre", "stinger", "vandal",
}


def _is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_game_metadata(series_id, data_dir):
    path = Path(data_dir) / "meta" / "games.csv"
    games = {}
    sessions = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("title") != "val" or str(row.get("series_id")) != str(series_id):
                continue
            valid = _is_true(row.get("session_valid", "true"))
            session = str(row.get("session", "1"))
            sessions[session].append(valid)
            games[str(row["game_id"])] = {
                "map": row.get("map") or None,
                "session": session,
                "valid": valid,
            }
    valid_sessions = {
        session for session, values in sessions.items() if values and all(values)
    }
    return games, valid_sessions


def _round_number(value):
    if value is None:
        return None
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def _timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _elapsed_seconds(started_at, occurred_at):
    start = _timestamp(started_at)
    end = _timestamp(occurred_at)
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _character_name(character):
    if isinstance(character, dict):
        return character.get("name") or character.get("id")
    return character


def _update_player(players, game_id, player, team_id=None, team_name=None, side=None):
    if not isinstance(player, dict):
        return
    player_id = player.get("id")
    if player_id is None:
        return
    key = (str(game_id), str(player_id))
    existing = players.setdefault(key, {})
    values = {
        "player_name": player.get("name"),
        "team_id": player.get("teamId") or team_id,
        "team_name": team_name,
        "side": player.get("side") or side,
        "agent": _character_name(player.get("character")),
    }
    for name, value in values.items():
        if value is not None:
            existing[name] = str(value) if name.endswith("_id") else value


def _update_game_context(state, game_context, players):
    if not isinstance(state, dict):
        return
    for game in state.get("games", []) or []:
        game_id = game.get("id")
        if game_id is None:
            continue
        game_id = str(game_id)
        context = game_context.setdefault(game_id, {"map": None, "teams": {}})
        map_info = game.get("map") or {}
        if isinstance(map_info, dict) and map_info.get("name"):
            context["map"] = map_info["name"]
        for team in game.get("teams", []) or []:
            team_id = team.get("id")
            if team_id is None:
                continue
            team_id = str(team_id)
            team_info = context["teams"].setdefault(team_id, {})
            if team.get("name") is not None:
                team_info["name"] = team["name"]
            if team.get("side") is not None:
                team_info["side"] = team["side"]
            for player in team.get("players", []) or []:
                _update_player(
                    players, game_id, player, team_id=team_id,
                    team_name=team_info.get("name"), side=team_info.get("side"),
                )


def _event_game_candidates(event):
    candidates = []
    for name in ("actor", "target"):
        entity = event.get(name) or {}
        if str(entity.get("type", "")).lower() in {"game", "esports"} and entity.get("id") is not None:
            candidates.append(str(entity["id"]))
    for state_name in ("seriesState", "seriesStateDelta"):
        state = event.get(state_name) or {}
        for game in state.get("games", []) or []:
            if game.get("id") is not None:
                candidates.append(str(game["id"]))
    return candidates


def _event_game_id(event, known_game_ids, active_game_id=None):
    direct_candidates = [
        str(entity["id"])
        for name in ("actor", "target")
        if (entity := event.get(name) or {}).get("id") is not None
        and str(entity.get("type", "")).lower() in {"game", "esports"}
    ]
    for candidate in direct_candidates:
        if candidate in known_game_ids:
            return candidate
    if direct_candidates:
        return direct_candidates[0]
    if active_game_id is not None:
        return active_game_id
    for candidate in _event_game_candidates(event):
        if candidate in known_game_ids:
            return candidate
    return None


def _rollback_game_id(event):
    for name in ("target", "actor"):
        entity = event.get(name) or {}
        if entity.get("id") is not None and str(entity.get("type", "")).lower() in {"game", "esports"}:
            return str(entity["id"])
    for candidate in _event_game_candidates(event):
        return candidate
    return None


def _event_round_number(event, game_id, active_rounds):
    for name in ("target", "actor"):
        entity = event.get(name) or {}
        if str(entity.get("type", "")).lower() == "round":
            number = _round_number(entity.get("id"))
            if number is not None:
                return number
            state_number = (entity.get("state") or {}).get("sequenceNumber")
            if state_number is not None:
                return int(state_number)
        state = entity.get("state") or {}
        state_number = (state.get("round") or {}).get("sequenceNumber")
        if state_number is not None:
            return int(state_number)
    for state_name in ("seriesState", "seriesStateDelta"):
        for game in (event.get(state_name) or {}).get("games", []) or []:
            if str(game.get("id")) != str(game_id):
                continue
            for segment in reversed(game.get("segments", []) or []):
                if segment.get("type") == "round" or str(segment.get("id", "")).startswith("round-"):
                    sequence_number = segment.get("sequenceNumber")
                    if sequence_number is not None:
                        return int(sequence_number)
                    number = _round_number(segment.get("id"))
                    if number is not None and str(segment.get("id", "")).startswith("round-"):
                        return number
    return active_rounds.get(str(game_id))


def _event_player_state(entity):
    return (entity or {}).get("state") or {}


def _player_fields(entity, game_id, team_context, players):
    state = _event_player_state(entity)
    player_id = (entity or {}).get("id") or state.get("id")
    if player_id is None:
        return None, {}
    player_id = str(player_id)
    player = players.get((str(game_id), player_id), {}).copy()
    team_id = state.get("teamId") or player.get("team_id")
    if team_id is not None:
        team_id = str(team_id)
        team = team_context.get("teams", {}).get(team_id, {})
        player.update({
            "team_id": team_id,
            "team_name": team.get("name") or player.get("team_name"),
            "side": state.get("side") or team.get("side") or player.get("side"),
        })
    if state.get("name") is not None:
        player["player_name"] = state["name"]
    if state.get("side") is not None:
        player["side"] = state["side"]
    if state.get("character") is not None:
        player["agent"] = _character_name(state["character"])
    if player_id not in {None, ""}:
        player["player_id"] = player_id
    return player_id, player


def _weapon_from_kill(actor, game_id, weapon_counts):
    state = _event_player_state(actor)
    state_game = state.get("game") or {}
    delta_game = ((actor or {}).get("stateDelta") or {}).get("game") or {}
    counts = delta_game.get("weaponKills")
    if not isinstance(counts, dict):
        counts = state_game.get("weaponKills")
    if not isinstance(counts, dict):
        return None
    normalized = {str(key): value for key, value in counts.items() if isinstance(value, (int, float))}
    player_id = str((actor or {}).get("id") or state.get("id") or "")
    key = (str(game_id), player_id)
    previous = weapon_counts.get(key, {})
    changed = [
        (weapon, count - previous.get(weapon, 0))
        for weapon, count in normalized.items()
        if count > previous.get(weapon, 0)
    ]
    weapon_counts[key] = normalized
    if len(changed) == 1:
        return changed[0][0]
    if changed:
        return max(changed, key=lambda pair: pair[1])[0]
    if len(normalized) == 1:
        return next(iter(normalized))
    return None


def _frame_location(record):
    quality = record.get("quality")
    if quality == "gap_exceeded":
        frames = [frame for frame in (record.get("frame_before"), record.get("frame_after")) if isinstance(frame, dict)]
        frames = [frame for frame in frames if isinstance(frame.get("position"), dict)]
        if not frames:
            return None, None
        frame = min(
            frames,
            key=lambda item: abs(item["offset_ms"]) if isinstance(item.get("offset_ms"), (int, float)) else float("inf"),
        )
        return frame.get("position"), frame.get("view")
    interpolated = record.get("interpolated")
    if not isinstance(interpolated, dict):
        return None, None
    return interpolated.get("position"), interpolated.get("view")


def _coords(position, view):
    position = position if isinstance(position, dict) else {}
    view = view if isinstance(view, dict) else {}
    return {
        "x": position.get("x"), "y": position.get("y"), "z": position.get("z"),
        "view_x": view.get("x"), "view_y": view.get("y"), "view_z": view.get("z"),
    }


def _game_map(game_context, game_id, metadata):
    return game_context.get(str(game_id), {}).get("map") or metadata.get(str(game_id), {}).get("map")


def extract_series(series_id, data_dir=DATA_DIR):
    """Stream one GRID series feed into rounds, ability casts, kills and buys tables."""
    series_id = str(series_id)
    data_dir = Path(data_dir)
    game_metadata, valid_sessions = _load_game_metadata(series_id, data_dir)
    valid_game_ids = {game_id for game_id, info in game_metadata.items() if info["valid"]}
    if not valid_game_ids:
        return {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}

    supplement_path = data_dir / "val-ability-supplements" / f"{series_id}.jsonl.zst"
    supplements = {}
    if supplement_path.is_file():
        for record in read_jsonl_zst(supplement_path):
            event_id = record.get("grid_event_id")
            if event_id and record.get("in_grid"):
                supplements[str(event_id)] = record

    rows = {name: [] for name in TABLE_COLUMNS}
    game_context = {}
    players = {}
    active_rounds = {}
    round_live_at = {}
    round_indices = defaultdict(list)
    round_row_indices = {}
    weapon_counts = {}
    rolled_back_game_ids = set()
    active_game_by_session = {}
    session_number = 0
    feed_path = data_dir / "val" / f"events_{series_id}_grid.jsonl.zst"

    for envelope in read_jsonl_zst(feed_path):
        occurred_at = envelope.get("occurredAt")
        for event in envelope.get("events", []) or []:
            event_type = event.get("type", "")
            if event_type == "grid-started-feed":
                session_number = 1 if session_number == 0 else session_number + 1
                active_game_by_session.pop(session_number, None)
                continue
            if session_number == 0:
                session_number = 1

            if event_type == "series-!started-game":
                rollback_id = _rollback_game_id(event)
                if rollback_id:
                    rolled_back_game_ids.add(rollback_id)
                    if active_game_by_session.get(session_number) == rollback_id:
                        active_game_by_session.pop(session_number, None)
                continue

            session_key = str(session_number)
            if session_key not in valid_sessions:
                continue
            game_id = _event_game_id(
                event, game_metadata, active_game_by_session.get(session_number),
            )
            if game_id is None:
                continue
            game_id = str(game_id)
            active_game_by_session[session_number] = game_id
            if game_id not in valid_game_ids:
                continue
            state = event.get("seriesState") or {}
            _update_game_context(state, game_context, players)
            _update_game_context(event.get("seriesStateDelta") or {}, game_context, players)

            for entity_name in ("actor", "target"):
                entity = event.get(entity_name) or {}
                entity_state = _event_player_state(entity)
                if entity_state.get("teamId") is not None and entity_state.get("id") is not None:
                    _update_player(players, game_id, entity_state)

            if event_type == "game-started-round":
                round_number = None
                target = event.get("target") or {}
                round_number = _round_number(target.get("id"))
                game_state = next(
                    (game for game in state.get("games", []) or [] if str(game.get("id")) == game_id),
                    None,
                )
                segment = None
                if game_state:
                    for candidate in reversed(game_state.get("segments", []) or []):
                        if candidate.get("type") == "round" or str(candidate.get("id", "")).startswith("round-"):
                            segment = candidate
                            break
                if round_number is None and segment:
                    round_number = _round_number(segment.get("id"))
                if round_number is None:
                    continue
                started_at = (
                    (segment or {}).get("startedAt")
                    or ((target.get("state") or {}).get("startedAt"))
                    or occurred_at
                )
                active_rounds[game_id] = round_number
                if game_state:
                    context = game_context.setdefault(game_id, {"map": None, "teams": {}})
                    for team in game_state.get("teams", []) or []:
                        team_id = team.get("id")
                        if team_id is None:
                            continue
                        team_id = str(team_id)
                        team_info = context["teams"].setdefault(team_id, {})
                        team_info.update({k: team[k] for k in ("name", "side") if team.get(k) is not None})
                        row_key = (game_id, round_number, team_id)
                        row = {
                            "series_id": series_id,
                            "game_id": game_id,
                            "map": _game_map(game_context, game_id, game_metadata),
                            "round_number": round_number,
                            "round_start_time": started_at,
                            "team_id": team_id,
                            "team_name": team_info.get("name"),
                            "side": team_info.get("side"),
                            "won_round": None,
                        }
                        if row_key in round_row_indices:
                            rows["rounds"][round_row_indices[row_key]].update(row)
                        else:
                            index = len(rows["rounds"])
                            rows["rounds"].append(row)
                            round_row_indices[row_key] = index
                            round_indices[(game_id, round_number)].append(index)
                continue

            round_number = _event_round_number(event, game_id, active_rounds)
            if round_number is not None:
                active_rounds[game_id] = round_number

            if event_type == "team-won-round":
                actor = event.get("actor") or {}
                winner_id = actor.get("id") or _event_player_state(actor).get("id")
                winner_id = str(winner_id) if winner_id is not None else None
                for index in round_indices.get((game_id, round_number), []):
                    rows["rounds"][index]["won_round"] = rows["rounds"][index]["team_id"] == winner_id
                continue

            if event_type == "round-ended-freezetime":
                if round_number is not None and occurred_at is not None:
                    round_live_at[(game_id, round_number)] = occurred_at
                if not event.get("includesFullState"):
                    continue
                for game in state.get("games", []) or []:
                    if str(game.get("id")) != game_id:
                        continue
                    round_number = _event_round_number(event, game_id, active_rounds)
                    for team in game.get("teams", []) or []:
                        team_id = str(team.get("id")) if team.get("id") is not None else None
                        side = team.get("side")
                        context = game_context.setdefault(game_id, {"map": None, "teams": {}})
                        team_info = context["teams"].get(team_id, {}) if team_id else {}
                        team_name = team.get("name") or team_info.get("name")
                        for player in team.get("players", []) or []:
                            _update_player(players, game_id, player, team_id, team_name, side)
                            player_id = str(player.get("id")) if player.get("id") is not None else None
                            if player_id is None:
                                continue
                            inventory = player.get("inventory") or {}
                            items = inventory.get("items", []) if isinstance(inventory, dict) else []
                            item_ids = [str(item.get("id", "")).lower() for item in items]
                            primary = next((weapon for weapon in item_ids if weapon in PRIMARY_WEAPONS), None)
                            row = {
                                "series_id": series_id,
                                "game_id": game_id,
                                "map": _game_map(game_context, game_id, game_metadata),
                                "round_number": round_number,
                                "player_id": player_id,
                                "player_name": player.get("name"),
                                "team_id": team_id,
                                "side": side,
                                "agent": _character_name(player.get("character")),
                                "primary_weapon": primary,
                                "has_operator": "operator" in item_ids,
                            }
                            rows["buys"].append(row)
                continue

            if event_type == "player-used-ability":
                actor = event.get("actor") or {}
                player_id, player = _player_fields(actor, game_id, game_context.get(game_id, {}), players)
                event_id = event.get("id")
                supplement = supplements.get(str(event_id)) if event_id is not None else None
                position, view = _frame_location(supplement) if supplement else (None, None)
                coords = _coords(position, view)
                target_state = _event_player_state(event.get("target"))
                row = {
                    "series_id": series_id,
                    "game_id": game_id,
                    "map": _game_map(game_context, game_id, game_metadata),
                    "round_number": round_number,
                    "_occurred_at": occurred_at,
                    "player_id": player_id,
                    "player_name": player.get("player_name"),
                    "team_id": player.get("team_id"),
                    "team_name": player.get("team_name"),
                    "side": player.get("side"),
                    "agent": player.get("agent"),
                    "ability": (supplement or {}).get("ability") or target_state.get("name") or target_state.get("id"),
                    "slot": (supplement or {}).get("slot"),
                    **coords,
                    "quality": (supplement or {}).get("quality"),
                }
                rows["casts"].append(row)
                continue

            if event_type == "player-completed-plantBomb":
                actor = event.get("actor") or {}
                planter_id, planter = _player_fields(actor, game_id, game_context.get(game_id, {}), players)
                actor_state = _event_player_state(actor)
                game_state = actor_state.get("game") or {}
                position = game_state.get("position") or actor_state.get("position") or {}
                rows["plants"].append({
                    "series_id": series_id,
                    "game_id": game_id,
                    "map": _game_map(game_context, game_id, game_metadata),
                    "round_number": round_number,
                    "planter_id": planter_id,
                    "team_id": planter.get("team_id"),
                    "team_name": planter.get("team_name"),
                    "x": position.get("x"),
                    "y": position.get("y"),
                })
                continue

            if event_type == "player-killed-player":
                actor = event.get("actor") or {}
                target = event.get("target") or {}
                killer_id, killer = _player_fields(actor, game_id, game_context.get(game_id, {}), players)
                victim_id, victim = _player_fields(target, game_id, game_context.get(game_id, {}), players)
                killer_state = _event_player_state(actor).get("game") or {}
                victim_state = _event_player_state(target).get("game") or {}
                killer_position = killer_state.get("position") or {}
                victim_position = victim_state.get("position") or {}
                rows["kills"].append({
                    "series_id": series_id,
                    "game_id": game_id,
                    "map": _game_map(game_context, game_id, game_metadata),
                    "round_number": round_number,
                    "_occurred_at": occurred_at,
                    "killer_id": killer_id,
                    "killer_name": killer.get("player_name"),
                    "killer_team_id": killer.get("team_id"),
                    "killer_team_name": killer.get("team_name"),
                    "killer_side": killer.get("side"),
                    "killer_agent": killer.get("agent"),
                    "victim_id": victim_id,
                    "victim_name": victim.get("player_name"),
                    "victim_team_id": victim.get("team_id"),
                    "victim_team_name": victim.get("team_name"),
                    "victim_side": victim.get("side"),
                    "victim_agent": victim.get("agent"),
                    "weapon": _weapon_from_kill(actor, game_id, weapon_counts),
                    "killer_x": killer_position.get("x"),
                    "killer_y": killer_position.get("y"),
                    "victim_x": victim_position.get("x"),
                    "victim_y": victim_position.get("y"),
                })

    for table_name in ("casts", "kills"):
        for row in rows[table_name]:
            row["seconds_into_round"] = _elapsed_seconds(
                round_live_at.get((str(row["game_id"]), row["round_number"])),
                row.pop("_occurred_at", None),
            )

    result = {}
    for name, columns in TABLE_COLUMNS.items():
        frame = pd.DataFrame(rows[name], columns=columns)
        if rolled_back_game_ids and not frame.empty:
            frame = frame[~frame["game_id"].isin(rolled_back_game_ids)].reset_index(drop=True)
        result[name] = frame
    return result