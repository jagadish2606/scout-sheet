import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from analysis.extract import TABLE_COLUMNS, extract_series
from analysis.sites import load_site_centres
from config import DATA_DIR


CLOUD9_NAMES = {"Cloud9", "Cloud9 GC"}
SAMPLE_COLUMNS = {
    "rounds": ["series_id", "game_id", "map", "round_number", "round_start_time", "team_name", "side", "won_round"],
    "casts": ["series_id", "game_id", "map", "round_number", "seconds_into_round", "player_name", "team_name", "agent", "ability", "slot", "x", "y", "quality"],
    "kills": ["series_id", "game_id", "map", "round_number", "seconds_into_round", "killer_name", "killer_team_name", "killer_side", "victim_name", "victim_team_name", "victim_side", "weapon", "killer_x", "killer_y", "victim_x", "victim_y"],
    "damage": ["series_id", "game_id", "map", "round_number", "seconds_into_round", "player_name", "team_name", "side", "victim_name", "weapon", "damage_amount", "x", "y"],
    "buys": ["series_id", "game_id", "map", "round_number", "player_name", "team_id", "side", "agent", "primary_weapon", "has_operator"],
    "spawn_positions": ["series_id", "game_id", "map", "round_number", "player_name", "team_name", "side", "agent", "x", "y"],
    "plants": ["series_id", "game_id", "map", "round_number", "planter_id", "team_name", "x", "y"],
}


def select_series_ids(data_dir=DATA_DIR):
    """Select Cloud9 series and same-event series featuring their opponents."""
    data_dir = Path(data_dir)
    with (data_dir / "val" / "series.csv").open(newline="", encoding="utf-8") as file:
        series_rows = list(csv.DictReader(file))
    with (data_dir / "meta" / "games.csv").open(newline="", encoding="utf-8") as file:
        game_rows = [row for row in csv.DictReader(file) if row.get("title") == "val"]

    games_by_series = defaultdict(list)
    for game in game_rows:
        games_by_series[str(game["series_id"])].append(game)

    selected = set()
    opponents_by_event = defaultdict(set)
    event_by_series = {}
    for row in series_rows:
        series_id = str(row["series_id"])
        event = row.get("event", "")
        event_by_series[series_id] = event
        team_names = {name.strip() for name in row.get("teams", "").split(" vs ")}
        cloud9_names = team_names & CLOUD9_NAMES
        if not cloud9_names:
            continue
        selected.add(series_id)
        for game in games_by_series.get(series_id, []):
            c9_team_ids = {
                str(game[f"team{number}_id"])
                for number in (1, 2)
                if game.get(f"team{number}_name") in cloud9_names and game.get(f"team{number}_id")
            }
            opponents_by_event[event].update(
                str(game[f"team{number}_id"])
                for number in (1, 2)
                if game.get(f"team{number}_id") and str(game[f"team{number}_id"]) not in c9_team_ids
            )

    for row in series_rows:
        series_id = str(row["series_id"])
        event = event_by_series[series_id]
        opponent_ids = opponents_by_event.get(event, set())
        if any(
            str(game.get(f"team{number}_id")) in opponent_ids
            for game in games_by_series.get(series_id, [])
            for number in (1, 2)
        ):
            selected.add(series_id)

    return [str(row["series_id"]) for row in series_rows if str(row["series_id"]) in selected]


def build_cache(series_ids=None, limit=None, data_dir=DATA_DIR, cache_dir=None, sample_rows=5):
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).parent / "cache"
    if series_ids is None:
        series_ids = select_series_ids(data_dir)
    else:
        series_ids = [str(series_id) for series_id in series_ids]
    if limit is not None:
        series_ids = series_ids[:limit]

    combined = {name: [] for name in TABLE_COLUMNS}
    for index, series_id in enumerate(series_ids, start=1):
        print(f"[{index}/{len(series_ids)}] Extracting series {series_id}", flush=True)
        extracted = extract_series(series_id, data_dir=data_dir)
        for table_name, frame in extracted.items():
            combined[table_name].append(frame)

    cache_dir.mkdir(parents=True, exist_ok=True)
    final = {}
    for table_name, columns in TABLE_COLUMNS.items():
        frames = combined[table_name]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        path = cache_dir / f"{table_name}.parquet"
        frame.to_parquet(path, index=False)
        final[table_name] = frame
        print(f"{table_name}: {len(frame):,} rows -> {path}")
        if sample_rows > 0 and not frame.empty:
            sample_columns = [column for column in SAMPLE_COLUMNS[table_name] if column in frame]
            print(f"Sample {table_name} (first {min(sample_rows, len(frame))} rows):")
            print(frame[sample_columns].head(sample_rows).to_string(index=False, max_colwidth=22))
    site_centres = load_site_centres(final["plants"], cache_dir)
    print(f"Site centres: {sum(len(sites) for sites in site_centres.values()):,} across {len(site_centres):,} maps")
    return final


def main():
    parser = argparse.ArgumentParser(description="Build ignored Parquet caches from local GRID VALORANT feeds.")
    parser.add_argument("--series-ids", nargs="+", help="Series IDs to process instead of the default Cloud9/opponent selection.")
    parser.add_argument("--limit", type=int, help="Process at most this many selected series.")
    parser.add_argument("--sample-rows", type=int, default=5, help="Print this many sample rows per table (default: 5; use 0 to suppress).")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory (default: ./cache).")
    args = parser.parse_args()
    build_cache(args.series_ids, args.limit, cache_dir=args.cache_dir, sample_rows=args.sample_rows)


if __name__ == "__main__":
    main()