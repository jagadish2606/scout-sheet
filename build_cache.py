import argparse
import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from analysis.extract import TABLE_COLUMNS, extract_series
from analysis.landing import (
    add_landing_estimates,
    calibrate_thrown_ranges,
    calibrate_thrown_ranges_batches,
    print_landing_method_counts,
    print_landing_sanity_report,
)
from analysis.map_shape import load_map_shapes
from analysis.operator import estimate_spawn_anchors
from analysis.sites import load_site_centres
from config import DATA_DIR
from analysis.incremental_cache import (
    bootstrap_existing_cache,
    combine_series_shards,
    iter_series_tables,
    read_series_table,
    series_is_complete,
    shard_disk_bytes,
    write_series_casts,
    write_series_shard,
)


CLOUD9_NAMES = {"Cloud9", "Cloud9 GC"}
SAMPLE_COLUMNS = {
    "rounds": ["series_id", "game_id", "game_number", "map", "round_number", "round_start_time", "round_live_time", "team_name", "opponent_name", "side", "won_round"],
    "casts": ["series_id", "game_id", "map", "round_number", "seconds_into_round", "player_name", "team_name", "agent", "ability", "slot", "x", "y", "quality", "landing_x", "landing_y", "landing_method", "estimated"],
    "kills": ["series_id", "game_id", "map", "round_number", "seconds_into_round", "event_time", "killer_name", "killer_team_name", "killer_side", "victim_name", "victim_team_name", "victim_side", "weapon", "killer_x", "killer_y", "victim_x", "victim_y"],
    "damage": ["series_id", "game_id", "map", "round_number", "seconds_into_round", "event_time", "player_name", "team_name", "side", "victim_name", "weapon", "damage_amount", "x", "y"],
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


def all_series_ids(data_dir=DATA_DIR):
    with (Path(data_dir) / "val" / "series.csv").open(newline="", encoding="utf-8") as file:
        return [str(row["series_id"]) for row in csv.DictReader(file) if row.get("series_id")]


def _extract_series_to_shard(series_id, data_dir, shard_root):
    if series_is_complete(shard_root, series_id):
        return str(series_id), 0.0, "cached"
    started = time.perf_counter()
    extracted = extract_series(series_id, data_dir=data_dir)
    write_series_shard(shard_root, series_id, extracted)
    return str(series_id), time.perf_counter() - started, "processed"


def _shape_inputs(series_ids, shard_root):
    cast_chunks, kill_chunks = [], []
    for series_id in series_ids:
        casts = read_series_table(shard_root, series_id, "casts", ["map", "x", "y"])
        kills = read_series_table(
            shard_root, series_id, "kills",
            ["map", "killer_x", "killer_y", "victim_x", "victim_y"],
        )
        if not casts.empty:
            cast_chunks.append(casts)
        if not kills.empty:
            kill_chunks.append(kills)
    cast_points = pd.concat(cast_chunks, ignore_index=True) if cast_chunks else pd.DataFrame(columns=["map", "x", "y"])
    kill_points = pd.concat(kill_chunks, ignore_index=True) if kill_chunks else pd.DataFrame(
        columns=["map", "killer_x", "killer_y", "victim_x", "victim_y"],
    )
    return cast_points, kill_points


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _table_bytes(cache_dir):
    return sum(
        (Path(cache_dir) / f"{table_name}.parquet").stat().st_size
        for table_name in TABLE_COLUMNS
        if (Path(cache_dir) / f"{table_name}.parquet").is_file()
    )


def _save_spawn_estimates(spawn_positions, cache_dir):
    anchors = estimate_spawn_anchors(spawn_positions)
    payload = {
        "source": "spawn_positions.parquet",
        "method": "per-map and per-side median freeze-end player position",
        "source_rows": int(len(spawn_positions)),
        "maps": anchors,
    }
    destination = Path(cache_dir) / "spawn_estimates.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, destination)
    return anchors


def build_all_cache(limit=None, data_dir=DATA_DIR, cache_dir=None, workers=None):
    """Incrementally extract all VALORANT series and publish consolidated Parquet tables."""
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).parent / "cache"
    shard_root = cache_dir / "series_shards"
    all_ids = all_series_ids(data_dir)
    initial_table_bytes = _table_bytes(cache_dir)

    bootstrap_existing_cache(cache_dir, shard_root)
    already_cached = [series_id for series_id in all_ids if series_is_complete(shard_root, series_id)]
    uncached = [series_id for series_id in all_ids if not series_is_complete(shard_root, series_id)]
    requested = uncached[:limit] if limit is not None else uncached
    shard_bytes_before = shard_disk_bytes(shard_root)
    worker_count = max(1, int(workers or min(4, os.cpu_count() or 1)))
    print(
        f"VALORANT series: {len(all_ids):,}; cached: {len(already_cached):,}; "
        f"to process: {len(requested):,}; workers: {worker_count}", flush=True,
    )

    extraction_started = time.perf_counter()
    worker_seconds = []
    failures = []
    if requested:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_extract_series_to_shard, series_id, str(data_dir), str(shard_root)): series_id
                for series_id in requested
            }
            for index, future in enumerate(as_completed(futures), start=1):
                series_id = futures[future]
                try:
                    result_id, elapsed, status = future.result()
                    if status == "processed":
                        worker_seconds.append(elapsed)
                    print(f"[{index}/{len(requested)}] {status}: {result_id} ({elapsed:.1f}s)", flush=True)
                except Exception as error:
                    failures.append((str(series_id), f"{type(error).__name__}: {error}"))
                    print(f"[{index}/{len(requested)}] FAILED: {series_id} — {error}", flush=True)
                completed = index
                elapsed_wall = time.perf_counter() - extraction_started
                remaining = len(requested) - completed
                eta = elapsed_wall * remaining / completed if completed else 0
                print(
                    f"  progress {completed}/{len(requested)} · elapsed {_format_duration(elapsed_wall)} "
                    f"· ETA {_format_duration(eta)}", flush=True,
                )
    extraction_wall = time.perf_counter() - extraction_started

    finalization_started = time.perf_counter()
    valid_ids = [series_id for series_id in all_ids if series_is_complete(shard_root, series_id)]
    shape_casts, shape_kills = _shape_inputs(valid_ids, shard_root)
    shapes = load_map_shapes(shape_casts, shape_kills, cache_dir)

    calibration_started = time.perf_counter()
    thrown_ranges, adjustments = calibrate_thrown_ranges_batches(
        iter_series_tables(valid_ids, shard_root, "casts"), shapes,
    )
    for series_id in valid_ids:
        casts = read_series_table(shard_root, series_id, "casts")
        if casts.empty:
            continue
        write_series_casts(shard_root, series_id, add_landing_estimates(casts, shapes, thrown_ranges))
    calibration_elapsed = time.perf_counter() - calibration_started

    row_counts, byte_counts = combine_series_shards(valid_ids, shard_root, cache_dir)
    plants = pd.read_parquet(cache_dir / "plants.parquet")
    site_centres = load_site_centres(plants, cache_dir)
    spawn_positions = pd.read_parquet(cache_dir / "spawn_positions.parquet")
    spawn_anchors = _save_spawn_estimates(spawn_positions, cache_dir)
    finalization_elapsed = time.perf_counter() - finalization_started
    print("Final table rows:")
    for table_name, row_count in row_counts.items():
        print(f"  {table_name}: {row_count:,} ({byte_counts[table_name] / (1024 ** 2):.1f} MiB)")
    print(f"Site centres: {sum(len(sites) for sites in site_centres.values()):,} across {len(site_centres):,} maps")
    print(f"Full-data spawn estimates: {len(spawn_anchors):,} maps -> {cache_dir / 'spawn_estimates.json'}")
    print(f"Completed series: {len(valid_ids):,}/{len(all_ids):,}; failed this run: {len(failures):,}")
    for series_id, reason in failures:
        print(f"  {series_id}: {reason}")
    for ability, adjustment in adjustments.items():
        print(
            f"  Range calibration {ability}: {adjustment['old_maximum']} → {adjustment['new_maximum']} "
            f"({adjustment['max_hits']}/{adjustment['valid_rays']} reached max)",
        )

    processed_count = len(worker_seconds)
    if processed_count:
        average_worker = sum(worker_seconds) / processed_count
        remaining_after = max(0, len(all_ids) - len(valid_ids))
        per_series_finalize = finalization_elapsed / max(len(valid_ids), 1)
        full_estimate = len(all_ids) * (average_worker / worker_count + per_series_finalize)
        remaining_estimate = remaining_after * (average_worker / worker_count + per_series_finalize)
        print(f"Mean extraction/write time: {average_worker:.2f}s per series ({worker_count} workers)")
        print(f"Extraction throughput: {extraction_wall / processed_count:.2f}s wall time per series")
        print(f"Estimated {len(all_ids):,}-series wall time at this rate: {_format_duration(full_estimate)}")
        print(f"Estimated remaining after this run: {_format_duration(remaining_estimate)}")

    final_table_bytes = sum(byte_counts.values())
    new_shard_bytes = max(0, shard_disk_bytes(shard_root) - shard_bytes_before)
    new_series_count = len([series_id for series_id in requested if series_is_complete(shard_root, series_id)])
    remaining_after = max(0, len(all_ids) - len(valid_ids))
    table_growth = max(0, final_table_bytes - initial_table_bytes)
    table_bytes_per_series = (
        table_growth / new_series_count if new_series_count
        else final_table_bytes / max(len(valid_ids), 1)
    )
    shard_bytes_per_series = (
        new_shard_bytes / new_series_count if new_series_count
        else shard_disk_bytes(shard_root) / max(len(valid_ids), 1)
    )
    projected_table_bytes = final_table_bytes + table_bytes_per_series * remaining_after
    projected_shard_bytes = shard_disk_bytes(shard_root) + shard_bytes_per_series * remaining_after
    print(
        "Estimated full cache disk size (consolidated tables + resumable shards): "
        f"{(projected_table_bytes + projected_shard_bytes) / (1024 ** 3):.2f} GiB "
        f"(tables {projected_table_bytes / (1024 ** 3):.2f} GiB; shards "
        f"{projected_shard_bytes / (1024 ** 3):.2f} GiB; excludes small metadata files).",
    )
    print(
        f"Extraction phase: {_format_duration(extraction_wall)}; landing recalibration: "
        f"{_format_duration(calibration_elapsed)}; full finalization: {_format_duration(finalization_elapsed)}",
    )
    return {
        "series_total": len(all_ids), "series_cached": len(valid_ids), "processed": new_series_count,
        "failures": failures, "rows": row_counts, "worker_seconds": worker_seconds,
        "extraction_wall": extraction_wall, "finalization_seconds": finalization_elapsed,
        "table_bytes": final_table_bytes, "shard_bytes": shard_disk_bytes(shard_root),
        "projected_table_bytes": projected_table_bytes,
        "projected_shard_bytes": projected_shard_bytes,
    }


def build_cache(series_ids=None, limit=None, data_dir=DATA_DIR, cache_dir=None, sample_rows=5):
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).parent / "cache"
    previous_casts_path = cache_dir / "casts.parquet"
    previous_casts = pd.read_parquet(previous_casts_path) if previous_casts_path.is_file() else None
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
        final[table_name] = frame

    shapes = load_map_shapes(final["casts"], final["kills"], cache_dir)
    thrown_ranges, adjustments = calibrate_thrown_ranges(final["casts"], shapes)
    final["casts"] = add_landing_estimates(final["casts"], shapes, thrown_ranges)
    if previous_casts is not None:
        print_landing_method_counts(previous_casts, "BEFORE landing method counts by ability (existing cache):")
    print_landing_method_counts(final["casts"], "AFTER landing method counts by ability:")
    print_landing_sanity_report(final["casts"], shapes, thrown_ranges, adjustments)

    for table_name, frame in final.items():
        path = cache_dir / f"{table_name}.parquet"
        frame.to_parquet(path, index=False)
        print(f"{table_name}: {len(frame):,} rows -> {path}")
        if sample_rows > 0 and not frame.empty:
            sample_columns = [column for column in SAMPLE_COLUMNS[table_name] if column in frame]
            print(f"Sample {table_name} (first {min(sample_rows, len(frame))} rows):")
            print(frame[sample_columns].head(sample_rows).to_string(index=False, max_colwidth=22))
    site_centres = load_site_centres(final["plants"], cache_dir)
    _save_spawn_estimates(final["spawn_positions"], cache_dir)
    print(f"Site centres: {sum(len(sites) for sites in site_centres.values()):,} across {len(site_centres):,} maps")
    return final


def main():
    parser = argparse.ArgumentParser(description="Build ignored Parquet caches from local GRID VALORANT feeds.")
    parser.add_argument("--all", action="store_true", help="Process all VALORANT series incrementally with worker processes.")
    parser.add_argument("--series-ids", nargs="+", help="Series IDs to process instead of the default Cloud9/opponent selection.")
    parser.add_argument("--limit", type=int, help="Process at most this many selected series.")
    parser.add_argument("--workers", type=int, help="Worker processes for --all (default: up to 4).")
    parser.add_argument("--sample-rows", type=int, default=5, help="Print this many sample rows per table (default: 5; use 0 to suppress).")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory (default: ./cache).")
    args = parser.parse_args()
    if args.all:
        build_all_cache(args.limit, cache_dir=args.cache_dir, workers=args.workers)
        return
    build_cache(args.series_ids, args.limit, cache_dir=args.cache_dir, sample_rows=args.sample_rows)


if __name__ == "__main__":
    main()