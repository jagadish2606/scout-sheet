"""Per-series Parquet shards and bounded-memory cache assembly."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.extract import TABLE_COLUMNS


STRING_COLUMNS = {
    "series_id", "game_id", "map", "round_start_time", "round_live_time", "team_id",
    "team_name", "opponent_name", "side", "player_id", "player_name", "agent", "ability",
    "slot", "quality", "landing_method", "event_time", "killer_id", "killer_name",
    "killer_team_id", "killer_team_name", "killer_side", "killer_agent", "victim_id",
    "victim_name", "victim_team_id", "victim_team_name", "victim_side", "victim_agent",
    "weapon", "primary_weapon", "planter_id",
}
INTEGER_COLUMNS = {"game_number", "round_number"}
BOOLEAN_COLUMNS = {"won_round", "estimated", "has_operator"}


def table_schema(table_name):
    fields = []
    for column in TABLE_COLUMNS[table_name]:
        if column in STRING_COLUMNS:
            data_type = pa.string()
        elif column in INTEGER_COLUMNS:
            data_type = pa.int64()
        elif column in BOOLEAN_COLUMNS:
            data_type = pa.bool_()
        else:
            data_type = pa.float64()
        fields.append(pa.field(column, data_type, nullable=True))
    return pa.schema(fields)


def normalize_frame(table_name, frame):
    """Normalize worker output so shards from different series share one schema."""
    result = frame.reindex(columns=TABLE_COLUMNS[table_name]).copy()
    for column in result:
        if column in STRING_COLUMNS:
            result[column] = result[column].map(
                lambda value: None if pd.isna(value) else str(value),
            ).astype(object)
        elif column in INTEGER_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
        elif column in BOOLEAN_COLUMNS:
            result[column] = result[column].astype("boolean")
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    return result


def _write_frame(path, table_name, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(
        normalize_frame(table_name, frame), schema=table_schema(table_name),
        preserve_index=False, safe=False,
    )
    pq.write_table(table, temporary_path, compression="zstd")
    os.replace(temporary_path, path)


def series_shard_dir(shard_root, series_id):
    return Path(shard_root) / str(series_id)


def series_is_complete(shard_root, series_id):
    return (series_shard_dir(shard_root, series_id) / ".complete").is_file()


def write_series_shard(shard_root, series_id, tables):
    """Write all tables first and publish the completion marker last."""
    shard_dir = series_shard_dir(shard_root, series_id)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for table_name in TABLE_COLUMNS:
        frame = tables.get(table_name, pd.DataFrame(columns=TABLE_COLUMNS[table_name]))
        _write_frame(shard_dir / f"{table_name}.parquet", table_name, frame)
    marker = shard_dir / ".complete.tmp"
    marker.write_text("complete\n", encoding="utf-8")
    os.replace(marker, shard_dir / ".complete")


def write_series_casts(shard_root, series_id, casts):
    """Atomically replace a series' cast shard after landing estimates are added."""
    shard_dir = series_shard_dir(shard_root, series_id)
    _write_frame(shard_dir / "casts.parquet", "casts", casts)


def bootstrap_existing_cache(cache_dir, shard_root):
    """Split the existing consolidated cache into shards once, preserving its rows."""
    cache_dir, shard_root = Path(cache_dir), Path(shard_root)
    series_ids = set()
    for table_name in TABLE_COLUMNS:
        path = cache_dir / f"{table_name}.parquet"
        if path.is_file():
            ids = pd.read_parquet(path, columns=["series_id"])["series_id"].dropna().astype(str)
            series_ids.update(ids.unique())
    pending = {series_id for series_id in series_ids if not series_is_complete(shard_root, series_id)}
    if not pending:
        return series_ids

    for table_name in TABLE_COLUMNS:
        path = cache_dir / f"{table_name}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=TABLE_COLUMNS[table_name])
        if frame.empty:
            continue
        frame["series_id"] = frame["series_id"].astype(str)
        for series_id, group in frame[frame["series_id"].isin(pending)].groupby("series_id", sort=False):
            _write_frame(series_shard_dir(shard_root, series_id) / f"{table_name}.parquet", table_name, group)

    for series_id in pending:
        shard_dir = series_shard_dir(shard_root, series_id)
        for table_name, columns in TABLE_COLUMNS.items():
            path = shard_dir / f"{table_name}.parquet"
            if not path.is_file():
                _write_frame(path, table_name, pd.DataFrame(columns=columns))
        marker = shard_dir / ".complete.tmp"
        marker.write_text("bootstrapped\n", encoding="utf-8")
        os.replace(marker, shard_dir / ".complete")
    return series_ids


def read_series_table(shard_root, series_id, table_name, columns=None):
    path = series_shard_dir(shard_root, series_id) / f"{table_name}.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=columns or TABLE_COLUMNS[table_name])
    return pd.read_parquet(path, columns=columns)


def combine_series_shards(series_ids, shard_root, cache_dir):
    """Merge ordered per-series shards using Arrow row batches rather than table concatenation."""
    shard_root, cache_dir = Path(shard_root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    row_counts, byte_counts = {}, {}
    for table_name in TABLE_COLUMNS:
        destination = cache_dir / f"{table_name}.parquet"
        temporary = cache_dir / f".{table_name}.parquet.tmp"
        schema = table_schema(table_name)
        row_count = 0
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for series_id in series_ids:
                if not series_is_complete(shard_root, series_id):
                    continue
                path = series_shard_dir(shard_root, series_id) / f"{table_name}.parquet"
                if not path.is_file():
                    continue
                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(batch_size=32_768):
                    table = pa.Table.from_batches([batch]).cast(schema, safe=False)
                    writer.write_table(table)
                    row_count += table.num_rows
        os.replace(temporary, destination)
        row_counts[table_name] = row_count
        byte_counts[table_name] = destination.stat().st_size
    return row_counts, byte_counts


def iter_series_tables(series_ids, shard_root, table_name, columns=None):
    for series_id in series_ids:
        if series_is_complete(shard_root, series_id):
            yield read_series_table(shard_root, series_id, table_name, columns)


def shard_disk_bytes(shard_root):
    return sum(path.stat().st_size for path in Path(shard_root).rglob("*.parquet"))