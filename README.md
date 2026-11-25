# Scout Sheet

Scout Sheet is a VALORANT opponent-scouting tool for coaches, built for the Cloud9 x JetBrains Sky's The Limit 2026 hackathon.

Keep the GRID data repository alongside this project (or set `STL_DATA_DIR` to its path); never copy or commit its data here. With Python 3.14 and uv installed, run `uv run explore.py` to explore the local data.

Build scouting tables from a small sample with `uv run build_cache.py --limit 3`; by default this selects Cloud9 and Cloud9 GC series plus their opponents' series in those events. The script streams each feed, skips invalid sessions and rolled-back games, and writes `rounds.parquet`, `casts.parquet`, `kills.parquet`, and `buys.parquet` to the git-ignored `cache/` directory. Omit `--limit` to process the full selection, or pass `--series-ids ID ...` to choose series explicitly. The cast table uses the nearest movement frame for `gap_exceeded` supplement records.