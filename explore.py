import csv
from collections import Counter

from config import DATA_DIR, read_jsonl_zst


def main():
    with (DATA_DIR / "val/series.csv").open(newline="", encoding="utf-8") as file:
        series = [
            row for row in csv.DictReader(file)
            if any(team.strip() in {"Cloud9", "Cloud9 GC"} for team in row["teams"].split(" vs "))
        ]

    print(f"Cloud9 or Cloud9 GC series: {len(series)}")
    selected = next((row for row in series if row["ability_supplement"] and (DATA_DIR / row["ability_supplement"]).is_file()), None)
    if selected is None:
        print("No Cloud9 series with an available ability supplement found.")
        return

    print(f"Series: {selected['series_id']} ({selected['teams']})")
    counts = Counter()
    ability_keys = None
    player_keys = None
    games = []
    for envelope in read_jsonl_zst(DATA_DIR / selected["file"]):
        for event in envelope["events"]:
            counts[event["type"]] += 1
            if ability_keys is None and event["type"] == "player-used-ability":
                ability_keys = sorted(event)
            if event.get("includesFullState") and event.get("seriesState"):
                state = event["seriesState"]
                games = state.get("games", [])
                if player_keys is None:
                    for game in games:
                        for team in game.get("teams", []):
                            if team.get("players"):
                                player_keys = sorted(team["players"][0])
                                break
                        if player_keys is not None:
                            break

    supplement = next(read_jsonl_zst(DATA_DIR / selected["ability_supplement"]), None)
    print("Top 30 event types:")
    for event_type, count in counts.most_common(30):
        print(f"  {event_type}: {count}")
    print(f"Ability event keys: {ability_keys}")
    print(f"Full-state player keys: {player_keys}")
    print(f"Supplement record keys: {sorted(supplement) if supplement is not None else None}")
    print("Maps and bounds (last full state):")
    for game in games:
        map_info = game.get("map") or {}
        print(f"  {map_info.get('name')}: {map_info.get('bounds')}")


if __name__ == "__main__":
    main()