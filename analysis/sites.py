"""Plant-derived site centres based on public VALORANT map callouts."""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


SITE_SEEDS = {
    "abyss": {"A": (4300.0, -200.0), "B": (-4425.0, -1175.0)},
    "ascent": {"A": (6153.585, -6626.2114), "B": (-2344.065, -7548.511)},
    "bind": {"A": (10747.902, 2664.4436), "B": (11108.1, -4831.5)},
    "breeze": {"A": (4825.0, 6325.0), "B": (6450.0, -5650.0)},
    "corrode": {"A": (3458.2793, -1824.5004), "B": (-2716.7, -1174.5)},
    "fracture": {"A": (8125.8, 3373.8), "B": (8178.0, -5942.0)},
    "haven": {
        "A": (6309.3076, -9225.703),
        "B": (1884.706, -9231.335),
        "C": (-2378.1, -9010.6),
    },
    "icebox": {"A": (-6400.0, 3200.0), "B": (1725.0, 2575.0)},
    "lotus": {
        "A": (7735.5396, 5557.309),
        "B": (6368.0, 668.2),
        "C": (6676.7, -4265.9),
    },
    "pearl": {"A": (6613.846, 5569.5254), "B": (5800.0, -2850.0)},
    "split": {"A": (6588.6597, -6761.131), "B": (-2167.2, -6264.8)},
    "sunset": {"A": (1000.0, 3200.0), "B": (-600.0, -5850.0)},
}

PLANT_ASSIGNMENT_RADIUS = 2600.0
SITE_RADIUS = 1800.0
NEAR_SITE_RADIUS = 3500.0


def build_site_centres(plants):
    """Cluster plant locations around the map's named A/B/C site callouts."""
    if plants.empty:
        return {}
    locations = plants[["map", "x", "y"]].copy()
    locations["map_key"] = locations["map"].astype("string").str.casefold()
    locations["x"] = pd.to_numeric(locations["x"], errors="coerce")
    locations["y"] = pd.to_numeric(locations["y"], errors="coerce")
    locations = locations.dropna(subset=["map_key", "x", "y"])

    result = {}
    for map_key, group in locations.groupby("map_key", sort=True):
        seeds = SITE_SEEDS.get(str(map_key))
        if not seeds:
            continue
        points = group[["x", "y"]].to_numpy(dtype=float)
        assigned = {letter: [] for letter in seeds}
        for point in points:
            letter = min(
                seeds,
                key=lambda site: np.linalg.norm(point - np.asarray(seeds[site], dtype=float)),
            )
            if np.linalg.norm(point - np.asarray(seeds[letter], dtype=float)) <= PLANT_ASSIGNMENT_RADIUS:
                assigned[letter].append(point)
        centres = {}
        for letter, site_points in assigned.items():
            if not site_points:
                continue
            centre = np.asarray(site_points, dtype=float).mean(axis=0)
            centres[letter] = {
                "x": float(centre[0]),
                "y": float(centre[1]),
                "plant_count": len(site_points),
            }
        if centres:
            result[str(map_key)] = centres
    return result


def _plants_signature(plants):
    columns = [column for column in ("map", "x", "y") if column in plants]
    digest = hashlib.blake2b(digest_size=20)
    digest.update(json.dumps([SITE_SEEDS, PLANT_ASSIGNMENT_RADIUS], sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(plants[columns], index=False).values.tobytes())
    return digest.hexdigest()


def load_site_centres(plants, cache_dir="cache"):
    """Load or generate ignored JSON site centres from the plant Parquet table."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "site_centres.json"
    signature = _plants_signature(plants)
    if cache_path.is_file():
        try:
            with cache_path.open(encoding="utf-8") as file:
                cached = json.load(file)
            if cached.get("signature") == signature:
                return cached.get("sites", {})
        except (OSError, ValueError):
            pass

    sites = build_site_centres(plants)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump({"signature": signature, "sites": sites}, file, separators=(",", ":"))
    return sites


def load_cached_site_centres(cache_dir="cache"):
    """Read the site centres already generated from the full cached plant table."""
    cache_path = Path(cache_dir) / "site_centres.json"
    with cache_path.open(encoding="utf-8") as file:
        return json.load(file).get("sites", {})


def site_label(map_name, x, y, site_centres):
    """Return a rough site/near-site/mid label for a coordinate."""
    try:
        point = np.asarray([float(x), float(y)], dtype=float)
    except (TypeError, ValueError):
        return "Site unknown"
    if not np.isfinite(point).all():
        return "Site unknown"
    sites = site_centres.get(str(map_name).casefold(), {})
    if not sites:
        return "Unlabeled area" if str(map_name).casefold() == "summit" else "Site unknown"
    letter, centre = min(
        sites.items(),
        key=lambda item: np.linalg.norm(point - np.asarray([item[1]["x"], item[1]["y"]])),
    )
    distance = np.linalg.norm(point - np.asarray([centre["x"], centre["y"]]))
    if distance <= SITE_RADIUS:
        return f"{letter} site"
    if distance <= NEAR_SITE_RADIUS:
        return f"near {letter} site"
    return "mid"


def disambiguate_zone_labels(zones, map_name, site_centres):
    """Append a compass hint when independent zones would otherwise share a title."""
    labeled = zones.copy()
    if labeled.empty or "site_label" not in labeled:
        return labeled
    map_sites = site_centres.get(str(map_name).casefold(), {})
    duplicated = labeled["site_label"].duplicated(keep=False)
    for label, indices in labeled[duplicated].groupby("site_label", sort=False).groups.items():
        indices = list(indices)
        match = re.search(r"(?:near\s+)?([A-Z])\s+site", str(label), flags=re.IGNORECASE)
        if match and match.group(1).upper() in map_sites:
            site = map_sites[match.group(1).upper()]
            anchor_x, anchor_y = float(site["x"]), float(site["y"])
        elif map_sites:
            anchor_x = sum(float(site["x"]) for site in map_sites.values()) / len(map_sites)
            anchor_y = sum(float(site["y"]) for site in map_sites.values()) / len(map_sites)
        else:
            anchor_x = float(labeled.loc[indices, "x"].median())
            anchor_y = float(labeled.loc[indices, "y"].median())

        directions = []
        for index in indices:
            dx = float(labeled.at[index, "x"]) - anchor_x
            dy = float(labeled.at[index, "y"]) - anchor_y
            angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            directions.append((index, ("east", "north-east", "north", "north-west", "west",
                                       "south-west", "south", "south-east")[int((angle + 22.5) // 45) % 8]))
        counts = {}
        for _, direction in directions:
            counts[direction] = counts.get(direction, 0) + 1
        seen = {}
        for index, direction in directions:
            seen[direction] = seen.get(direction, 0) + 1
            hint = f"{direction} {seen[direction]}" if counts[direction] > 1 else direction
            labeled.at[index, "site_label"] = f"{label} ({hint})"
    return labeled


def label_zones(zones, map_name, site_centres, disambiguate=False):
    """Add approximate map-site labels to aggregated hot zones."""
    labeled = zones.copy()
    if labeled.empty:
        labeled["site_label"] = pd.Series(dtype=str)
        return labeled
    labeled["site_label"] = [
        site_label(map_name, row.x, row.y, site_centres)
        for row in labeled.itertuples()
    ]
    return disambiguate_zone_labels(labeled, map_name, site_centres) if disambiguate else labeled