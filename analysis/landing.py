"""Heuristic utility landing locations from aim direction and observed map density."""

import math

import numpy as np
import pandas as pd

from analysis.abilities import ability_category, ability_is_remote_cast


WALL_PLACEMENTS = {
    ("cypher", "trapwire"): 1800.0,
    ("chamber", "trademark"): 1800.0,
    ("deadlock", "sonic-sensor"): 1800.0,
    ("vyse", "arc-rose"): 1800.0,
    ("vyse", "shear"): 1800.0,
}

# (base range, minimum range, maximum range), in GRID map units.
THROWN_RANGES = {
    ("cypher", "spycam"): (5200.0, 1800.0, 7200.0),
    ("killjoy", "alarmbot"): (4200.0, 1600.0, 6000.0),
    ("killjoy", "nanoswarm"): (4800.0, 1600.0, 6800.0),
    ("vyse", "razorvine"): (4400.0, 1600.0, 6200.0),
    ("brimstone", "incendiary"): (5600.0, 2000.0, 7600.0),
    ("phoenix", "hot-hands"): (4700.0, 1600.0, 6400.0),
    ("viper", "snake-bite"): (5600.0, 2000.0, 7600.0),
    ("generic", "cloudburst"): (4200.0, 1600.0, 6000.0),
    ("generic", "poison-cloud"): (5200.0, 1800.0, 7000.0),
    ("generic", "mosh-pit"): (6000.0, 2200.0, 8000.0),
    ("generic", "snake-bite"): (5600.0, 2000.0, 7600.0),
    ("generic", "incendiary"): (5600.0, 2000.0, 7600.0),
    ("generic", "hot-hands"): (4700.0, 1600.0, 6400.0),
    ("generic", "paint-shells"): (5000.0, 1800.0, 7000.0),
    ("generic", "frag-ment"): (5000.0, 1800.0, 7000.0),
    ("generic", "special-delivery"): (4400.0, 1600.0, 6200.0),
    ("generic", "blindside"): (4800.0, 1600.0, 6600.0),
    ("generic", "curveball"): (4000.0, 1400.0, 5600.0),
    ("generic", "guiding-light"): (5000.0, 1800.0, 7000.0),
    ("generic", "flash-drive"): (5200.0, 1800.0, 7200.0),
    ("generic", "recon-bolt"): (5600.0, 2000.0, 7600.0),
    ("generic", "shock-bolt"): (5600.0, 2000.0, 7600.0),
    ("generic", "relay-bolt"): (5000.0, 1800.0, 7000.0),
    ("generic", "cove"): (4800.0, 1600.0, 6600.0),
    ("generic", "razorvine"): (4400.0, 1600.0, 6200.0),
    ("generic", "nanoswarm"): (4800.0, 1600.0, 6800.0),
    ("generic", "spycam"): (5200.0, 1800.0, 7200.0),
}

SELF_CENTERED_ABILITIES = {
    "heal", "healing-orb", "resurrection", "pick-me-up", "stim-beacon",
    "tailwind", "updraft", "blast-pack", "gatecrash", "shrouded-step",
    "dimensional-drift", "from-the-shadows", "high-gear", "light-speed",
    "refract", "rendezvous", "convergent-paths",
}

LANDING_COLUMNS = ("landing_x", "landing_y", "landing_method", "estimated")


def build_walkable_mask(shape, threshold=0.025, dilation_cells=1):
    """Threshold a density silhouette and lightly dilate its observed walkable area."""
    density = np.asarray(shape["density"], dtype=float)
    mask = np.isfinite(density) & (density >= float(threshold))
    radius = max(0, int(dilation_cells))
    if radius == 0:
        return mask
    dilated = mask.copy()
    height, width = mask.shape
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if delta_x * delta_x + delta_y * delta_y > radius * radius + 0.5:
                continue
            source_x = slice(max(0, -delta_x), min(width, width - delta_x))
            target_x = slice(max(0, delta_x), min(width, width + delta_x))
            source_y = slice(max(0, -delta_y), min(height, height - delta_y))
            target_y = slice(max(0, delta_y), min(height, height + delta_y))
            dilated[target_y, target_x] |= mask[source_y, source_x]
    return dilated


def _grid_step(shape):
    steps = []
    for axis in ("x", "y"):
        values = np.asarray(shape[axis], dtype=float)
        differences = np.abs(np.diff(values))
        differences = differences[differences > 0]
        if len(differences):
            steps.append(float(np.median(differences)))
    return min(steps) if steps else 100.0


def is_walkable(x, y, shape, mask=None):
    """Look up the closest density cell, rejecting coordinates outside the grid."""
    if mask is None:
        mask = build_walkable_mask(shape)
    x_values = np.asarray(shape["x"], dtype=float)
    y_values = np.asarray(shape["y"], dtype=float)
    if not len(x_values) or not len(y_values):
        return False
    if x < x_values.min() or x > x_values.max() or y < y_values.min() or y > y_values.max():
        return False

    def nearest_index(values, coordinate):
        upper = int(np.searchsorted(values, coordinate, side="left"))
        if upper <= 0:
            return 0
        if upper >= len(values):
            return len(values) - 1
        lower = upper - 1
        return lower if coordinate - values[lower] <= values[upper] - coordinate else upper

    column = nearest_index(x_values, float(x))
    row = nearest_index(y_values, float(y))
    return bool(mask[row, column])


def ray_march_landing(caster_x, caster_y, view_x, view_y, shape, max_range, step=None, mask=None):
    """March along horizontal aim and return the last walkable point or the range limit."""
    direction_length = math.hypot(float(view_x), float(view_y))
    if direction_length <= 0 or float(max_range) <= 0:
        return float(caster_x), float(caster_y), "caster_position"
    direction_x = float(view_x) / direction_length
    direction_y = float(view_y) / direction_length
    if mask is None:
        mask = build_walkable_mask(shape)
    if not is_walkable(caster_x, caster_y, shape, mask):
        return float(caster_x), float(caster_y), "caster_position"

    step = max(1.0, float(step) if step is not None else _grid_step(shape) / 2)
    max_range = float(max_range)
    last_x, last_y = float(caster_x), float(caster_y)
    distance = min(step, max_range)
    while True:
        candidate_x = float(caster_x) + direction_x * distance
        candidate_y = float(caster_y) + direction_y * distance
        if not is_walkable(candidate_x, candidate_y, shape, mask):
            return last_x, last_y, "wall_hit"
        last_x, last_y = candidate_x, candidate_y
        if distance >= max_range:
            return last_x, last_y, "max_range"
        distance = min(distance + step, max_range)


def _normalized_name(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower().replace("_", "-")


def _landing_rule(agent, ability):
    key = (_normalized_name(agent), _normalized_name(ability))
    if ability_is_remote_cast(key[1], key[0]):
        return "caster", None
    if key[1] in SELF_CENTERED_ABILITIES or ability_category(key[1], key[0]) == "movement":
        return "caster", None
    if key in WALL_PLACEMENTS:
        return "wall", WALL_PLACEMENTS[key]
    thrown_range = THROWN_RANGES.get(key) or THROWN_RANGES.get(("generic", key[1]))
    if thrown_range and thrown_range[2] > 0:
        return "thrown", thrown_range
    return "caster", None


def _scaled_thrown_range(view_x, view_y, view_z, ranges):
    base_range, minimum, maximum = ranges
    horizontal_length = math.hypot(float(view_x), float(view_y))
    upward_angle = math.atan2(max(0.0, float(view_z)), horizontal_length)
    return min(max(float(base_range) * math.cos(upward_angle), float(minimum)), float(maximum))


def estimate_cast_landing(cast, shapes, masks=None):
    """Return landing fields for one cast, preserving caster position when uncertain."""
    x = pd.to_numeric(cast.get("x"), errors="coerce")
    y = pd.to_numeric(cast.get("y"), errors="coerce")
    if pd.isna(x) or pd.isna(y):
        return {"landing_x": np.nan, "landing_y": np.nan, "landing_method": "caster_position", "estimated": False}
    x, y = float(x), float(y)
    fallback = {"landing_x": x, "landing_y": y, "landing_method": "caster_position", "estimated": False}
    rule, details = _landing_rule(cast.get("agent"), cast.get("ability"))
    if rule == "caster":
        return fallback
    map_name = str(cast.get("map", ""))
    shape = shapes.get(map_name)
    if shape is None:
        shape = shapes.get(map_name.casefold())
    if shape is None:
        return fallback
    view_x = pd.to_numeric(cast.get("view_x"), errors="coerce")
    view_y = pd.to_numeric(cast.get("view_y"), errors="coerce")
    view_z = pd.to_numeric(cast.get("view_z"), errors="coerce")
    if pd.isna(view_x) or pd.isna(view_y) or math.hypot(float(view_x), float(view_y)) <= 0:
        return fallback
    mask = None
    if masks is not None:
        mask = masks.get(map_name, masks.get(map_name.casefold()))
    if mask is None:
        mask = build_walkable_mask(shape)
    max_range = float(details) if rule == "wall" else _scaled_thrown_range(
        view_x, view_y, 0.0 if pd.isna(view_z) else view_z, details,
    )
    landing_x, landing_y, method = ray_march_landing(
        x, y, view_x, view_y, shape, max_range, mask=mask,
    )
    if method == "caster_position":
        return fallback
    return {
        "landing_x": landing_x, "landing_y": landing_y,
        "landing_method": method, "estimated": True,
    }


def add_landing_estimates(casts, shapes):
    """Add estimated landing fields to a cast table without replacing caster coordinates."""
    result = casts.copy()
    masks = {map_name: build_walkable_mask(shape) for map_name, shape in shapes.items()}
    columns = ["map", "agent", "ability", "x", "y", "view_x", "view_y", "view_z"]
    estimates = [
        estimate_cast_landing(dict(zip(columns, values)), shapes, masks)
        for values in result[columns].itertuples(index=False, name=None)
    ]
    for column in LANDING_COLUMNS:
        result[column] = [row[column] for row in estimates]
    return result


def landing_sanity_report(casts, shapes):
    """Summarize estimated-point mask coverage, per-ability distance, and examples."""
    estimated = casts[casts.get("estimated", pd.Series(False, index=casts.index)).fillna(False).astype(bool)].copy()
    map_coverage = {}
    for map_name, group in estimated.groupby("map", dropna=True, sort=True):
        shape = shapes.get(str(map_name), shapes.get(str(map_name).casefold()))
        mask = build_walkable_mask(shape) if shape is not None else None
        inside = [
            is_walkable(row.landing_x, row.landing_y, shape, mask)
            if shape is not None and pd.notna(row.landing_x) and pd.notna(row.landing_y) else False
            for row in group.itertuples(index=False)
        ]
        map_coverage[str(map_name)] = (int(sum(inside)), len(inside))
    distances = np.hypot(
        pd.to_numeric(estimated.get("landing_x"), errors="coerce")
        - pd.to_numeric(estimated.get("x"), errors="coerce"),
        pd.to_numeric(estimated.get("landing_y"), errors="coerce")
        - pd.to_numeric(estimated.get("y"), errors="coerce"),
    )
    estimated["caster_landing_distance"] = distances
    medians = estimated.groupby("ability", dropna=True)["caster_landing_distance"].median().sort_index()
    samples = estimated.dropna(subset=["x", "y", "view_x", "view_y", "landing_x", "landing_y"]).head(5)
    return {"map_coverage": map_coverage, "median_distance_by_ability": medians, "samples": samples}


def print_landing_sanity_report(casts, shapes):
    report = landing_sanity_report(casts, shapes)
    print("Landing sanity report (estimated points only):")
    for map_name, (inside, total) in report["map_coverage"].items():
        percentage = 100 * inside / total if total else 0.0
        print(f"  {map_name}: {percentage:.1f}% inside walkable mask ({inside}/{total})")
    print("Median caster-to-landing distance by ability:")
    for ability, distance in report["median_distance_by_ability"].items():
        print(f"  {ability}: {distance:.0f} map units")
    print("Sample casts (map, ability, caster, view, landing):")
    for row in report["samples"].itertuples(index=False):
        print(
            f"  {row.map} · {row.ability}: caster=({row.x:.0f}, {row.y:.0f}), "
            f"view=({row.view_x:.3f}, {row.view_y:.3f}, {row.view_z:.3f}), "
            f"landing=({row.landing_x:.0f}, {row.landing_y:.0f}) [{row.landing_method}]"
        )