"""Heuristic utility landing locations from aim direction and observed map density."""

import math

import numpy as np
import pandas as pd

from analysis.abilities import ability_category, ability_is_remote_cast


SHORT_PLACEMENTS = {
    ("killjoy", "alarmbot"),
    ("killjoy", "nanoswarm"),
    ("killjoy", "turret"),
    ("chamber", "trademark"),
    ("deadlock", "sonic-sensor"),
    ("vyse", "shear"),
}
# The supplement does not identify a long-throw nanoswarm cast, so all are treated as short placement.
SHORT_PLACEMENT_RANGE = 500.0
WALL_PLACEMENTS = {("cypher", "trapwire"): 800.0}

# (base range, minimum range, maximum range), in GRID map units.
THROWN_RANGES = {
    ("cypher", "spycam"): (3800.0, 1400.0, 4800.0),
    ("vyse", "arc-rose"): (3000.0, 1000.0, 3800.0),
    ("vyse", "razorvine"): (3400.0, 1200.0, 4200.0),
    ("brimstone", "incendiary"): (5000.0, 1800.0, 6200.0),
    ("phoenix", "hot-hands"): (4200.0, 1400.0, 5200.0),
    ("viper", "snake-bite"): (5000.0, 1800.0, 6200.0),
    ("generic", "cloudburst"): (3500.0, 1200.0, 4400.0),
    ("generic", "poison-cloud"): (4400.0, 1500.0, 5500.0),
    ("generic", "mosh-pit"): (5000.0, 1800.0, 6200.0),
    ("generic", "snake-bite"): (5000.0, 1800.0, 6200.0),
    ("generic", "incendiary"): (5000.0, 1800.0, 6200.0),
    ("generic", "hot-hands"): (4200.0, 1400.0, 5200.0),
    ("generic", "paint-shells"): (4500.0, 1600.0, 5600.0),
    ("generic", "frag-ment"): (4500.0, 1600.0, 5600.0),
    ("generic", "special-delivery"): (3500.0, 1200.0, 4400.0),
    ("generic", "blindside"): (3500.0, 1200.0, 4400.0),
    ("generic", "curveball"): (3000.0, 1000.0, 3800.0),
    ("generic", "guiding-light"): (4200.0, 1500.0, 5200.0),
    ("generic", "flash-drive"): (4400.0, 1500.0, 5500.0),
    ("generic", "recon-bolt"): (5000.0, 1800.0, 6200.0),
    ("generic", "shock-bolt"): (5000.0, 1800.0, 6200.0),
    ("generic", "relay-bolt"): (4500.0, 1500.0, 5600.0),
    ("generic", "cove"): (4000.0, 1400.0, 5000.0),
    ("generic", "razorvine"): (3400.0, 1200.0, 4200.0),
    ("generic", "spycam"): (3800.0, 1400.0, 4800.0),
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


def _range_key(agent, ability, thrown_ranges):
    key = (_normalized_name(agent), _normalized_name(ability))
    if key in thrown_ranges:
        return key
    fallback = ("generic", key[1])
    return fallback if fallback in thrown_ranges else None


def _landing_rule(agent, ability, thrown_ranges=None):
    key = (_normalized_name(agent), _normalized_name(ability))
    thrown_ranges = THROWN_RANGES if thrown_ranges is None else thrown_ranges
    if ability_is_remote_cast(key[1], key[0]):
        return "caster", None
    if key[1] in SELF_CENTERED_ABILITIES or ability_category(key[1], key[0]) == "movement":
        return "caster", None
    if key in SHORT_PLACEMENTS:
        return "short", SHORT_PLACEMENT_RANGE
    if key in WALL_PLACEMENTS:
        return "wall", WALL_PLACEMENTS[key]
    range_key = _range_key(key[0], key[1], thrown_ranges)
    thrown_range = thrown_ranges.get(range_key) if range_key else None
    if thrown_range and thrown_range[2] > 0:
        return "thrown", thrown_range
    return "caster", None


def _scaled_thrown_range(view_x, view_y, view_z, ranges):
    base_range, minimum, maximum = ranges
    horizontal_length = math.hypot(float(view_x), float(view_y))
    upward_angle = math.atan2(max(0.0, float(view_z)), horizontal_length)
    return min(max(float(base_range) * math.cos(upward_angle), float(minimum)), float(maximum))


def estimate_cast_landing(cast, shapes, masks=None, thrown_ranges=None):
    """Return landing fields for one cast, preserving caster position when uncertain."""
    x = pd.to_numeric(cast.get("x"), errors="coerce")
    y = pd.to_numeric(cast.get("y"), errors="coerce")
    if pd.isna(x) or pd.isna(y):
        return {"landing_x": np.nan, "landing_y": np.nan, "landing_method": "caster_position", "estimated": False}
    x, y = float(x), float(y)
    fallback = {"landing_x": x, "landing_y": y, "landing_method": "caster_position", "estimated": False}
    thrown_ranges = THROWN_RANGES if thrown_ranges is None else thrown_ranges
    rule, details = _landing_rule(cast.get("agent"), cast.get("ability"), thrown_ranges)
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
    if rule in {"short", "wall"}:
        max_range = float(details)
    else:
        max_range = _scaled_thrown_range(view_x, view_y, 0.0 if pd.isna(view_z) else view_z, details)
    landing_x, landing_y, method = ray_march_landing(
        x, y, view_x, view_y, shape, max_range, mask=mask,
    )
    if method == "caster_position":
        return fallback
    return {
        "landing_x": landing_x, "landing_y": landing_y,
        "landing_method": method, "estimated": True,
    }


def add_landing_estimates(casts, shapes, thrown_ranges=None):
    """Add estimated landing fields to a cast table without replacing caster coordinates."""
    result = casts.copy()
    masks = {map_name: build_walkable_mask(shape) for map_name, shape in shapes.items()}
    thrown_ranges = THROWN_RANGES if thrown_ranges is None else thrown_ranges
    columns = ["map", "agent", "ability", "x", "y", "view_x", "view_y", "view_z"]
    estimates = [
        estimate_cast_landing(dict(zip(columns, values)), shapes, masks, thrown_ranges)
        for values in result[columns].itertuples(index=False, name=None)
    ]
    for column in LANDING_COLUMNS:
        result[column] = [row[column] for row in estimates]
    return result


def calibrate_thrown_ranges(casts, shapes, hit_threshold=0.4, reduction=0.8):
    """Lower configured thrown caps once when too many valid rays reach their limit."""
    return calibrate_thrown_ranges_batches((casts,), shapes, hit_threshold, reduction)


def calibrate_thrown_ranges_batches(cast_frames, shapes, hit_threshold=0.4, reduction=0.8):
    """Calibrate thrown caps across a stream of per-series cast frames."""
    ranges = {key: tuple(value) for key, value in THROWN_RANGES.items()}
    masks = {map_name: build_walkable_mask(shape) for map_name, shape in shapes.items()}
    counts = {}
    columns = ["map", "agent", "ability", "x", "y", "view_x", "view_y", "view_z"]
    for casts in cast_frames:
        if casts.empty or not set(columns).issubset(casts.columns):
            continue
        for values in casts[columns].itertuples(index=False, name=None):
            cast = dict(zip(columns, values))
            rule, _ = _landing_rule(cast["agent"], cast["ability"], ranges)
            if rule != "thrown":
                continue
            key = _range_key(cast["agent"], cast["ability"], ranges)
            method = estimate_cast_landing(cast, shapes, masks, ranges)["landing_method"]
            stats = counts.setdefault(key, {"valid_rays": 0, "max_hits": 0})
            if method in {"wall_hit", "max_range"}:
                stats["valid_rays"] += 1
                stats["max_hits"] += method == "max_range"

    adjustments = {}
    for range_key, stats in counts.items():
        valid_rays = stats["valid_rays"]
        max_hits = stats["max_hits"]
        hit_fraction = max_hits / valid_rays if valid_rays else 0.0
        if hit_fraction <= hit_threshold:
            continue
        base_range, minimum, old_maximum = ranges[range_key]
        new_maximum = max(minimum, math.floor(old_maximum * reduction / 100) * 100)
        if new_maximum >= old_maximum:
            new_maximum = max(minimum, old_maximum - 100)
        if new_maximum < old_maximum:
            ranges[range_key] = (base_range, minimum, new_maximum)
            adjustments[range_key] = {
                "max_hits": max_hits,
                "valid_rays": valid_rays,
                "hit_fraction": hit_fraction,
                "old_maximum": old_maximum,
                "new_maximum": new_maximum,
            }
    return ranges, adjustments


def _effective_limit(cast, thrown_ranges):
    rule, details = _landing_rule(cast.get("agent"), cast.get("ability"), thrown_ranges)
    if rule in {"short", "wall"}:
        return float(details)
    if rule != "thrown":
        return np.nan
    view_x = pd.to_numeric(cast.get("view_x"), errors="coerce")
    view_y = pd.to_numeric(cast.get("view_y"), errors="coerce")
    view_z = pd.to_numeric(cast.get("view_z"), errors="coerce")
    if pd.isna(view_x) or pd.isna(view_y) or math.hypot(float(view_x), float(view_y)) <= 0:
        return np.nan
    return _scaled_thrown_range(view_x, view_y, 0.0 if pd.isna(view_z) else view_z, details)


def landing_method_counts(casts):
    """Return wall/range/caster method counts indexed by ability."""
    methods = ["wall_hit", "max_range", "caster_position"]
    if casts.empty or not {"ability", "landing_method"}.issubset(casts.columns):
        return pd.DataFrame(columns=methods, dtype=int)
    return casts.groupby("ability", dropna=False)["landing_method"].value_counts().unstack(
        fill_value=0,
    ).reindex(columns=methods, fill_value=0).sort_index()


def print_landing_method_counts(casts, title):
    counts = landing_method_counts(casts)
    print(title)
    for ability, row in counts.iterrows():
        print(
            f"  {ability}: wall_hit={int(row['wall_hit']):,}, "
            f"max_range={int(row['max_range']):,}, "
            f"caster_position={int(row['caster_position']):,}"
        )


def landing_sanity_report(casts, shapes, thrown_ranges=None, adjustments=None):
    """Summarize estimated-point mask coverage, per-ability distance, and examples."""
    thrown_ranges = THROWN_RANGES if thrown_ranges is None else thrown_ranges
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
    estimated["effective_limit"] = [
        _effective_limit(row._asdict(), thrown_ranges)
        for row in estimated.itertuples(index=False)
    ]
    effective_limits = estimated.groupby("ability", dropna=True)["effective_limit"].median().sort_index()
    median_at_limit = {}
    for ability, distance in medians.items():
        limit = effective_limits.get(ability, np.nan)
        if pd.isna(limit):
            median_at_limit[ability] = False
            continue
        map_names = estimated.loc[estimated["ability"].eq(ability), "map"].dropna().astype(str)
        step_sizes = [
            _grid_step(shapes[name])
            for name in map_names
            if name in shapes or name.casefold() in shapes
        ]
        tolerance = max(step_sizes, default=100.0) * 0.51
        median_at_limit[ability] = abs(float(distance) - float(limit)) <= tolerance
    samples = estimated.dropna(subset=["x", "y", "view_x", "view_y", "landing_x", "landing_y"]).head(5)
    return {
        "map_coverage": map_coverage,
        "median_distance_by_ability": medians,
        "median_effective_limit_by_ability": effective_limits,
        "median_at_limit": median_at_limit,
        "method_counts_by_ability": landing_method_counts(casts),
        "adjustments": adjustments or {},
        "samples": samples,
    }


def print_landing_sanity_report(casts, shapes, thrown_ranges=None, adjustments=None):
    report = landing_sanity_report(casts, shapes, thrown_ranges, adjustments)
    print("Landing sanity report (estimated points only):")
    for map_name, (inside, total) in report["map_coverage"].items():
        percentage = 100 * inside / total if total else 0.0
        print(f"  {map_name}: {percentage:.1f}% inside walkable mask ({inside}/{total})")
    print("Median caster-to-landing distance by ability (map units; max-limit flags):")
    for ability, distance in report["median_distance_by_ability"].items():
        flag = " [FLAG: median equals effective maximum range]" if report["median_at_limit"][ability] else ""
        print(f"  {ability}: {distance:.0f}{flag}")
    print("Thrown maximum-range adjustments (>40% hit the configured range limit):")
    if not report["adjustments"]:
        print("  none")
    for (agent, ability), adjustment in report["adjustments"].items():
        print(
            f"  {agent} / {ability}: {adjustment['hit_fraction']:.1%} hit max "
            f"({adjustment['max_hits']}/{adjustment['valid_rays']}); "
            f"max {adjustment['old_maximum']:.0f} → {adjustment['new_maximum']:.0f} map units"
        )
    print("Sample casts (map, ability, caster, view, landing):")
    for row in report["samples"].itertuples(index=False):
        print(
            f"  {row.map} · {row.ability}: caster=({row.x:.0f}, {row.y:.0f}), "
            f"view=({row.view_x:.3f}, {row.view_y:.3f}, {row.view_z:.3f}), "
            f"landing=({row.landing_x:.0f}, {row.landing_y:.0f}) [{row.landing_method}]"
        )