"""Cached map-specific position density, derived only from local Parquet tables."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _positions_by_map(casts, kills):
    positions = {}
    cast_positions = casts[["map", "x", "y"]].rename(columns={"x": "x", "y": "y"})
    killer_positions = kills[["map", "killer_x", "killer_y"]].rename(
        columns={"killer_x": "x", "killer_y": "y"},
    )
    victim_positions = kills[["map", "victim_x", "victim_y"]].rename(
        columns={"victim_x": "x", "victim_y": "y"},
    )
    for frame in (cast_positions, killer_positions, victim_positions):
        frame = frame.copy()
        frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
        frame = frame.dropna(subset=["map", "x", "y"])
        for map_name, group in frame.groupby("map", sort=False):
            positions.setdefault(str(map_name), []).append(group[["x", "y"]].to_numpy(dtype=float))
    return {
        map_name: np.concatenate(groups, axis=0)
        for map_name, groups in positions.items()
        if groups
    }


def _source_signature(casts, kills, bins):
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(bins).encode())
    for frame, columns in (
        (casts, ["map", "x", "y"]),
        (kills, ["map", "killer_x", "killer_y", "victim_x", "victim_y"]),
    ):
        available = [column for column in columns if column in frame.columns]
        digest.update(pd.util.hash_pandas_object(frame[available], index=False).values.tobytes())
    return digest.hexdigest()


def _density_shape(points, bins):
    x_values, y_values = points[:, 0], points[:, 1]
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1
    histogram, x_edges, y_edges = np.histogram2d(
        x_values, y_values, bins=bins, range=((x_min, x_max), (y_min, y_max)),
    )
    density = np.log1p(histogram.T)
    maximum = float(density.max())
    if maximum:
        density /= maximum
    return {
        "density": density,
        "x": (x_edges[:-1] + x_edges[1:]) / 2,
        "y": (y_edges[:-1] + y_edges[1:]) / 2,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "bins": bins,
        "position_count": len(points),
    }


def load_map_shapes(casts, kills, cache_dir="cache", bins=120):
    """Load or build cached 2D position-density shapes for all maps."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "map_shapes.npz"
    signature = _source_signature(casts, kills, bins)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                if str(archive["signature"].item()) == signature:
                    result = {}
                    for index, map_name in enumerate(archive["map_names"].tolist()):
                        result[map_name] = {
                            key: archive[f"{key}_{index}"].copy()
                            for key in ("density", "x", "y")
                        }
                        for key in ("x_min", "x_max", "y_min", "y_max", "bins", "position_count"):
                            result[map_name][key] = archive[f"{key}_{index}"].item()
                    return result
        except (KeyError, OSError, ValueError):
            pass

    shapes = {
        map_name: _density_shape(points, bins)
        for map_name, points in _positions_by_map(casts, kills).items()
    }
    archive_values = {
        "signature": np.asarray(signature),
        "map_names": np.asarray(list(shapes), dtype=str),
    }
    for index, shape in enumerate(shapes.values()):
        for key in ("density", "x", "y", "x_min", "x_max", "y_min", "y_max", "bins", "position_count"):
            archive_values[f"{key}_{index}"] = np.asarray(shape[key])
    np.savez_compressed(cache_path, **archive_values)
    return shapes


def density_for_points(points, shape):
    """Rasterize x/y points on the same grid as a cached map shape."""
    if len(points) == 0:
        return np.zeros_like(shape["density"])
    values = np.asarray(points, dtype=float)
    valid = np.isfinite(values).all(axis=1)
    values = values[valid]
    histogram, _, _ = np.histogram2d(
        values[:, 0], values[:, 1], bins=int(shape["bins"]),
        range=((shape["x_min"], shape["x_max"]), (shape["y_min"], shape["y_max"])),
    )
    density = np.log1p(histogram.T)
    maximum = float(density.max())
    if maximum:
        density /= maximum
    return density