"""Cached map-specific position density, derived only from local Parquet tables."""

import hashlib
import math
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
        frame = frame[np.isfinite(frame[["x", "y"]]).all(axis=1)]
        for map_name, group in frame.groupby("map", sort=False):
            positions.setdefault(str(map_name), []).append(group[["x", "y"]].to_numpy(dtype=float))
    return {
        map_name: np.concatenate(groups, axis=0)
        for map_name, groups in positions.items()
        if groups
    }


def _source_signature(casts, kills, bins, sigma):
    digest = hashlib.blake2b(digest_size=20)
    digest.update(f"robust-extent-v1:{bins}:{sigma}".encode())
    for frame, columns in (
        (casts, ["map", "x", "y"]),
        (kills, ["map", "killer_x", "killer_y", "victim_x", "victim_y"]),
    ):
        available = [column for column in columns if column in frame.columns]
        digest.update(pd.util.hash_pandas_object(frame[available], index=False).values.tobytes())
    return digest.hexdigest()


def _trim_coordinate_outliers(points):
    """Remove only extreme per-axis outliers before deriving map bounds."""
    points = np.asarray(points, dtype=float)
    if len(points) < 8:
        return points
    keep = np.ones(len(points), dtype=bool)
    for axis in (0, 1):
        values = points[:, axis]
        lower_quartile, upper_quartile = np.percentile(values, [25, 75])
        interquartile_range = upper_quartile - lower_quartile
        if interquartile_range <= 0:
            continue
        lower = lower_quartile - 3 * interquartile_range
        upper = upper_quartile + 3 * interquartile_range
        keep &= (values >= lower) & (values <= upper)
    if keep.any():
        return points[keep]
    return points


def _gaussian_blur(values, sigma):
    if sigma <= 0:
        return values
    radius = max(1, math.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    vertical = np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"),
        axis=0,
        arr=np.pad(values, ((radius, radius), (0, 0)), mode="edge"),
    )
    return np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"),
        axis=1,
        arr=np.pad(vertical, ((0, 0), (radius, radius)), mode="edge"),
    )


def _density_shape(points, bins, sigma):
    points = points[np.isfinite(points).all(axis=1)]
    points = _trim_coordinate_outliers(points)
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
    density = _gaussian_blur(np.log1p(histogram.T), sigma)
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


def load_map_shapes(casts, kills, cache_dir="cache", bins=240, sigma=1.5):
    """Load or build cached 2D position-density shapes for all maps."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "map_shapes.npz"
    signature = _source_signature(casts, kills, bins, sigma)
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
        map_name: _density_shape(points, bins, sigma)
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


def density_for_points(points, shape, sigma=10):
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
    density = _gaussian_blur(np.log1p(histogram.T), sigma)
    maximum = float(density.max())
    if maximum:
        density /= maximum
    return density