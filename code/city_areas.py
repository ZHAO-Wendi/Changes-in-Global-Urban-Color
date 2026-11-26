"""Measure land cover inside each GUAB city boundary on the Equal Earth grid.
在 Equal Earth 等积网格上按 GUAB 城市边界计算土地覆盖面积。

python code/city_areas.py --boundaries GUAB_v1.0.gpkg --tiles masked_tiles --output areas

Areas are in km². A directory gives one CSV per year; a file combines dates.
面积单位为 km²。输出为目录时按年保存 CSV，指定文件时合并保存。"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.transform import array_bounds, rowcol
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.windows import Window

from spatial_io import read_units, unit_arguments, write_table, write_years, years_argument


EQUAL_EARTH = "EPSG:8859"
AREA_FIELDS = ("red_km2", "green_km2", "blue_km2", "other_km2", "outside_gub_km2")
ATTRIBUTES = ("city_name", "country", "default_level", "city_level", "type", "engtype")


def project_units(units):
    projected = []
    for start in range(0, len(units), 250):
        part = units.iloc[start:start + 250].copy()
        invalid = ~part.geometry.is_valid
        if invalid.any():
            part.loc[invalid, "geometry"] = part.geometry[invalid].buffer(0)
        part = part.loc[part.geometry.is_valid & ~part.geometry.is_empty].to_crs(EQUAL_EARTH)
        projected.append(part.loc[part.geometry.is_valid & ~part.geometry.is_empty])
    result = pd.concat(projected, ignore_index=True)
    if len(result) < len(units):
        warnings.warn(f"{len(units) - len(result)} boundaries were unusable on the area grid.",
                      RuntimeWarning)
    return result


def city_window(bounds, transform, width, height):
    # The source tables used a floored, exclusive lower/right edge.
    left, bottom, right, top = bounds
    row0, col0 = rowcol(transform, left, top)
    row1, col1 = rowcol(transform, right, bottom)
    row0 = max(0, min(int(row0), height))
    col0 = max(0, min(int(col0), width))
    row1 = max(0, min(int(row1), height))
    col1 = max(0, min(int(col1), width))
    if row1 - row0 < 2 or col1 - col0 < 2:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def pixel_counts(vrt, window, geometry):
    # Read in strips so unusually large administrative units fit in memory.
    counts = np.zeros(6, dtype=np.int64)
    has_pixels = False
    for offset in range(0, int(window.height), 1024):
        strip = Window(window.col_off, window.row_off + offset, window.width,
                       min(1024, window.height - offset))
        values = vrt.read(1, window=strip)
        mask = rasterize([(geometry, 1)], out_shape=values.shape,
                         transform=vrt.window_transform(strip), all_touched=False,
                         dtype=np.uint8).astype(bool)
        selected = values[mask]
        if selected.size:
            if selected.max() > 5:
                raise ValueError("Expected RGBO classes 1–4, outside-GUB code 5 or nodata 0.")
            counts += np.bincount(selected, minlength=6)
            has_pixels = True
    return counts[1:], has_pixels


def measure_year(units, tree, tiles, year, grids):
    if units.empty:
        return pd.DataFrame(columns=["gid", *ATTRIBUTES, "year", "admin_km2", *AREA_FIELDS])
    files = sorted((tiles / str(year)).rglob("*.tif"))
    if not files or len({p.name for p in files}) != len(files):
        raise ValueError(f"Missing tiles or duplicate tile names for {year}.")
    areas = np.zeros((len(units), 5))
    measured = np.zeros(len(units), dtype=bool)
    for path in files:
        with rasterio.open(path) as src:
            if src.count != 1 or src.crs is None:
                raise ValueError(f"Expected a single georeferenced RGBO band in {path.name}.")
            key = (src.crs, src.transform, src.width, src.height)
            if key not in grids:
                grids[key] = calculate_default_transform(src.crs, EQUAL_EARTH, src.width,
                                                         src.height, *src.bounds)
            transform, width, height = grids[key]
            tile = shapely.box(*array_bounds(height, width, transform))
            indices = tree.query(tile, predicate="intersects")
            if not len(indices):
                continue
            km2 = abs(transform.a * transform.e) / 1_000_000
            with WarpedVRT(src, crs=EQUAL_EARTH, transform=transform, width=width, height=height,
                           resampling=Resampling.nearest, nodata=0, warp_mem_limit=128) as vrt:
                for i in indices:
                    geometry = units.geometry.iloc[i]
                    window = city_window(geometry.bounds, transform, width, height)
                    if window is None:
                        continue
                    counts, found = pixel_counts(vrt, window, geometry)
                    if found:
                        areas[i] += counts * km2
                        measured[i] = True
    data = pd.DataFrame(units.drop(columns="geometry"))
    data["year"] = year
    data["admin_km2"] = units.area / 1_000_000
    for i, name in enumerate(AREA_FIELDS):
        data[name] = areas[:, i]
    # A geom with no sampled pixel has no annual row in the source protocol.
    columns = ["gid", *ATTRIBUTES, "year", "admin_km2", *AREA_FIELDS]
    return data.loc[measured, columns].sort_values("gid").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    unit_arguments(parser)
    years_argument(parser)
    parser.add_argument("--tiles", type=Path, required=True,
                        help="Reclassified tile root containing one directory per year. / 重分类图幅的根目录，各年份分别存放")
    parser.add_argument("--output", type=Path, required=True,
                        help="A combined .csv/.parquet, or a directory for all_cities_YEAR.csv. / 汇总输出的 CSV 或 Parquet 文件，或用于分年存放 all_cities_YEAR.csv 的目录")
    args = parser.parse_args()
    if len(set(args.year)) != len(args.year):
        parser.error("A year was specified more than once.")
    units = project_units(read_units(args.boundaries, args.gid, ATTRIBUTES))
    tree = shapely.STRtree(units.geometry.to_numpy())
    grids, frames = {}, []
    with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024, GDAL_NUM_THREADS="1"):
        for year in args.year:
            frame = measure_year(units, tree, args.tiles, year, grids)
            if args.output.suffix:
                frames.append(frame)
            else:
                write_table(frame, args.output / f"all_cities_{year}.csv")
    if frames:
        write_years(frames, args.output)


if __name__ == "__main__":
    main()

