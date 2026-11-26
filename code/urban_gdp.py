"""Aggregate gridded GDP and product-implied population inside same-year urban footprints.
按城市行政边界和同期 GUB 提取栅格 GDP 总量与产品隐含人口。

python code/urban_gdp.py --boundaries GUAB_v1.0.gpkg --gub GUB --total rast_gdpTot_1990_2024_5arcmin.tif --per-capita rast_adm2_gdp_perCapita_1990_2024.tif --output city_gdp.parquet

Use --panel to limit extraction to the locally generated panel.
使用 --panel 可仅提取本地生成的面板所列城市。
GDP source / GDP 栅格来源: Kummu, Kosonen & Masoumzadeh Sayyar, doi:10.5281/zenodo.18429133."""

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from rasterio.windows import Window, bounds as window_bounds, from_bounds

from spatial_io import polygonal, read_gub, read_units, unit_arguments, write_years, years_argument


NAMES = {
    "gdp_sum": "gdp_total_ppp_2021usd",
    "pop_sum": "product_implied_population",
    "pc_min": "source_GDPpc_min",
    "pc_max": "source_GDPpc_max",
    "pc_count": "GDP_valid_cell_equivalents",
    "grid_count": "GUB_GUAB_mask_cell_equivalents",
}


def check_grids(total, per_capita, year):
    if total.crs is None or total.crs.to_epsg() != 4326 or total.crs != per_capita.crs:
        raise ValueError("The source GDP grids must both use WGS 84.")
    if total.shape != per_capita.shape or total.transform != per_capita.transform:
        raise ValueError("GDP total and GDP per capita do not have the same grid.")
    if total.count != 35 or per_capita.count != 35:
        raise ValueError("Expected all 35 annual bands of the 1990–2024 GDP product.")
    band = year - 1990 + 1
    if total.descriptions[band - 1] != f"gdp_tot_{year}" or per_capita.descriptions[band - 1] != f"gdp_pc_{year}":
        raise ValueError(f"Unexpected GDP band descriptions for {year}.")
    return band


def raster_window(dataset, bounds):
    if bounds is None:
        return Window(0, 0, dataset.width, dataset.height)
    raw = from_bounds(*bounds, transform=dataset.transform)
    col0 = max(0, math.floor(raw.col_off) - 1)
    row0 = max(0, math.floor(raw.row_off) - 1)
    col1 = min(dataset.width, math.ceil(raw.col_off + raw.width) + 1)
    row1 = min(dataset.height, math.ceil(raw.row_off + raw.height) + 1)
    if col1 <= col0 or row1 <= row0:
        raise ValueError("City mask falls outside the GDP release's global grid.")
    return Window(col0, row0, col1 - col0, row1 - row0)


def grid_sources(total_ds, pc_ds, band, bounds=None):
    window = raster_window(total_ds, bounds)
    total = total_ds.read(band, window=window).astype("float64")
    per_capita = pc_ds.read(band, window=window).astype("float64")
    valid = np.isfinite(total) & np.isfinite(per_capita) & (total >= 0) & (per_capita > 0)
    total[~valid] = np.nan
    per_capita[~valid] = np.nan
    population = np.divide(total, per_capita, out=np.full_like(total, np.nan), where=valid)
    left, bottom, right, top = window_bounds(window, total_ds.transform)
    extent = dict(xmin=left, ymin=bottom, xmax=right, ymax=top,
                  srs_wkt=total_ds.crs.to_wkt(), nodata=np.nan)
    return [NumPyRasterSource(total, name="gdp", **extent),
            NumPyRasterSource(population, name="pop", **extent),
            NumPyRasterSource(per_capita, name="pc", **extent),
            NumPyRasterSource(np.ones(total.shape, dtype="float32"), name="grid", **extent)]


def urban_mask(geometry, polygons, tree):
    if not geometry.is_valid:
        geometry = polygonal(shapely.make_valid(geometry))
    matches = tree.query(geometry, predicate="intersects")
    if not len(matches):
        return shapely.Polygon()
    parts = polygons[matches].copy()
    contained = shapely.covers(geometry, parts)
    parts[~contained] = shapely.intersection(parts[~contained], geometry)
    return polygonal(shapely.union_all(parts))


def aggregate(frame, sources):
    table = exact_extract(sources, frame, ["sum", "min", "max", "count"],
                          include_cols=["gid"], output="pandas", progress=False,
                          max_cells_in_memory=2_000_000)
    return table[["gid", *NAMES]].rename(columns=NAMES)


def measure_year(units, gub_root, total_ds, pc_ds, year, local):
    band = check_grids(total_ds, pc_ds, year)
    polygons = read_gub(gub_root, year, bounds=tuple(units.total_bounds) if local else None)
    invalid = ~shapely.is_valid(polygons)
    if invalid.any():
        polygons[invalid] = [polygonal(x) for x in shapely.make_valid(polygons[invalid])]
    tree = shapely.STRtree(polygons)
    all_sources = None if local else grid_sources(total_ds, pc_ds, band)
    frames = []
    for start in range(0, len(units), 250):
        group = units.iloc[start:start + 250]
        masks = [urban_mask(geometry, polygons, tree) for geometry in group.geometry]
        masked = gpd.GeoDataFrame({"gid": group.gid.to_numpy()}, geometry=masks, crs=units.crs)
        selected = masked.loc[~masked.geometry.is_empty]
        if not len(selected):
            continue
        if local:
            for i in selected.index:
                cell_sources = grid_sources(total_ds, pc_ds, band, masked.geometry[i].bounds)
                frames.append(aggregate(selected.loc[[i]], cell_sources))
        else:
            frames.append(aggregate(selected, all_sources))

    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["gid", *NAMES.values()])
    result = units.drop(columns="geometry").merge(stats, on="gid", how="left", validate="one_to_one")
    result["year"] = year
    result["GDP_available_grid_fraction"] = (result.GDP_valid_cell_equivalents /
                                             result.GUB_GUAB_mask_cell_equivalents.where(
                                                 result.GUB_GUAB_mask_cell_equivalents > 0))
    result["gdp_pc_ppp_2021usd"] = (result.gdp_total_ppp_2021usd /
                                    result.product_implied_population.where(result.product_implied_population > 0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    unit_arguments(parser)
    years_argument(parser)
    parser.add_argument("--total", type=Path, required=True, help="35-band gridded GDP total TIFF. / 含 35 个波段的 GDP 总量 TIFF")
    parser.add_argument("--per-capita", type=Path, required=True, help="35-band GDP per capita TIFF. / 含 35 个波段的人均 GDP TIFF")
    parser.add_argument("--gub", type=Path, required=True,
                        help="Parent directory of the annual GUB_Global_YEAR shapefiles. / 各年 GUB_Global_YEAR 矢量数据的上级目录")
    parser.add_argument("--output", type=Path, required=True, help="City-year .csv or .parquet. / 按城市和年份存储的 CSV 或 Parquet 文件")
    parser.add_argument("--panel", type=Path, help="Extract the cities listed in this panel Parquet file. / 提取指定 Parquet 面板中列出的城市")
    args = parser.parse_args()
    if len(set(args.year)) != len(args.year):
        parser.error("A year was specified more than once.")
    if args.panel and args.gid:
        parser.error("Choose either a panel file or explicit city identifiers.")
    gids = args.gid
    if args.panel:
        gids = pd.read_parquet(args.panel, columns=["gid"]).gid.drop_duplicates().tolist()
        if not gids:
            parser.error("The panel contains no city identifiers.")
    units = read_units(args.boundaries, gids)
    frames = []
    with rasterio.open(args.total) as total, rasterio.open(args.per_capita) as pc:
        for year in args.year:
            frames.append(measure_year(units, args.gub, total, pc, year, local=len(units) <= 32))
    write_years(frames, args.output)


if __name__ == "__main__":
    main()

