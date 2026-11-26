"""Common city and GUB readers, grid-year selection and tabular output.
土地覆盖和 GDP 提取共用的边界读取、年份选择与结果写出函数。"""

import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyogrio
import shapely


YEARS = (1990, 1995, 2000, 2005, 2010, 2015, 2018)


def years_argument(parser):
    parser.add_argument("--year", type=int, choices=YEARS, nargs="+", default=YEARS,
                        help="One or more observation years; default: all seven. / 指定一个或多个观测年份，默认处理全部七期")


def unit_arguments(parser):
    parser.add_argument("--boundaries", type=Path, required=True,
                        help="GUAB GeoPackage, layer cities. / 含 cities 图层的 GUAB GeoPackage 文件")
    parser.add_argument("--gid", nargs="+", help="Restrict the run to these city identifiers. / 只提取指定编码对应的城市")


def read_units(path, gids, attributes=()):
    columns = ["gid", *attributes]
    where = None
    if gids:
        if len(set(gids)) != len(gids):
            raise ValueError("Duplicate city identifier in the requested selection.")
        terms = ["'" + gid.replace("'", "''") + "'" for gid in gids]
        where = "gid IN (" + ", ".join(terms) + ")"
    units = pyogrio.read_dataframe(path, layer="cities", columns=columns, where=where)
    if units.crs is None or units.crs.to_epsg() != 4326:
        raise ValueError("Expected WGS 84 city geometries.")
    if units.empty or units.gid.isna().any() or not units.gid.is_unique:
        raise ValueError("City layer is empty or has missing or duplicate identifiers.")
    if gids and set(gids) != set(units.gid):
        raise ValueError(f"Unknown city identifiers: {sorted(set(gids) - set(units.gid))}")
    if units.geometry.isna().any() or units.geometry.is_empty.any():
        raise ValueError("Empty city geometry in the selected units.")
    return units


def gub_path(root, year):
    return root / f"GUB_Global_{year}" / f"GUB_Global_{year}.shp"


def read_gub(root, year, bounds=None):
    path = gub_path(root, year)
    frame = pyogrio.read_dataframe(path, bbox=bounds, columns=[])
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise ValueError(f"Expected WGS 84 urban boundaries in {path.name}.")
    return frame.geometry.dropna().loc[lambda g: ~g.is_empty].to_numpy(copy=True)


def polygonal(geometry):
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    if geometry.geom_type == "GeometryCollection":
        return shapely.union_all([polygonal(part) for part in geometry.geoms])
    return shapely.Polygon()


@contextmanager
def temporary_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}{path.suffix}")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_table(table, path):
    with temporary_output(path) as temporary:
        if path.suffix.lower() == ".parquet":
            table.to_parquet(temporary, index=False)
        elif path.suffix.lower() == ".csv":
            table.to_csv(temporary, index=False, encoding="utf-8")
        else:
            raise ValueError("Choose a .csv or .parquet output file.")


def write_years(frames, path):
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["gid", "year"]).any():
        raise ValueError("Expected one row per city and year.")
    write_table(result, path)

