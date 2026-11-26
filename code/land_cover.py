"""Reclassify GLC_FCS30D and mask it with each year's GUB urban footprints.
将 GLC_FCS30D 重分类为 RGBO 并按同期 GUB 裁切建成区。

python code/land_cover.py --stacks GLC_FCS30D --gub GUB --output masked_tiles

For a smaller run select --year and --tile. 局部提取可用 --year 和 --tile。"""

import argparse
from pathlib import Path
import re
import warnings

import numpy as np
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.windows import Window

from spatial_io import read_gub, temporary_output, years_argument


TILE_NAME = re.compile(r"[EW]\d+[NS]\d+")
CODES = np.full(256, 4, dtype="uint8")
CODES[190] = 1
CODES[[51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 120, 121, 122,
       130, 140, 150, 152, 153]] = 2
CODES[[181, 182, 183, 184, 185, 186, 187, 210]] = 3


def find_tiles(root, year, annual, selected):
    if annual:
        directory = root / f"GLC_FC30D_{year}"
        pattern = f"{year}_*.tif"
        band, length = 1, 1
    elif year < 2000:
        directory = root
        pattern = "GLC_FCS30_19852000_5yearsMap_V1_*.tif"
        band, length = 1 + (year - 1985) // 5, 3
    else:
        directory = root
        pattern = "GLC_FCS30D_20002022_*_Annual.tif"
        band, length = year - 1999, 23
    files = {}
    for path in sorted(directory.rglob(pattern)):
        name = TILE_NAME.search(path.name)
        if name is None:
            raise ValueError(f"Missing spatial tile identifier in {path.name}.")
        tile = name.group()
        if selected and tile not in selected:
            continue
        if tile in files:
            raise ValueError(f"Duplicate source tile {tile} in {year}.")
        files[tile] = path
    if not files:
        raise FileNotFoundError(f"No matching land-cover rasters for {year}.")
    if selected and (missing := set(selected) - set(files)):
        raise ValueError(f"Missing source tiles for {year}: {sorted(missing)}")
    return files, band, length


def clip_to_tile(polygons, bounds):
    tile = shapely.box(*bounds)
    pieces = []
    failed = 0
    for p in polygons:
        try:
            if p.intersects(tile):
                part = p.intersection(tile)
                if not part.is_empty:
                    pieces.append(part)
        except shapely.GEOSException:
            # The archived classifier omitted shapes that failed a tile intersection.
            failed += 1
    if failed:
        warnings.warn(f"Could not clip {failed} GUB geometries at {bounds}.", RuntimeWarning)
    return np.asarray(pieces, dtype=object)


def tile_windows(width, height):
    for row in range(0, height, 1024):
        for col in range(0, width, 1024):
            yield Window(col, row, min(1024, width - col), min(1024, height - row))


def classify_window(src, band, window, pieces, index):
    shape = (int(window.height), int(window.width))
    result = np.full(shape, 5, dtype="uint8")
    bounds = rasterio.windows.bounds(window, src.transform)
    ix = index.query(shapely.box(*bounds))
    if len(ix):
        # All unspecified source values, including fill codes, belong to O inside GUB.
        mask = rasterize(((piece, 1) for piece in pieces[ix]), out_shape=shape, fill=0,
                         transform=src.window_transform(window), all_touched=False,
                         dtype="uint8").astype(bool)
        if mask.any():
            codes = src.read(band, window=window)[mask]
            result[mask] = CODES[np.clip(codes, 0, 255).astype("uint8")]
    return result


def classify_tile(source, band, band_count, destination, year, gub, global_index=None):
    with rasterio.open(source) as src:
        if src.crs is None or src.crs.to_epsg() != 4326 or src.count != band_count:
            raise ValueError(f"Unexpected CRS or band count in {source.name}.")
        if global_index is None:
            polygons = read_gub(gub, year, bounds=tuple(src.bounds))
        else:
            all_polygons, tree = global_index
            polygons = all_polygons[tree.query(shapely.box(*src.bounds))]
        pieces = clip_to_tile(polygons, src.bounds)
        index = shapely.STRtree(pieces)
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate",
                       predictor=2, tiled=True, blockxsize=256, blockysize=256)
        with temporary_output(destination) as temporary, rasterio.open(temporary, "w", **profile) as dst:
            for window in tile_windows(src.width, src.height):
                dst.write(classify_window(src, band, window, pieces, index), 1, window=window)
            dst.set_band_description(1, f"RGBO and outside-GUB classes, {year}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stacks", type=Path, help="Directory of the two original multiband GLC products. / 存放两种 GLC 原始多波段产品的目录")
    source.add_argument("--annual-tiles", type=Path, help="Parent of GLC_FC30D_YEAR single-band directories. / 各 GLC_FC30D_YEAR 单波段年度目录的上级目录")
    parser.add_argument("--gub", type=Path, required=True,
                        help="Parent directory of the annual GUB_Global_YEAR shapefiles. / 各年 GUB_Global_YEAR 矢量数据的上级目录")
    parser.add_argument("--tile", nargs="+", help="Only these spatial tiles, for example E45S15 E50S15. / 限定要处理的空间图幅，例如 E45S15 E50S15")
    parser.add_argument("--output", type=Path, required=True, help="Directory for masked tiles by year. / 按年存放掩膜图幅的目录")
    years_argument(parser)
    args = parser.parse_args()
    if len(set(args.year)) != len(args.year):
        parser.error("A year was specified more than once.")
    with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024, GDAL_NUM_THREADS="1"):
        for year in args.year:
            root = args.annual_tiles or args.stacks
            files, band, length = find_tiles(root, year, args.annual_tiles is not None, args.tile)
            global_index = None
            if not args.tile:
                polygons = read_gub(args.gub, year)
                global_index = polygons, shapely.STRtree(polygons)
            for name, path in files.items():
                destination = args.output / str(year) / f"{year}_{name}.tif"
                if destination.resolve() == path.resolve():
                    raise ValueError("Input and output raster paths must differ.")
                classify_tile(path, band, length, destination, year, args.gub, global_index)


if __name__ == "__main__":
    main()

