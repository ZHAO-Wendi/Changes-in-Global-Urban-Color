"""Construct the seven-date panel and calculate the study's core statistics.
构建七期平衡面板，统计城市色彩组成、跨期变化、建成区规模效应和 GDP 相关性。

python code/study_statistics.py --areas areas --out stats
python code/study_statistics.py --areas areas --gdp city_gdp.parquet --out stats

--groups accepts a city-level table keyed by gid with any of continent, hemisphere,
region, income_group and development_group. --income accepts an economy,fy19 CSV.
--groups 可传入按 gid 匹配的地理或收入分组，--income 可传入含 economy 和 fy19 的收入分类表。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

YEARS = (1990, 1995, 2000, 2005, 2010, 2015, 2018)
CHANNELS = list("RGBO")
AREA_COLUMNS = ["red_km2", "green_km2", "blue_km2", "other_km2"]
GROUP_FIELDS = ("continent", "hemisphere", "region", "income_group", "development_group")
PHASES = [(1990, 2018), (1990, 2000), (2000, 2018), (2000, 2015), (2015, 2018)]
SEED = 20260929


def read_table(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, float_precision="round_trip")


def read_panel(path):
    if path.is_dir():
        frames = []
        for year in YEARS:
            data = read_table(path / f"all_cities_{year}.csv")
            if not data.year.eq(year).all():
                raise ValueError(f"Conflicting observation years in the {year} file.")
            frames.append(data)
        areas = pd.concat(frames, ignore_index=True)
    else:
        areas = read_table(path)
    fields = ["gid", "country", "year", "admin_km2", *AREA_COLUMNS]
    areas = areas[fields].copy()
    if areas.gid.isna().any() or areas.gid.eq("").any() or areas.duplicated(["gid", "year"]).any():
        raise ValueError("Missing or repeated city-year key.")
    if set(areas.year) != set(YEARS):
        raise ValueError("Expected all seven observation years from 1990 to 2018.")
    numeric = areas[["admin_km2", *AREA_COLUMNS]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
        raise ValueError("City areas must be finite and nonnegative.")
    areas[numeric.columns] = numeric
    areas["built_km2"] = areas[AREA_COLUMNS].sum(axis=1)
    positive = areas.loc[areas.built_km2.gt(0)].copy()
    observed = positive.groupby("gid").year.transform("nunique")
    panel = positive.loc[observed.eq(len(YEARS))].copy()
    if panel.empty or panel.admin_km2.le(0).any():
        raise ValueError("Missing balanced sample or non-positive administrative area.")
    for field, channel in zip(AREA_COLUMNS, CHANNELS):
        panel[channel] = 100 * panel[field] / panel.built_km2
    return panel.sort_values(["year", "gid"]).reset_index(drop=True)


def add_groups(panel, groups, income):
    if groups is not None:
        labels = read_table(groups)
        fields = [f for f in GROUP_FIELDS if f in labels]
        if not fields or labels.gid.duplicated().any() or not set(panel.gid).issubset(labels.gid):
            raise ValueError("Supply unique groups covering all study city IDs.")
        panel = panel.merge(labels[["gid", *fields]], on="gid", how="left", validate="many_to_one")
    if income is not None:
        table = read_table(income)
        if table.economy.duplicated().any() or "income_group" in panel or "development_group" in panel:
            raise ValueError("Supply one source of income assignments and unique economy codes.")
        economy = panel.gid.str.split(".").str[0].str.split("_").str[0].copy()
        economy.loc[panel.country.eq("Ukraine")] = "UKR"
        for gid, code in {"CHN.HKG": "HKG", "CHN.MAC": "MAC"}.items():
            economy.loc[panel.gid.eq(gid)] = code
        codes = economy.map(table.set_index("economy").fy19)
        names = {"H": "High income", "UM": "Upper middle income",
                 "LM": "Lower middle income", "L": "Low income"}
        panel["income_group"] = codes.map(names).fillna("Unclassified")
        panel["development_group"] = np.select(
            [codes.eq("H"), codes.isin(["UM", "LM", "L"])],
            ["High income", "Low and middle income"], default="Unclassified")
    return panel


def compositions(panel):
    tables = []
    for group in ["global", *(name for name in GROUP_FIELDS if name in panel), "country"]:
        data = panel.assign(global_group="Global") if group == "global" else panel
        if group == "region":
            data = data[data.region.ne("")]
        key = "global_group" if group == "global" else group
        grouped = data.groupby([key, "year"], observed=True)
        total_area = grouped.built_km2.sum()
        city = grouped[CHANNELS].mean()
        pooled = grouped[AREA_COLUMNS].sum().div(total_area, axis=0) * 100
        pooled.columns = CHANNELS
        for label, values in [("city_equal", city), ("area_pooled", pooled)]:
            values = values.assign(n=grouped.size(), inside_km2=total_area, weighting=label, group_type=group)
            tables.append(values.reset_index().rename(columns={key: "group"}))
    return pd.concat(tables, ignore_index=True)


def paired_changes(panel):
    countries = np.sort(panel.country.unique())
    random = np.random.default_rng(SEED)
    frequencies = random.multinomial(len(countries), np.full(len(countries), 1 / len(countries)), size=499)
    endpoints = {year: group.set_index("gid") for year, group in panel.groupby("year")}
    intervals = []
    for first, last in PHASES:
        change = endpoints[last][CHANNELS] - endpoints[first][CHANNELS]
        change["country"] = endpoints[last].country
        by_country = change.groupby("country")
        sums = by_country[CHANNELS].sum().reindex(countries)
        counts = by_country.size().reindex(countries)
        sample = (frequencies @ sums.to_numpy()) / (frequencies @ counts.to_numpy())[:, None]
        for i, channel in enumerate(CHANNELS):
            lower, upper = np.quantile(sample[:, i], [0.025, 0.975])
            intervals.append(dict(start=first, end=last, channel=channel, n=len(change),
                                  mean_pp=change[channel].mean(), increase_pct=100 * change[channel].gt(0).mean(),
                                  ci_low=lower, ci_high=upper))
    change = endpoints[2018][CHANNELS] - endpoints[1990][CHANNELS]
    change.columns = ["d" + c for c in CHANNELS]
    red, green = change.dR, change.dG
    change["quadrant"] = np.select([red.gt(0) & green.gt(0), red.lt(0) & green.gt(0),
                                     red.lt(0) & green.lt(0), red.gt(0) & green.lt(0)],
                                    ["R+G+", "R-G+", "R-G-", "R+G-"], default="Axis")
    change["material_quadrant"] = change.quadrant.where(red.abs().ge(0.5) & green.abs().ge(0.5), "Near axis")
    return pd.DataFrame(intervals), change.reset_index()


def gdp_associations(panel, city_gdp):
    data = panel.merge(city_gdp, on=["gid", "year"], how="left", validate="one_to_one", indicator=True)
    if data._merge.ne("both").any() or len(data) != len(city_gdp):
        raise ValueError("GDP and panel identifiers must cover the same city-years, including missing estimates.")
    total, population = data.gdp_total_ppp_2021usd, data.product_implied_population
    data["GDPpc"] = (total / population.where(population.gt(0))).where(total.gt(0))
    data["log_GDPpc"] = np.log(data.GDPpc)
    end = data[data.year.eq(2018)].set_index("gid")
    start = data[data.year.eq(1990)].set_index("gid")
    cross = end[["GDPpc", "log_GDPpc", *list("RGB")]].replace([np.inf, -np.inf], np.nan).dropna()
    columns = ["log_GDPpc", *list("RGB")]
    delta = (end[columns] - start[columns]).replace([np.inf, -np.inf], np.nan).dropna()
    records = []
    for scope, values in [("2018", cross), ("1990_2018", delta)]:
        for channel in "RGB":
            fit = spearmanr(values.log_GDPpc, values[channel], alternative="two-sided")
            records.append(dict(scope=scope, channel=channel, n=len(values), rho=fit.statistic, p_value=fit.pvalue))
    return pd.DataFrame(records), cross.reset_index(), delta.reset_index()


def covariates(x, knots):
    hinges = np.maximum(x[:, None] - knots, 0)
    return np.stack([np.ones_like(hinges), np.broadcast_to(x[:, None], hinges.shape), hinges], axis=-1)


def estimate(frame, within=False, weights=None):
    x, y = np.log10(frame.built_km2.to_numpy()), frame.G.to_numpy()
    countries, labels = np.unique(frame.country, return_inverse=True)
    grid = np.linspace(np.quantile(x, 0.10), np.quantile(x, min(0.995, 1 - 30 / len(x))), 161)
    normals, products, squares = [], [], []
    groups = [np.flatnonzero(labels == index) for index in range(len(countries))]
    for positions in groups:
        design = covariates(x[positions], grid)
        response = y[positions].copy()
        if within:
            design[:, :, 1:] -= design[:, :, 1:].mean(axis=0)
            response -= response.mean()
        normals.append(np.einsum("nki,nkj->kij", design, design))
        products.append(np.einsum("nki,n->ki", design, response))
        squares.append(response @ response)
    normals, products, squares = map(np.asarray, (normals, products, squares))

    def grid_fit(frequencies):
        lhs = np.einsum("c,ckij->kij", frequencies, normals)
        rhs = np.einsum("c,cki->ki", frequencies, products)
        try:
            coefficients = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficients = np.einsum("kij,kj->ki", np.linalg.pinv(lhs), rhs)
        residual = frequencies @ squares - np.einsum("ki,ki->k", coefficients, rhs)
        best = residual.argmin()
        return grid[best], coefficients[best]

    def residual(knot=None):
        design = np.column_stack([np.ones(len(x)), x])
        if knot is not None:
            design = np.column_stack([design, np.maximum(x - knot, 0)])
        response = y.copy()
        if within:
            for positions in groups:
                design[positions, 1:] -= design[positions, 1:].mean(axis=0)
                response[positions] -= response[positions].mean()
        coef = np.linalg.lstsq(design, response, rcond=None)[0]
        return np.sum((response - design @ coef) ** 2), coef

    initial, _ = grid_fit(np.ones(len(countries)))
    step = grid[1] - grid[0]
    fit = minimize_scalar(lambda knot: residual(knot)[0], method="bounded",
                          bounds=(max(grid[0], initial - step), min(grid[-1], initial + step)))
    knot = fit.x
    rss, (intercept, before, hinge) = residual(knot)
    rss_linear, _ = residual()
    n = len(frame)
    result = dict(n=n, n_country=len(countries), knot_km2=10**knot, intercept=intercept,
                  slope_before=before, slope_after=before + hinge, n_above=(x > knot).sum(),
                  delta_bic=n * np.log(rss_linear / rss) - 2 * np.log(n), within_country=within)
    samples = []
    if weights is not None:
        for i, count in enumerate(weights):
            k, coef = grid_fit(count)
            samples.append((i, 10**k, coef[1], coef[1] + coef[2]))
        bootstrap = pd.DataFrame(samples, columns=["replicate", "knot_km2", "slope_before", "slope_after"])
        result["ci_low"], result["ci_high"] = bootstrap.knot_km2.quantile([0.025, 0.975])
        result["peak_bootstrap_pct"] = 100 * (bootstrap.slope_before.gt(0) & bootstrap.slope_after.lt(0)).mean()
        return result, bootstrap
    return result, None


def size_results(panel):
    minimum = panel.groupby("gid").built_km2.min()
    fixed = panel[panel.gid.isin(minimum.index[minimum.ge(1)])]
    if fixed.empty:
        return pd.DataFrame()
    countries = fixed.country.nunique()
    weights = np.random.default_rng(SEED).multinomial(
        countries, np.full(countries, 1 / countries), size=499)

    def identifiable(frame):
        if len(frame) <= 33:
            return False
        x = np.log10(frame.built_km2)
        return x.quantile(0.1) < x.quantile(min(0.995, 1 - 30 / len(x))) and frame.G.nunique() > 1

    fits = []
    for year, group in fixed.groupby("year"):
        if identifiable(group):
            fit, _ = estimate(group, weights=weights)
            fits.append(dict(scope="fixed_ge1", year=year, **fit))
    final = fixed[fixed.year.eq(2018)]
    all_final = panel[panel.year.eq(2018)]
    for scope, group, within in [
        ("country_fixed_effects", final, True),
        ("without_China", final[final.country.ne("China")], False),
        ("without_US", final[final.country.ne("United States")], False),
        ("all_positive_area", all_final, False),
        ("all_dates_ge5", all_final[all_final.gid.isin(minimum.index[minimum.ge(5)])], False),
    ]:
        if identifiable(group):
            fit, _ = estimate(group, within=within)
            fits.append(dict(scope=scope, year=2018, **fit))
    return pd.DataFrame(fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--areas", type=Path, required=True, help="Annual area directory or combined table / 年度面积目录或汇总表")
    parser.add_argument("--out", type=Path, required=True, help="Output directory / 结果目录")
    parser.add_argument("--gdp", type=Path, help="City-year GDP estimates / 城市年度 GDP 提取结果")
    parser.add_argument("--groups", type=Path, help="City-level grouping labels / 城市分组对照")
    parser.add_argument("--income", type=Path, help="World Bank FY19 economy codes / 世界银行 FY19 经济体编码表")
    args = parser.parse_args()
    panel = add_groups(read_panel(args.areas), args.groups, args.income)
    args.out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out / "panel.parquet", index=False)
    compositions(panel).to_csv(args.out / "composition.csv", index=False)
    phases, change = paired_changes(panel)
    phases.to_csv(args.out / "paired_change.csv", index=False)
    for column in ["quadrant", "material_quadrant"]:
        counts = change.groupby(column).size().rename("n").reset_index()
        counts["pct"] = 100 * counts.n / len(change)
        counts.to_csv(args.out / f"{column}.csv", index=False)
    final = panel.loc[panel.year.eq(2018)]
    blue_ids = final.loc[final.B.ge(final.B.quantile(0.95)) & final.built_km2.ge(10)
                         & final.blue_km2.ge(1), "gid"]
    blue = panel[panel.gid.isin(blue_ids)].groupby("year")
    blue[CHANNELS].mean().assign(n=blue.size(), blue_area_km2=blue.blue_km2.sum()).to_csv(
        args.out / "blue_trajectory.csv")
    fits = size_results(panel)
    if not fits.empty:
        fits.to_csv(args.out / "size_models.csv", index=False)
    if args.gdp is not None:
        gdp = read_table(args.gdp)
        keys = ["gid", "year"]
        values = ["gdp_total_ppp_2021usd", "product_implied_population"]
        gdp[values] = gdp[values].replace("", np.nan).apply(pd.to_numeric, errors="raise")
        gdp = gdp.merge(panel[keys], on=keys, how="inner", validate="one_to_one")
        correlations, _, _ = gdp_associations(panel, gdp)
        correlations.to_csv(args.out / "gdp_correlations.csv", index=False)


if __name__ == "__main__":
    main()
