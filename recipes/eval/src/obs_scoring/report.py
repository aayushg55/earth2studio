# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pooling, comparison and plots over per-time scores and per-observation pairs."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
import pandas as pd
import xarray as xr

from . import observations
from .score import UNITS, field_units, score

PRESSURE_TICKS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000)
StationMask = Callable[[pd.DataFrame], np.ndarray]


def pooled(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    weighted = table.assign(
        se=table["rmse"] ** 2 * table["count"],
        ae=table["mae"] * table["count"],
        e=table["bias"] * table["count"],
    )
    sums = weighted.groupby(by)[["se", "ae", "e", "count"]].sum()
    return pd.DataFrame(
        {
            "rmse": np.sqrt(sums["se"] / sums["count"]),
            "mae": sums["ae"] / sums["count"],
            "bias": sums["e"] / sums["count"],
            "count": sums["count"],
        }
    ).reset_index()


def common_cycles(table: pd.DataFrame) -> pd.DataFrame:
    """Rows at the (time, region, [dataset,] field) keys every system scored with equal
    counts."""
    key = [
        "time",
        *(["match"] if "match" in table else []),
        "region",
        *(["dataset"] if "dataset" in table else []),
        "field",
        "stations",
    ]
    systems = table["system"].nunique()
    groups = table.groupby(key)["count"]
    shared = (groups.transform("size") == systems) & (groups.transform("nunique") == 1)
    return table[shared]


def group_median(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """The median of ``values`` over each report's group, per report."""
    order = np.lexsort((values, group))
    sorted_group, sorted_values = group[order], values[order]
    starts = np.flatnonzero(np.r_[True, sorted_group[1:] != sorted_group[:-1]])
    sizes = np.diff(np.r_[starts, len(order)])
    lower = sorted_values[starts + (sizes - 1) // 2]
    upper = sorted_values[starts + sizes // 2]
    out = np.empty_like(values)
    out[order] = np.repeat((lower + upper) / 2, sizes)
    return out


def consensus_keep(errors: np.ndarray, group: np.ndarray, factor: float) -> np.ndarray:
    """The consensus gross-error check: c = the median over systems of each report's
    error (``errors`` is (system, report), finite); a report is kept when c is within
    ``factor`` scaled MADs (1.4826 x MAD) of the median c of its ``group``."""
    consensus = np.median(errors, axis=0)
    deviation = np.abs(consensus - group_median(consensus, group))
    spread = 1.4826 * group_median(deviation, group)
    return deviation <= factor * spread


def check_same_reports(tables: dict, columns, where: str = "") -> None:
    """Raise unless every system's table (``{system: DataFrame or dict of arrays}``) holds
    the first system's values in ``columns``, NaN equal to NaN."""
    (first_system, first), *rest = tables.items()
    for system, table in rest:
        for column in columns:
            if not pd.Series(table[column]).equals(pd.Series(first[column])):
                raise ValueError(
                    f"{system}: {column} differs from {first_system}{where}; "
                    "score every system from the same observations"
                )


def common_pairs(
    pairs: pathlib.Path,
    factor: float | None,
    station_mask: StationMask | None = None,
    match: str = "synoptic",
):
    """Yield ``(cycle, obs, {system: model})`` on the observations every system scored,
    in the ``match`` selection (``observations.match_rows``).
    ``pairs`` holds one ``{system}.parquet`` per system. With ``factor``, observations
    whose median-over-systems error is more than ``factor`` scaled MADs from that field's
    median in that cycle are dropped; with ``station_mask``, only the rows it selects are
    kept."""
    paths = sorted(pairs.glob("*.parquet")) if pairs.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"no system pairs under {pairs}; run score --pairs")
    systems = [p.stem for p in paths]
    cycles = [set(pd.read_parquet(p, columns=["cycle"])["cycle"]) for p in paths]
    for cycle in sorted(set.intersection(*cycles)):
        frames = [
            pd.read_parquet(p, filters=[("cycle", "==", cycle)]).drop(columns="cycle")
            for p in paths
        ]
        rows = observations.match_rows(frames[0], pd.Timestamp(cycle), match)
        frames = [f[rows].reset_index(drop=True) for f in frames]
        obs = frames[0].drop(columns="model")
        check_same_reports(
            dict(zip(systems, frames)), obs.columns, f" in {pairs} at {cycle}"
        )
        errors = (
            np.stack([f["model"].to_numpy() for f in frames]) - obs["obs"].to_numpy()
        )
        keep = np.isfinite(errors).all(axis=0)
        if factor:
            field = pd.factorize(obs["field"])[0]
            keep[keep] = consensus_keep(errors[:, keep], field[keep], factor)
        if station_mask is not None:
            keep &= station_mask(obs)
        models = {s: f["model"].to_numpy()[keep] for s, f in zip(systems, frames)}
        yield pd.Timestamp(cycle), obs[keep].reset_index(drop=True), models


def from_pairs(
    pairs: pathlib.Path,
    factor: float | None,
    regions: dict | None,
    station_mask: StationMask | None = None,
    by: tuple[str, ...] = ("field",),
    match: str = "synoptic",
) -> pd.DataFrame:
    rows = []
    for time, obs, models in common_pairs(pairs, factor, station_mask, match):
        for system, model in models.items():
            table = score(obs, model, regions, by)
            table.insert(0, "time", time)
            rows.append(table.assign(system=system, match=match))
    if not rows:
        raise ValueError(f"no time under {pairs} is scored by every system")
    return pd.concat(rows, ignore_index=True)


METRICS = ["rmse", "mae", "bias", "count"]


def to_dataset(table: pd.DataFrame, leading: tuple[str, ...] = ("time",)) -> xr.Dataset:
    """One system's per-time scores on (*leading, [match,] region, [dataset,] field); count 0
    where a combination has no observations. Forecasts pass ``leading=("init", "lead")``."""
    key = [
        *leading,
        *(["match"] if "match" in table else []),
        "region",
        *(["dataset"] if "dataset" in table else []),
        "field",
    ]
    ds = table.set_index(key)[METRICS].to_xarray()
    ds["count"] = ds["count"].fillna(0).astype(np.int64)
    ds = ds.assign_coords(units=("field", [field_units(f) for f in ds.field.values]))
    ds.attrs = {
        "system": table["system"].iloc[0],
        "stations": table["stations"].iloc[0],
    }
    return ds


def from_dataset(ds: xr.Dataset) -> pd.DataFrame:
    table = ds.reset_coords("units")[[*METRICS, "units"]].to_dataframe().reset_index()
    table = table[table["count"] > 0].reset_index(drop=True)
    return table.assign(system=ds.attrs["system"], stations=ds.attrs["stations"])


def score_store(out: pathlib.Path, system: str, label: str) -> pathlib.Path:
    return out / system / f"scores_{label}.zarr"


def system_score_stores(out: pathlib.Path, label: str) -> list[pathlib.Path]:
    return sorted(out.glob(f"*/scores_{label}.zarr"))


def _profile_rows(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    rows = frame[frame.field.str.fullmatch(rf"{variable}\d+")]
    # Levels with under 1% of the variable's largest count are not plotted.
    if not rows.empty:
        rows = rows[rows["count"] >= 0.01 * rows["count"].max()]
    return rows.assign(level=rows.field.str[1:].astype(int))


def plot_profiles(
    summary: pd.DataFrame,
    out: pathlib.Path,
    label: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = [v for v in "ZTUVSQ" if not _profile_rows(summary, v).empty]
    surface = [
        f
        for f in ["tas", "sp", "uas", "vas", "sfcwind", "q2m"]
        if f in set(summary.field)
    ]
    metrics_shown = ["rmse", "bias"]
    ncols = len(variables) + bool(surface)
    if ncols == 0:
        print(f"no fields to plot for {label}")
        return
    fig, axes = plt.subplots(
        len(metrics_shown),
        ncols,
        figsize=(3.4 * ncols, 4.2 * len(metrics_shown)),
        layout="constrained",
        squeeze=False,
    )
    levels = sorted(set().union(*(_profile_rows(summary, v).level for v in variables)))
    ticks = [
        level for level in PRESSURE_TICKS if levels and levels[0] <= level <= levels[-1]
    ]
    colors = {
        system: f"C{k}" for k, system in enumerate(sorted(summary.system.unique()))
    }
    for row, metric in enumerate(metrics_shown):
        for ax, variable in zip(axes[row], variables):
            rows = _profile_rows(summary, variable)
            for system, group in rows.groupby("system"):
                group = group.sort_values("level")
                ax.plot(
                    group[metric],
                    group["level"],
                    marker="o",
                    ms=2,
                    color=colors[system],
                    label=system,
                )
            if metric != "rmse":
                ax.axvline(0, color="0.5", lw=0.6)
            ax.set_yscale("log")
            ax.set_ylim(max(levels) * 1.02, min(levels) / 1.02)
            ax.set_yticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
            units = f" ({UNITS[variable]})"
            ax.set_title(f"{variable} {metric}{units}")
        axes[row, 0].set_ylabel("hPa")
        axes[row, 0].legend(fontsize="small")
    if surface:
        rows = summary[summary.field.isin(surface)]
        systems = sorted(rows.system.unique())
        width = 0.8 / max(len(systems), 1)
        reference = rows.groupby("field")["rmse"].max().reindex(surface)
        for k, system in enumerate(systems):
            group = rows[rows.system == system].set_index("field").reindex(surface)
            axes[0, -1].bar(
                np.arange(len(surface)) + k * width,
                group["rmse"] / reference,
                width,
                label=system,
            )
        axes[0, -1].set_xticks(np.arange(len(surface)) + 0.4 - width / 2, surface)
        axes[0, -1].set_title("surface RMSE / largest")
        for ax in axes[1:, -1]:
            ax.set_visible(False)
    fig.suptitle(label, fontsize="medium")
    path = out / f"profiles_{label}.png"
    fig.savefig(path, dpi=120)
    print(path)
