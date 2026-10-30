# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Online scoring against observations.

At every lead step of an online run the forecast is scored against the
observations valid at ``IC + lead``, without writing the forecast.  The
observation methodology (reading, quality control, the time window, the
interpolation to observation points and the per-region scores) is the
``obs_scoring`` package beside this module, whose ``score.cycle_observations``
and ``score.score_cycle`` this module calls on the in-memory forecast.  They
run in spawned worker processes, so their CPU work overlaps the next model
step.

Per IC and rank, the scores (the scorer's per-cycle table plus ``init``,
``lead_time`` and, for ensembles, ``member``) and optionally the matched
pairs (the scorer's pairs columns plus ``lead`` in hours and ``init``) are
written under ``output.path/<store_dir>``.  :func:`finalize_observation_scores`
gathers the score tables into ``<store_dir>/<system>/scores_<label>.zarr``,
the scorer's store layout with leading ``(init, lead_time[, member])``
dimensions in place of ``time`` and the valid time as a ``time``
coordinate.

Enabled by a ``scoring.online.observations`` block; absent or null keeps
it off::

    scoring:
      online:
        observations:
          obs: conventional                     # or vad / aircraft
          cache: /path/to/nnja-ai-mirror        # obs=conventional
          archive: null                         # PrepBUFR archive, obs=vad/aircraft
          stations: all                         # all | withheld | kept
          by_dataset: false
          dedupe_surface: false
          elevation_correct: false              # tas / sp to the station elevation
          vertical: pressure                    # pressure | height
          pairs: false
          system: null                          # null -> run_id
          store_dir: obs_scores
          workers: 2
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as tmp
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.utils.coords import CoordSystem

from .work import WorkItem, _online_marker_name

# earth2studio surface names -> observation-scorer channel names.  Pressure-level
# variables map by rule: z500 -> Z500 (geopotential), t/u/v/q likewise.
SURFACE_NAMES = {
    "t2m": "tas",
    "u10m": "uas",
    "v10m": "vas",
    "d2m": "d2m",
    "sp": "sp",
    "msl": "pres_msl",
}
_LEVEL_RE = re.compile(r"^([zqtuv])(\d+)$")


def scorer_name(variable: str) -> str | None:
    """Observation-scorer channel name of an earth2studio variable, or None."""
    if variable in SURFACE_NAMES:
        return SURFACE_NAMES[variable]
    match = _LEVEL_RE.match(variable)
    if match is None:
        return None
    return f"{match.group(1).upper()}{match.group(2)}"


@dataclass(frozen=True)
class ObservationSettings:
    """Parsed ``scoring.online.observations`` block (see the module docstring)."""

    obs: str = "conventional"
    cache: str | None = None
    archive: str | None = None
    stations: str = "all"
    by_dataset: bool = False
    dedupe_surface: bool = False
    elevation_correct: bool = False
    vertical: str = "pressure"
    pairs: bool = False
    system: str = "forecast"
    store_dir: str = "obs_scores"
    workers: int = 2

    @property
    def label(self) -> str:
        return f"{self.stations}_by_dataset" if self.by_dataset else self.stations

    @property
    def by(self) -> tuple[str, ...]:
        return ("dataset", "field") if self.by_dataset else ("field",)

    @property
    def source(self) -> tuple[str, Path | str]:
        location = self.cache if self.obs == "conventional" else self.archive
        if location is None:
            key = "cache" if self.obs == "conventional" else "archive"
            raise ValueError(
                f"scoring.online.observations.{key} is required for obs={self.obs}."
            )
        return self.obs, Path(location) if self.obs == "conventional" else location


def parse_observation_settings(cfg: DictConfig) -> ObservationSettings | None:
    """The ``scoring.online.observations`` block, or None when it is absent."""
    block = cfg.get("scoring", {}).get("online", {}) or {}
    obs_cfg = block.get("observations", None)
    if obs_cfg is None:
        return None
    opts = OmegaConf.to_container(obs_cfg, resolve=True)
    known = set(ObservationSettings.__dataclass_fields__)
    unknown = set(opts) - known
    if unknown:
        raise ValueError(
            f"Unknown scoring.online.observations option(s) {sorted(unknown)}."
        )
    if opts.get("system") is None:
        opts["system"] = str(cfg.get("run_id", "forecast"))
    settings = ObservationSettings(**opts)
    settings.source  # noqa: B018  (validates cache / archive up front)
    return settings


def store_root(cfg: DictConfig, settings: ObservationSettings) -> Path:
    return Path(cfg.output.path) / settings.store_dir


def _done_dir(root: Path, settings: ObservationSettings) -> Path:
    return root / "done" / settings.system


def _ic_outputs(root: Path, settings: ObservationSettings, time) -> list[Path]:
    """Every part, pairs file and done marker of one IC, from any rank."""
    stamp = f"{pd.Timestamp(time):%Y%m%dT%H%M}_r*.parquet"
    dirs = [
        root / "parts" / settings.system,
        root / "pairs" / settings.stations / settings.system,
    ]
    files = [f for d in dirs for f in d.glob(stamp)]
    marker = _done_dir(root, settings) / _online_marker_name(np.datetime64(time))
    return files + ([marker] if marker.exists() else [])


def filter_observation_completed(
    times: list[np.datetime64], cfg: DictConfig
) -> list[np.datetime64]:
    """IC times without an observation-score done marker; none when scoring is off."""
    settings = parse_observation_settings(cfg)
    if settings is None:
        return []
    done = _done_dir(store_root(cfg, settings), settings)
    have = {f.name for f in done.glob("*.done")} if done.exists() else set()
    return [t for t in times if _online_marker_name(t) not in have]


def clear_observation_scores(cfg: DictConfig) -> None:
    """Remove every part, pairs file and done marker, for an overwrite run."""
    import shutil

    settings = parse_observation_settings(cfg)
    if settings is None:
        return
    root = store_root(cfg, settings)
    for d in (
        root / "parts" / settings.system,
        root / "pairs" / settings.stations / settings.system,
        _done_dir(root, settings),
    ):
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker side: runs in spawned processes importing the scorer package.
# ---------------------------------------------------------------------------

_worker: dict[str, Any] = {}


def _init_worker(settings: ObservationSettings) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from .obs_scoring import fields, score

    _worker.update(
        settings=settings,
        score=score,
        fields=fields,
        regions=score.standard_regions(),
        obs={},
        orography={},
    )


def _orography(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """ERA5 terrain on the forecast grid, kept per grid."""
    key = (lat.size, float(lat[0]), lon.size, float(lon[0]))
    cache = _worker["orography"]
    if key not in cache:
        cache[key] = _worker["fields"].era5_orography_on(lat, lon)
    return cache[key]


def _cycle_obs(valid: pd.Timestamp) -> pd.DataFrame | None:
    """Observations of one valid time, kept for the next few ICs that share it."""
    cache = _worker["obs"]
    if valid not in cache:
        s = _worker["settings"]
        cache[valid] = _worker["score"].cycle_observations(
            s.source, valid, s.stations, s.dedupe_surface
        )
        while len(cache) > 64:
            cache.pop(next(iter(cache)))
    return cache[valid]


def _score_step(
    init: np.datetime64,
    lead: np.timedelta64,
    member: int | None,
    names: Sequence[str],
    values: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame | None] | None:
    """Scores (and pairs) of one (IC, lead, member) field set; None without observations."""
    import xarray as xr

    s = _worker["settings"]
    valid = pd.Timestamp(init + lead)
    obs = _cycle_obs(valid)
    if obs is None or obs.empty:
        return None
    index = {n: i for i, n in enumerate(names)}
    array = values.numpy()

    def fields_at(wanted: list[str]) -> xr.Dataset:
        data = {n: (("lat", "lon"), array[index[n]]) for n in wanted if n in index}
        if "orography" in wanted:
            data["orography"] = (("lat", "lon"), _orography(lat, lon))
        return xr.Dataset(data, coords={"lat": lat, "lon": lon})

    table, paired = _worker["score"].score_cycle(
        fields_at,
        obs,
        valid,
        _worker["regions"],
        s.by,
        s.vertical,
        s.elevation_correct,
        s.pairs,
    )
    extra = {"init": pd.Timestamp(init), "lead_time": pd.Timedelta(lead)}
    if member is not None:
        extra["member"] = member
    table = table.assign(**extra)
    if paired is not None:
        paired = paired.assign(
            lead=np.int16(pd.Timedelta(lead) // pd.Timedelta(hours=1)),
            init=pd.Timestamp(init),
            **({"member": member} if member is not None else {}),
        )
    return table, paired


def _gather_store(settings: ObservationSettings, root: str) -> str:
    """Score tables under *root*/parts -> the scorer's store layout (worker side)."""
    from .obs_scoring import report

    parts = sorted((Path(root) / "parts" / settings.system).glob("*.parquet"))
    if not parts:
        return "no observation score parts"
    table = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    leading = ["init", "lead_time", *(["member"] if "member" in table else [])]
    valid = table.drop_duplicates(["init", "lead_time"]).set_index(
        ["init", "lead_time"]
    )["time"]
    table = table.drop(columns="time").assign(
        system=settings.system, stations=settings.stations
    )
    ds = report.to_dataset(table, leading=tuple(leading))
    ds = ds.assign_coords(
        time=valid.to_xarray().reindex(init=ds.init, lead_time=ds.lead_time)
    )
    path = report.score_store(Path(root), settings.system, settings.label)
    ds.to_zarr(path, mode="w")
    return f"{len(parts)} IC parts -> {path}"


# ---------------------------------------------------------------------------
# Forecast side
# ---------------------------------------------------------------------------


def _spawn_pool(settings: ObservationSettings, workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=tmp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(settings,),
    )


class ObservationScorer:
    """Scores each yielded forecast chunk against observations (see module docstring).

    Driven like :class:`src.online.OnlineScorer`: :meth:`begin_item`,
    :meth:`update` per yielded chunk, :meth:`finish_item`.  Every rank scores
    its own members; nothing is reduced across ranks.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    settings : ObservationSettings
        Parsed ``scoring.online.observations`` block.
    ensemble_size : int
        Ensemble size; above 1 the outputs carry a ``member`` column.
    rank : int
        Global rank, for per-rank output file names.
    comm : GroupComm
        The ensemble group: its root clears an IC's earlier outputs and writes
        the IC's done marker once every rank of the group has written.
    device : torch.device
        Device the group's collectives are staged on.
    """

    def __init__(
        self,
        cfg: DictConfig,
        settings: ObservationSettings,
        ensemble_size: int,
        rank: int,
        comm: Any,
        device: torch.device,
    ) -> None:
        self._settings = settings
        self._root = store_root(cfg, settings)
        self._members = ensemble_size > 1
        self._rank = rank
        self._comm = comm
        self._device = device
        self._pool = _spawn_pool(settings, settings.workers)
        self._item: WorkItem | None = None
        self._pending: list[Future] = []

    def begin_item(self, item: WorkItem) -> None:
        # Outputs of an earlier, unfinished attempt at this IC, possibly from
        # other ranks. The group runs an IC in lockstep, so no peer has written
        # this IC's outputs before the root clears them here.
        if self._comm.is_root:
            for path in _ic_outputs(self._root, self._settings, item.time):
                path.unlink(missing_ok=True)
        self._item = item
        self._pending = []

    def update(self, x: torch.Tensor, coords: CoordSystem) -> None:
        if self._item is None:
            raise RuntimeError("ObservationScorer.update called before begin_item.")
        dims = list(coords)
        for dim in ("batch", "time"):
            if dim in dims:
                x = x.squeeze(dims.index(dim))
                dims.remove(dim)
        if "ensemble" not in dims:
            x = x.unsqueeze(0)
            dims.insert(0, "ensemble")
            members = [self._item.ensemble_id]
        else:
            members = [int(m) for m in coords["ensemble"]]
        x = x.permute(
            dims.index("lead_time"),
            dims.index("ensemble"),
            dims.index("variable"),
            dims.index("lat"),
            dims.index("lon"),
        )
        variables = [str(v) for v in coords["variable"]]
        keep = [(i, scorer_name(v)) for i, v in enumerate(variables)]
        keep = [(i, n) for i, n in keep if n is not None]
        index = torch.tensor([i for i, _ in keep], device=x.device)
        names = [n for _, n in keep]
        lat, lon = np.asarray(coords["lat"]), np.asarray(coords["lon"])
        init = np.datetime64(self._item.time, "ns")
        for k, lead in enumerate(np.asarray(coords["lead_time"])):
            for m, member in enumerate(members):
                values = x[k, m].index_select(0, index).float().cpu().share_memory_()
                self._pending.append(
                    self._pool.submit(
                        _score_step,
                        init,
                        np.timedelta64(lead, "ns"),
                        member if self._members else None,
                        names,
                        values,
                        lat,
                        lon,
                    )
                )

    def finish_item(self, item: WorkItem) -> None:
        self._write_item(item)
        # Every rank of the group has written its members before the root
        # marks the IC done.
        self._comm.all_agree(1, self._device)
        if self._comm.is_root:
            marker = _done_dir(self._root, self._settings) / _online_marker_name(
                np.datetime64(item.time)
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        self._item = None

    def _write_item(self, item: WorkItem) -> None:
        steps = len(self._pending)
        results = [f.result() for f in self._pending]
        self._pending = []
        results = [r for r in results if r is not None]
        if len(results) < steps:
            logger.warning(
                f"IC {item.time}: {steps - len(results)} of {steps} lead steps "
                "had no observations"
            )
        if not results:
            return
        name = f"{pd.Timestamp(item.time):%Y%m%dT%H%M}_r{self._rank:04d}.parquet"
        _write(
            pd.concat([t for t, _ in results], ignore_index=True),
            self._root / "parts" / self._settings.system / name,
        )
        if self._settings.pairs:
            pairs = [p for _, p in results if p is not None and len(p)]
            if pairs:
                _write(
                    pd.concat(pairs, ignore_index=True),
                    self._root
                    / "pairs"
                    / self._settings.stations
                    / self._settings.system
                    / name,
                )

    def close(self) -> None:
        self._pool.shutdown()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp_path, index=False, compression="zstd")
    tmp_path.rename(path)


class WithObservations:
    """An online scorer followed by an :class:`ObservationScorer` on the same chunks.

    The observation outputs of an IC are written before the inner scorer
    marks the IC done, so a resumed run never skips an IC whose observation
    scores are missing.
    """

    def __init__(self, inner: Any, observations: ObservationScorer) -> None:
        self._inner = inner
        self._observations = observations

    def begin_item(self, item: WorkItem) -> None:
        self._inner.begin_item(item)
        self._observations.begin_item(item)

    def update(self, x: torch.Tensor, coords: CoordSystem) -> None:
        self._observations.update(x, coords)
        self._inner.update(x, coords)

    def finish_item(self, item: WorkItem) -> None:
        self._observations.finish_item(item)
        self._inner.finish_item(item)

    def close(self) -> None:
        self._observations.close()


def finalize_observation_scores(cfg: DictConfig) -> None:
    """Gather the per-IC observation score tables into the scorer's zarr layout."""
    settings = parse_observation_settings(cfg)
    if settings is None:
        return
    with _spawn_pool(settings, 1) as pool:
        message = pool.submit(
            _gather_store, settings, str(store_root(cfg, settings))
        ).result()
    logger.info(f"Observation scores: {message}")
