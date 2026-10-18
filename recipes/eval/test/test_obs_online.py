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

"""Tests for online observation scoring (src/obs_online.py).

The sampling and scoring tests run the external observation scorer in
process; they need a checkout providing ``private.obs_scoring``, named by
``OBS_SCORER_ROOT``, and are skipped without one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from src import obs_online
from src.obs_online import (
    ObservationSettings,
    parse_observation_settings,
    scorer_name,
)

SCORER_ROOT = os.environ.get("OBS_SCORER_ROOT")
needs_scorer = pytest.mark.skipif(
    not SCORER_ROOT or not Path(SCORER_ROOT, "private/obs_scoring").is_dir(),
    reason="OBS_SCORER_ROOT does not name a checkout with private/obs_scoring",
)
GRAVITY = 9.80665


def test_scorer_names():
    assert scorer_name("t2m") == "tas"
    assert scorer_name("z500") == "Z500"
    assert scorer_name("q850") == "Q850"
    assert scorer_name("tcwv") is None


def test_settings_default_off_and_validated():
    assert parse_observation_settings(OmegaConf.create({"scoring": {}})) is None
    cfg = OmegaConf.create(
        {
            "run_id": "arm",
            "scoring": {
                "online": {"observations": {"scorer_root": "/x", "cache": "/m"}}
            },
        }
    )
    assert parse_observation_settings(cfg).system == "arm"
    cfg.scoring.online.observations.typo = 1
    with pytest.raises(ValueError, match="typo"):
        parse_observation_settings(cfg)


def synthetic_step(tmp_path):
    lat = np.linspace(90, -89.75, 720)
    lon = np.arange(1440) * 0.25
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    fields = {
        "tas": 280 + 0.1 * la + 0.01 * lo,
        "uas": np.full_like(la, 3.0),
        "vas": np.full_like(la, 4.0),
        "Z500": GRAVITY * (5500 + la),
    }
    names = list(fields)
    values = torch.tensor(np.stack([fields[n] for n in names]), dtype=torch.float32)
    rng = np.random.default_rng(0)
    n = 40
    obs_lat = rng.uniform(30, 60, n)
    obs_lon = rng.uniform(10, 50, n)
    valid = pd.Timestamp("2026-07-15T12")
    obs = pd.DataFrame(
        {
            "time": valid,
            "lat": np.tile(obs_lat, 3),
            "lon": np.tile(obs_lon, 3),
            "station": np.tile([f"s{i}" for i in range(n)], 3),
            "elevation": 0.0,
            "dataset": "synthetic",
            "field": np.repeat(["tas", "sfcwind", "Z500"], n),
            "obs": np.concatenate(
                [
                    280 + 0.1 * obs_lat + 0.01 * obs_lon + 1.0,
                    np.full(n, 5.0),
                    5500 + obs_lat,
                ]
            ),
        }
    )
    settings = ObservationSettings(
        scorer_root=SCORER_ROOT, cache=str(tmp_path), pairs=True, system="arm"
    )
    return settings, names, values, lat, lon, valid, obs


@needs_scorer
def test_sampling_and_scores_on_a_synthetic_field(tmp_path):
    settings, names, values, lat, lon, valid, obs = synthetic_step(tmp_path)
    obs_online._init_worker(settings)
    obs_online._worker["obs"][valid] = obs
    init = np.datetime64("2026-07-15T06", "ns")
    lead = np.timedelta64(6, "h").astype("timedelta64[ns]")
    table, pairs = obs_online._score_step(init, lead, None, names, values, lat, lon)

    error = (pairs.model - pairs.obs).groupby(pairs.field)
    np.testing.assert_allclose(error.mean()["tas"], -1.0, atol=2e-4)
    np.testing.assert_allclose(
        error.apply(lambda e: e.abs().max())["sfcwind"], 0, atol=1e-5
    )
    np.testing.assert_allclose(
        error.apply(lambda e: e.abs().max())["Z500"], 0, atol=1e-2
    )
    assert set(pairs.lead) == {6} and set(pairs.init) == {pd.Timestamp(init)}

    glob = table[(table.region == "global") & (table.match == "strict")].set_index(
        "field"
    )
    assert glob.loc["tas", "count"] == 40
    np.testing.assert_allclose(glob.loc["tas", "rmse"], 1.0, atol=2e-4)
    assert (table.time == valid).all() and (table.lead_time == pd.Timedelta(lead)).all()

    root = tmp_path / "store"
    (root / "parts" / "arm").mkdir(parents=True)
    table.to_parquet(root / "parts" / "arm" / "a.parquet")
    obs_online._gather_store(settings, str(root))
    import xarray as xr

    ds = xr.open_zarr(root / "arm" / "scores_all.zarr")
    assert ds.rmse.dims == ("init", "lead_time", "match", "region", "field")
    assert ds.time.values[0, 0] == np.datetime64(valid)


@needs_scorer
def test_elevation_correction_moves_tas_to_the_station(tmp_path):
    import dataclasses

    import xarray as xr

    settings, names, values, lat, lon, valid, obs = synthetic_step(tmp_path)
    init = np.datetime64("2026-07-15T06", "ns")
    lead = np.timedelta64(6, "h").astype("timedelta64[ns]")
    model = {}
    for correct in (False, True):
        obs_online._init_worker(
            dataclasses.replace(settings, elevation_correct=correct)
        )
        obs_online._worker["obs"][valid] = obs
        _, pairs = obs_online._score_step(init, lead, None, names, values, lat, lon)
        model[correct] = pairs[pairs.field == "tas"].model.to_numpy()

    tas = obs[obs.field == "tas"]
    terrain = (
        xr.DataArray(
            obs_online._worker["fields"].era5_orography(),
            coords={"lat": np.linspace(90, -90, 721), "lon": np.arange(1440) * 0.25},
        )
        .interp(lat=xr.DataArray(tas.lat.values), lon=xr.DataArray(tas.lon.values))
        .values
    )
    # Stations at 0 m: the model's 2 m temperature warms by the lapse rate over its terrain.
    np.testing.assert_allclose(model[True] - model[False], 0.0065 * terrain, atol=0.3)


def test_resume_reruns_ics_without_observation_scores_and_overwrite_clears(tmp_path):
    cfg = OmegaConf.create(
        {
            "run_id": "arm",
            "output": {"path": str(tmp_path)},
            "scoring": {
                "online": {"observations": {"scorer_root": "/x", "cache": "/m"}}
            },
        }
    )
    settings = parse_observation_settings(cfg)
    root = obs_online.store_root(cfg, settings)
    times = [np.datetime64("2026-06-01T06"), np.datetime64("2026-06-01T18")]
    done = obs_online._done_dir(root, settings)
    done.mkdir(parents=True)
    (done / obs_online._online_marker_name(times[0])).touch()
    part = root / "parts" / settings.system / "20260601T0600_r0001.parquet"
    part.parent.mkdir(parents=True)
    part.touch()

    assert obs_online.filter_observation_completed(times, cfg) == times[1:]
    assert part in obs_online._ic_outputs(root, settings, times[0])
    obs_online.clear_observation_scores(cfg)
    assert obs_online.filter_observation_completed(times, cfg) == times
    assert not part.exists()
    off = OmegaConf.create({"scoring": {}})
    assert obs_online.filter_observation_completed(times, off) == []
