# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
Aurora from a HealDA-v2 Analysis
================================

Driving a two-step prognostic model from analyses assimilated out of raw observations.

Aurora needs two states six hours apart rather than one, and wants 720 latitudes
rather than 721. This example assimilates two HealDA-v2 analyses to supply them, and
is the counterpart to
:ref:`sphx_glr_examples_05_data_assimilation_04_healda_v2_fcn3.py`, where the
prognostic model takes a single state.

In this example you will learn:

- How to assemble a multiple-lead-time initial condition from repeated assimilation
- How a prognostic model's latitude grid is matched without hand-written slicing
- Rolling Aurora forward from an observation-only initialization
"""

# /// script
# dependencies = [
#   "earth2studio[da-healda-v2,aurora] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA-v2 :py:class:`earth2studio.models.da.HealDAv2`.
# - Prognostic Model: Aurora :py:class:`earth2studio.models.px.Aurora`.
# - Datasource (conv): UFS conventional observations
#   :py:class:`earth2studio.data.UFSObsConv`.
# - Datasource (sat): UFS satellite observations
#   :py:class:`earth2studio.data.UFSObsSat`.
# - Datasource (verification): ERA5 :py:class:`earth2studio.data.NCAR_ERA5`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
import sys
import time
from collections import OrderedDict

from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# The DA model broadcasts each finished analysis to every rank, so under torchrun rank 0
# narrates and the other ranks report only failures.
rank = int(os.environ.get("RANK", 0))
logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    colorize=True,
    level="INFO" if rank == 0 else "WARNING",
)

from earth2studio.data import NCAR_ERA5, UFSObsConv, UFSObsSat, fetch_dataframe
from earth2studio.data.utils import prep_data_array
from earth2studio.models.da import HealDAv2
from earth2studio.models.px import Aurora
from earth2studio.utils.coords import map_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# HealDA-v2 predicts on a HEALPix level-6 grid. ``lat_lon=True`` regrids the output, in
# one bilinear step, to the equiangular grid given by ``output_resolution``: here 0.25
# degrees, the resolution Aurora expects.

# %%
da_model = HealDAv2.load_model(
    HealDAv2.load_default_package(), lat_lon=True, output_resolution=(721, 1440)
).to(device)
px_model = Aurora.load_model(Aurora.load_default_package()).to(device)

# %%
# Assimilate One Analysis per Lead Time
# -------------------------------------
# Aurora's ``input_coords`` declares lead times ``-6h`` and ``0h``, both relative to
# the initialization time, so it wants the state now and the state six hours ago. Each
# is a separate assimilation over its own 48-hour observation window, which is why
# this example runs the DA model twice.
#
# :py:meth:`~earth2studio.models.da.HealDAv2.input_time_tolerance` gives the window an
# analysis at a given time needs, as offsets from that time: ``(-45h, +3h)`` on one
# rank, narrower under model parallelism where a rank holds only some frames. The two
# windows overlap by 42 hours, so most observations are fetched twice; each analysis
# takes its own DataFrame because the model reads its frame placement from the
# ``request_time`` metadata that :py:func:`earth2studio.data.fetch_dataframe` attaches.

# %%
px_coords = px_model.input_coords()
init_time = np.array([np.datetime64("2024-01-01T00:00")])
lead_times = px_coords["lead_time"]

conv_schema, sat_schema = da_model.input_coords()
analyses = []
for lead in lead_times:
    analysis_time = init_time + lead
    tolerance = da_model.input_time_tolerance(analysis_time[0])
    fetch_start = time.perf_counter()
    conv_df = fetch_dataframe(
        UFSObsConv(time_tolerance=tolerance),
        time=analysis_time,
        variable=np.array(conv_schema["variable"]),
        fields=np.array(list(conv_schema.keys())),
    )
    sat_df = fetch_dataframe(
        UFSObsSat(time_tolerance=tolerance),
        time=analysis_time,
        variable=np.array(sat_schema["variable"]),
        fields=np.array(list(sat_schema.keys())),
    )
    fetch_seconds = time.perf_counter() - fetch_start
    da_start = time.perf_counter()
    analyses.append(da_model(conv_obs=conv_df, sat_obs=sat_df))
    logger.info(
        f"{str(analysis_time[0])[:16]} UTC: {len(conv_df)} conv, {len(sat_df)} sat obs "
        f"fetched in {fetch_seconds:.1f} s, assimilated in "
        f"{time.perf_counter() - da_start:.1f} s"
    )

# %%
# Analyses to Initial Condition
# -----------------------------
# The two analyses stack on a lead-time axis, in the order Aurora asked for. Their
# variables are selected in Aurora's order; the two the analysis does not carry would
# raise here rather than at the forward pass.
#
# Latitude is the other difference: Aurora drops the south pole row, taking 720
# latitudes where the analysis has 721. Those 720 values are exactly the analysis's
# first 720, so :py:func:`earth2studio.utils.coords.map_coords` slices the row off
# without interpolating anything.

# %%
missing = set(px_coords["variable"]) - set(analyses[0].coords["variable"].values)
if missing:
    raise ValueError(f"analysis does not carry {sorted(missing)}")

frames = [
    prep_data_array(analysis.sel(variable=px_coords["variable"]), device=device)[0]
    for analysis in analyses
]
x = torch.stack(frames, dim=1)
coords = OrderedDict(
    time=init_time,
    lead_time=lead_times,
    variable=px_coords["variable"],
    lat=analyses[-1].coords["lat"].values,
    lon=analyses[-1].coords["lon"].values,
)
x, coords = map_coords(x, coords, px_coords)
logger.success(f"Initial condition {tuple(x.shape)}, {len(coords['lat'])} latitudes")

# %%
# Run the Forecast
# ----------------
# Aurora advances six hours per call and keeps the previous state internally, so the
# iterator is what manages the two-state window from here on.

# %%
nsteps = 4
forecast = {}
forecast_start = time.perf_counter()
for step, (y, y_coords) in enumerate(px_model.create_iterator(x, coords)):
    lead = y_coords["lead_time"][-1]
    logger.info(f"Step {step}, lead time {lead.astype('timedelta64[h]')}")
    forecast[lead] = y
    if step == nsteps:
        break
if device.type == "cuda":
    torch.cuda.synchronize()
forecast_seconds = time.perf_counter() - forecast_start
logger.success(f"{nsteps} six-hour steps in {forecast_seconds:.1f} s")

lead_24h = np.timedelta64(24, "h")
lead_hours = lead_24h.astype("timedelta64[h]").astype(int)
valid_time = init_time + lead_24h

# %%
# Verify Against ERA5
# -------------------
# ERA5 at the forecast's valid time, interpolated onto Aurora's own 720-row grid.
#
# Every rank holds the same analyses and forecast, so scoring and plotting are rank 0's
# alone; running them everywhere would race on the ERA5 cache and on the output files.

# %%
if rank != 0:
    sys.exit(0)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

score_vars = ["t2m", "u10m", "t850", "u500", "z500", "q700"]
plot_vars = ["t2m", "z500"]
cmaps = ["Spectral_r", "viridis"]
lat = coords["lat"]
lon = coords["lon"]

era5_valid = NCAR_ERA5()(valid_time, score_vars).interp(
    lat=lat, lon=lon, method="nearest"
)
# The analysis keeps its own 721-row grid, so score it there rather than on Aurora's.
analysis_lat = analyses[-1].coords["lat"].values
era5_init = NCAR_ERA5()(init_time, score_vars).interp(
    lat=analysis_lat, lon=analyses[-1].coords["lon"].values, method="nearest"
)


def to_numpy(arr):
    """CuPy / Numpy helper function"""
    return arr.get() if hasattr(arr, "get") else arr


def rmse(prediction, truth, latitudes):
    """Latitude-weighted root mean square error, cells weighted by cos(lat)."""
    weights = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], truth.shape)
    return float(np.sqrt(np.average((prediction - truth) ** 2, weights=weights)))


variables = list(coords["variable"])
logger.info(f"{'variable':>9}  {'analysis':>10}  {'+' + str(lead_hours) + ' h':>10}")
for var in score_vars:
    analysis_error = rmse(
        to_numpy(analyses[-1].sel(variable=var).data[0]),
        to_numpy(era5_init.sel(variable=var).data[0]),
        analysis_lat,
    )
    forecast_error = rmse(
        forecast[lead_24h][0, -1, variables.index(var)].cpu().numpy(),
        to_numpy(era5_valid.sel(variable=var).data[0]),
        lat,
    )
    logger.info(f"{var:>9}  {analysis_error:10.4g}  {forecast_error:10.4g}")

plt.close("all")
fig, axes = plt.subplots(
    len(plot_vars), 3, subplot_kw={"projection": ccrs.Robinson()}, figsize=(18, 7)
)
for row, var in enumerate(plot_vars):
    predicted = forecast[lead_24h][0, -1, variables.index(var)].cpu().numpy()
    truth = to_numpy(era5_valid.sel(variable=var).data[0])
    difference = predicted - truth
    scale = np.abs(difference).max()
    panels = [
        (f"Aurora +{lead_hours} h", predicted, cmaps[row], None),
        ("ERA5", truth, cmaps[row], None),
        (
            f"difference, rmse {rmse(predicted, truth, lat):.3g}",
            difference,
            "RdBu_r",
            scale,
        ),
    ]
    for col, (label, field, cmap, limit) in enumerate(panels):
        ax = axes[row, col]
        im = ax.imshow(
            field,
            transform=ccrs.PlateCarree(),
            extent=[lon[0], lon[0] + 360, lat[-1], lat[0]],
            origin="upper",
            cmap=cmap,
            vmin=None if limit is None else -limit,
            vmax=limit,
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(f"{var} {label}", fontsize=12)

fig.suptitle(
    f"Aurora initialized by HealDA-v2 at {str(init_time[0])[:16]} UTC, "
    f"lead +{lead_hours} h, valid {str(valid_time[0])[:16]} UTC",
    fontsize=18,
    y=0.99,
)
plt.tight_layout()
plt.savefig("outputs/25_healda_v2_aurora.jpg", dpi=150)
