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
Forecasting from a HealDA-v2 Analysis
=====================================

Initializing FourCastNet 3 from an analysis assimilated out of raw observations.

An analysis is an initial condition, so a data assimilation model and a forecast
model compose: HealDA-v2 turns an observation window into a global state, and
FourCastNet 3 steps that state forward. This example runs the pair end to end and
verifies the 24-hour forecast against ERA5, with no reanalysis anywhere in the
initialization path.

In this example you will learn:

- How to hand a DA model's analysis to a prognostic model
- Matching the variable set, grid and lead-time axis a prognostic model asks for
- Rolling the forecast forward and verifying it against ERA5
"""

# /// script
# dependencies = [
#   "earth2studio[da-healda-v2,fcn3] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA-v2 :py:class:`earth2studio.models.da.HealDAv2`.
# - Prognostic Model: FourCastNet 3 :py:class:`earth2studio.models.px.FCN3`.
# - Datasource (conv): UFS conventional observations
#   :py:class:`earth2studio.data.UFSObsConv`.
# - Datasource (sat): UFS satellite observations
#   :py:class:`earth2studio.data.UFSObsSat`.
# - Datasource (verification): ERA5 :py:class:`earth2studio.data.NCAR_ERA5`.
#
# Set ``HEALDA_V2_PACKAGE`` to the HealDA-v2 package before running::
#
#     export HEALDA_V2_PACKAGE=/path/to/e2s-healda-v2
#
# Any fsspec URI works in place of a local directory. See
# :ref:`sphx_glr_examples_05_data_assimilation_03_healda_v2.py` for the assimilation on
# its own.
#
# FourCastNet 3 needs torch-harmonics and makani installed from source before its extra
# resolves; the install guide's FourCastNet 3 tab gives the commands.

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

# The DA model broadcasts the finished analysis to every rank, so under torchrun rank 0
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
from earth2studio.models.px import FCN3
from earth2studio.utils.coords import map_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# FCN3 takes 0.25-degree fields on 721 latitudes, so the analysis is produced on that
# grid directly: ``output_resolution`` regrids the model's native HEALPix level-6
# output in one bilinear step.

# %%
da_model = HealDAv2.load_model(
    HealDAv2.load_default_package(), lat_lon=True, output_resolution=(721, 1440)
).to(device)
px_model = FCN3.load_model(FCN3.load_default_package()).to(device)

# %%
# Assimilate the Observations
# ---------------------------
# Eight frames at six-hour spacing ending at the analysis time, each frame taking its
# own observations, so the window reaches 45 hours back.
#
# The window spans nine six-hourly cycles, and each cycle has one GSI diagnostic file
# per observation type: 45 conventional and 216 satellite files for this analysis. That
# fetch dominates the wall clock on a cold cache;
# :py:class:`earth2studio.data.UFSObsConv` and :py:class:`earth2studio.data.UFSObsSat`
# cache the files, so a repeat run is far quicker.

# %%
analysis_time = np.array([np.datetime64("2024-01-01T00:00")])
tolerance = da_model.input_time_tolerance(analysis_time[0])
conv_schema, sat_schema = da_model.input_coords()

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
logger.info(
    f"Fetched {len(conv_df)} conventional and {len(sat_df)} satellite obs "
    f"in {fetch_seconds:.1f} s"
)

da_start = time.perf_counter()
analysis = da_model(conv_obs=conv_df, sat_obs=sat_df)
da_seconds = time.perf_counter() - da_start
logger.success(
    f"Analysis {analysis.shape} at {str(analysis_time[0])[:16]} UTC "
    f"in {da_seconds:.1f} s"
)

# %%
# Analysis to Initial Condition
# -----------------------------
# Three things have to agree before a prognostic model will accept a field: the
# variable set, the grid, and the dimensions. The analysis carries 74 channels while
# FCN3 asks for 72 of them in its own order, which ``sel`` handles; the grids already
# match; and the analysis has no lead-time axis, since an analysis is valid at one
# instant, so one is added holding the zero hour.
#
# :py:func:`earth2studio.utils.coords.map_coords` then checks what remains, and is
# where a grid or ordering mistake surfaces rather than silently producing a bad
# forecast.

# %%
px_coords = px_model.input_coords()
missing = set(px_coords["variable"]) - set(analysis.coords["variable"].values)
if missing:
    raise ValueError(f"analysis does not carry {sorted(missing)}")

x, coords = prep_data_array(analysis.sel(variable=px_coords["variable"]), device=device)
x = x.unsqueeze(1)
coords = OrderedDict(
    time=coords["time"],
    lead_time=np.array([np.timedelta64(0, "h")]),
    variable=coords["variable"],
    lat=coords["lat"],
    lon=coords["lon"],
)
x, coords = map_coords(x, coords, px_coords)

# %%
# Run the Forecast
# ----------------
# ``create_iterator`` yields the initial condition first, then one state per
# six-hour step, and manages whatever internal state the model keeps.

# %%
nsteps = 4
forecast = {}
forecast_start = time.perf_counter()
for step, (y, y_coords) in enumerate(px_model.create_iterator(x, coords)):
    lead = y_coords["lead_time"][0]
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
valid_time = analysis_time + lead_24h

# %%
# Verify Against ERA5
# -------------------
# ERA5 at the forecast's valid time, on the same grid and the same variable names.
#
# Every rank holds the same analysis and forecast, so scoring and plotting are rank 0's
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

era5_analysis = NCAR_ERA5()(analysis_time, score_vars).interp(
    lat=lat, lon=lon, method="nearest"
)
era5_valid = NCAR_ERA5()(valid_time, score_vars).interp(
    lat=lat, lon=lon, method="nearest"
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
        to_numpy(analysis.sel(variable=var).data[0]),
        to_numpy(era5_analysis.sel(variable=var).data[0]),
        lat,
    )
    forecast_error = rmse(
        forecast[lead_24h][0, 0, variables.index(var)].cpu().numpy(),
        to_numpy(era5_valid.sel(variable=var).data[0]),
        lat,
    )
    logger.info(f"{var:>9}  {analysis_error:10.4g}  {forecast_error:10.4g}")

plt.close("all")
fig, axes = plt.subplots(
    len(plot_vars), 3, subplot_kw={"projection": ccrs.Robinson()}, figsize=(18, 7)
)
for row, var in enumerate(plot_vars):
    predicted = forecast[lead_24h][0, 0, variables.index(var)].cpu().numpy()
    truth = to_numpy(era5_valid.sel(variable=var).data[0])
    difference = predicted - truth
    scale = np.abs(difference).max()
    panels = [
        (f"FCN3 +{lead_hours} h", predicted, cmaps[row], None),
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
    f"FCN3 initialized by HealDA-v2 at {str(analysis_time[0])[:16]} UTC, "
    f"lead +{lead_hours} h, valid {str(valid_time[0])[:16]} UTC",
    fontsize=18,
    y=0.99,
)
plt.tight_layout()
plt.savefig("outputs/24_healda_v2_fcn3.jpg", dpi=150)

# %%
# Other Prognostic Models
# -----------------------
# The handoff above is the general shape; what changes per model is the coordinate
# contract it declares. :py:class:`earth2studio.models.px.Aurora`, for instance, asks
# for two lead times, ``-6h`` and ``0h``, and 720 latitudes rather than 721, dropping
# the south pole row. That means two analyses six hours apart, concatenated on the
# lead-time axis; ``map_coords`` handles the latitudes on its own, since Aurora's rows
# are exactly the analysis rows with the last one removed.
#
# For a longer rollout with IO, wrapping the analysis in a
# :py:class:`earth2studio.data.DataSource` and calling
# :py:func:`earth2studio.run.deterministic` reuses the built-in workflow rather than
# the loop above.
