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
HealDA-v2 Global Data Assimilation
==================================

Producing a global weather analysis from a 48-hour observation window.

This example runs the HealDA-v2 data assimilation model on conventional and
satellite observations from the NOAA UFS replay archive, and compares the analysis
against ERA5. HealDA-v2 assimilates an eight-frame window at six-hour spacing,
each frame taking its own observations, and returns the analysis at the final
frame's valid time.

In this example you will learn:

- How to load HealDA-v2 from a model package
- Fetching UFS conventional and satellite observations for the model's window
- Choosing the resolution of the lat-lon output
- Comparing the analysis against ERA5
"""

# /// script
# dependencies = [
#   "earth2studio[da-healda-v2] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA-v2 :py:class:`earth2studio.models.da.HealDAv2`.
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
# The package holds the weights and the artifacts they were trained with: the static
# conditioning fields, the observation statistics, and the PCA codecs for the infrared
# sounders. Any fsspec URI works in place of a local directory.

# %%
import os
import sys

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# The model broadcasts the finished analysis to every rank, so under torchrun rank 0
# narrates and the other ranks report only failures.
rank = int(os.environ.get("RANK", 0))
logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    colorize=True,
    level="INFO" if rank == 0 else "WARNING",
)

from earth2studio.data import NCAR_ERA5, UFSObsConv, UFSObsSat, fetch_dataframe
from earth2studio.models.da import HealDAv2

# %%
# HealDA-v2 predicts on a HEALPix level-6 grid. ``lat_lon=True`` regrids the output,
# in one bilinear step, to the equiangular grid given by ``output_resolution``: here
# 0.25 degrees, which matches the ERA5 fields used for verification below.

# %%
package = HealDAv2.load_default_package()
model = HealDAv2.load_model(package, lat_lon=True, output_resolution=(721, 1440))

# %%
# Fetch Observations
# ------------------
# The UFS data sources return pandas DataFrames matching the schemas that
# :py:meth:`HealDAv2.input_coords` describes.
# :py:func:`earth2studio.data.fetch_dataframe` attaches the ``request_time``
# metadata the model reads to place the window.
#
# :py:meth:`HealDAv2.input_time_tolerance` gives the window the model wants. The
# eight frames are 6 hours apart ending at the analysis time, and each takes its own
# 6-hour context, so the observations span 45 hours before the analysis time to 3
# hours after. Under model parallelism each rank asks only for the frames it owns.

# %%
analysis_time = np.array([np.datetime64("2024-01-01T00:00")])
tolerance = model.input_time_tolerance(analysis_time[0])
conv_schema, sat_schema = model.input_coords()

conv_df = fetch_dataframe(
    UFSObsConv(time_tolerance=tolerance),
    time=analysis_time,
    variable=np.array(conv_schema["variable"]),
    fields=np.array(list(conv_schema.keys())),
)
logger.info(f"Fetched {len(conv_df)} conventional observations")

sat_df = fetch_dataframe(
    UFSObsSat(time_tolerance=tolerance),
    time=analysis_time,
    variable=np.array(sat_schema["variable"]),
    fields=np.array(list(sat_schema.keys())),
)
logger.info(f"Fetched {len(sat_df)} satellite observations")

# %%
# Run the Assimilation
# --------------------
# DA models can be called directly for stateless inference, or through
# :py:meth:`~earth2studio.models.da.HealDAv2.create_generator` for stateful
# assimilation over successive windows. Here we call the model directly, once with
# both observation types and once with each alone, to show what each contributes.

# %%
torch.manual_seed(42)
result_both = model(conv_obs=conv_df, sat_obs=sat_df)
logger.info(f"Combined analysis shape: {result_both.shape}")

torch.manual_seed(42)
result_sat = model(sat_obs=sat_df)

torch.manual_seed(42)
result_conv = model(conv_obs=conv_df)

# %%
# Post Processing
# ---------------
# The output is already on the requested lat-lon grid. Compare the three runs for
# 2 m temperature and 500 hPa geopotential.
#
# Every rank holds the same analysis, so verification and plotting are rank 0's alone;
# running them everywhere would race on the ERA5 cache and on the output files.

# %%
if rank != 0:
    sys.exit(0)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")
plot_vars = ["t2m", "z500"]
titles = ["Conv + Sat", "Sat only", "Conv only"]
results = [result_both, result_sat, result_conv]

fig, axes = plt.subplots(
    len(results),
    len(plot_vars),
    subplot_kw={"projection": ccrs.Robinson()},
    figsize=(14, 8),
)
fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.1, right=0.9)

lat = results[0].coords["lat"].values
lon = results[0].coords["lon"].values
cmaps = ["Spectral_r", "viridis"]


def to_numpy(arr):
    """CuPy / Numpy helper function"""
    return arr.get() if hasattr(arr, "get") else arr


for row, (title, da) in enumerate(zip(titles, results)):
    for col, var in enumerate(plot_vars):
        ax = axes[row, col]
        im = ax.imshow(
            to_numpy(da.sel(variable=var).data[0]),
            transform=ccrs.PlateCarree(),
            extent=[lon[0], lon[0] + 360, lat[-1], lat[0]],
            origin="upper",
            cmap=cmaps[col],
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        if row == 0:
            ax.set_title(var, fontsize=14)
        if col == 0:
            ax.text(
                -0.05,
                0.5,
                title,
                fontsize=12,
                va="bottom",
                ha="center",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

fig.suptitle(
    f"HealDA-v2 Analysis {str(analysis_time[0])[:16]} UTC", fontsize=18, y=0.97
)
plt.tight_layout()
plt.savefig("outputs/23_healda_v2_analysis.jpg", dpi=150)

# %%
# HealDA-v2 vs ERA5
# -----------------
# ERA5 at 0.25 degrees from the NCAR archive verifies the analysis. The output uses
# the same variable names, so the comparison needs no renaming, and both are on the
# same grid.

# %%
score_vars = ["t2m", "u10m", "t850", "u500", "z500", "q700"]
era5 = NCAR_ERA5()(analysis_time, score_vars)
era5 = era5.interp(lat=lat, lon=lon, method="nearest")


def rmse(prediction, truth, latitudes):
    """Latitude-weighted root mean square error, cells weighted by cos(lat)."""
    weights = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], truth.shape)
    return float(np.sqrt(np.average((prediction - truth) ** 2, weights=weights)))


logger.info(f"{'variable':>9}  {'analysis rmse':>14}")
for var in score_vars:
    error = rmse(
        to_numpy(result_both.sel(variable=var).data[0]),
        to_numpy(era5.sel(variable=var).data[0]),
        lat,
    )
    logger.info(f"{var:>9}  {error:14.4g}")

plt.close("all")
fig, axes = plt.subplots(
    len(plot_vars),
    3,
    subplot_kw={"projection": ccrs.Robinson()},
    figsize=(18, 7),
)

for row, var in enumerate(plot_vars):
    analysis = to_numpy(result_both.sel(variable=var).data[0])
    truth = to_numpy(era5.sel(variable=var).data[0])
    difference = analysis - truth
    scale = np.abs(difference).max()
    panels = [
        ("HealDA-v2", analysis, cmaps[row], None),
        ("ERA5", truth, cmaps[row], None),
        (
            f"difference, rmse {rmse(analysis, truth, lat):.3g}",
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
    f"HealDA-v2 against ERA5 {str(analysis_time[0])[:16]} UTC", fontsize=18, y=0.99
)
plt.tight_layout()
plt.savefig("outputs/23_healda_v2_vs_era5.jpg", dpi=150)

# %%
# Splitting the Window Across GPUs
# --------------------------------
# One rank holds all eight frames by default. ``model_parallel_size`` splits them
# across ranks, each holding its own frames and its share of the observations, with
# the analysis gathered on every rank:
#
# .. code-block:: bash
#
#     torchrun --nproc_per_node 4 03_healda_v2.py
#
# .. code-block:: python
#
#     model = HealDAv2.load_model(package, model_parallel_size=4)
