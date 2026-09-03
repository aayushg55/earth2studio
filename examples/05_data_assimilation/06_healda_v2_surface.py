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
HealDA-v2 Surface Analysis
==========================

Assimilating the three surface fields the 77-channel package adds.

The 77-channel HealDA-v2 package predicts 2 m dewpoint (``d2m``), skin temperature
(``skt``) and surface pressure (``sp``) alongside the 74 channels of the base
package. This example produces an analysis from the same UFS observations as
``03_healda_v2.py`` and verifies those three fields against ERA5.

In this example you will learn:

- How to select the 77-channel package with ``extra_surface``
- Which extra assets that package carries and why
- Verifying the surface fields against ERA5
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
# This example requires the same components as ``03_healda_v2.py``, pointed at a
# 77-channel package::
#
#     export HEALDA_V2_PACKAGE=/path/to/e2s-healda-v2-77ch
#
# ``extra_surface=True`` names the three channels, and applies the hypsometric
# anchor that ``sp`` was trained under: the head predicts a correction to
# ``msl exp(-g z / (Rd tas))`` rather than surface pressure itself, so inference
# adds the same estimate back. The ``z`` there is ERA5 orography in metres, which
# the package carries as ``static/orography_era5_hpx6_padxy.npy``. It is a
# different field from the orography in the model's own conditioning, which is
# UFS's and z-scored; the two disagree by 67 m rms.

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
# The analysis is kept on its native HEALPix level-6 grid, which the verification
# below needs.

# %%
package = HealDAv2.load_default_package()
model = HealDAv2.load_model(package, extra_surface=True)

# %%
# Fetch Observations
# ------------------
# Unchanged from the base example: the extra channels are outputs, and no
# observation type is added to predict them.

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
sat_df = fetch_dataframe(
    UFSObsSat(time_tolerance=tolerance),
    time=analysis_time,
    variable=np.array(sat_schema["variable"]),
    fields=np.array(list(sat_schema.keys())),
)
logger.info(f"Fetched {len(conv_df)} conventional and {len(sat_df)} satellite rows")

# %%
torch.manual_seed(42)
analysis = model(conv_obs=conv_df, sat_obs=sat_df)
logger.info(f"Analysis shape: {analysis.shape}")

# %%
# Verify Against ERA5
# -------------------
# The three names are ERA5's own, so the comparison needs no renaming. The four
# upper-air fields are shared with the 74-channel package, so the same numbers can
# be read off either one: the finetune trains the last trunk block, which feeds the
# original head too, and those channels are free to move.
#
# ERA5 is brought down to the analysis grid rather than the analysis up to ERA5's
# 0.25 degrees. A level-6 cell spans about 16 ERA5 cells, so the reduction is
# bilinear onto level 8 followed by a block mean over the level-8 children of each
# level-6 cell; HEALPix cells are equal area, which makes that mean an area average,
# and makes the error below an unweighted one. Scoring the other way charges the
# model for detail a 100 km field cannot carry, which for ``sp`` over orography is
# an order of magnitude: 886 Pa against the 78 Pa below.

# %%
if rank != 0:
    sys.exit(0)

import cartopy.crs as ccrs
import earth2grid
import matplotlib.pyplot as plt
from earth2grid import healpix

surface_vars = ["d2m", "skt", "sp"]
shared_vars = ["t850", "u500", "z500", "q700"]

ERA5_NLAT, ERA5_NLON = 721, 1440
FINE_LEVEL = 8
level = int(np.round(np.emath.logn(4, analysis.sizes["npix"] / 12)))

era5 = NCAR_ERA5()(analysis_time, surface_vars + shared_vars)

era5_to_fine = earth2grid.get_regridder(
    earth2grid.latlon.equiangular_lat_lon_grid(ERA5_NLAT, ERA5_NLON),
    healpix.Grid(FINE_LEVEL, pixel_order=healpix.PixelOrder.NEST),
)
# Half a degree for the maps, plenty for a 100 km analysis.
plot_lat = np.linspace(90, -90, 361)
plot_lon = np.linspace(0, 360, 720, endpoint=False)
hpx_to_plot = earth2grid.get_regridder(
    healpix.Grid(level, pixel_order=healpix.HEALPIX_PAD_XY),
    earth2grid.latlon.equiangular_lat_lon_grid(361, 720),
)


def to_numpy(arr):
    """CuPy / Numpy helper function"""
    return arr.get() if hasattr(arr, "get") else arr


def era5_on_hpx(var):
    """ERA5 area-averaged onto the analysis grid, in HEALPIX_PAD_XY order."""
    fine = era5_to_fine(
        torch.as_tensor(to_numpy(era5.sel(variable=var).data[0])).double()
    )
    coarse = fine.reshape(-1, 4 ** (FINE_LEVEL - level)).mean(-1)
    return healpix.reorder(coarse, healpix.PixelOrder.NEST, healpix.HEALPIX_PAD_XY)


def analysis_on_hpx(var):
    """One analysis channel as a torch tensor over npix."""
    return torch.as_tensor(to_numpy(analysis.sel(variable=var).data[0])).double()


def rmse(prediction, truth):
    """Root mean square error over equal-area HEALPix cells."""
    return float(torch.sqrt(torch.mean((prediction - truth) ** 2)))


def to_map(field):
    """A HEALPix field regridded to half-degree lat-lon for plotting."""
    return hpx_to_plot(field).numpy()


# %%
plt.close("all")
fig, axes = plt.subplots(
    len(surface_vars),
    3,
    subplot_kw={"projection": ccrs.Robinson()},
    figsize=(18, 10),
)
units = {
    "d2m": "K",
    "skt": "K",
    "sp": "Pa",
    "t850": "K",
    "u500": "m/s",
    "z500": "m^2/s^2",
    "q700": "kg/kg",
}
cmaps = {"d2m": "Spectral_r", "skt": "Spectral_r", "sp": "viridis"}

logger.info(f"{'variable':>9}  {'analysis rmse':>14}")
for row, var in enumerate(surface_vars):
    predicted = analysis_on_hpx(var)
    truth = era5_on_hpx(var)
    error = rmse(predicted, truth)
    logger.info(f"{var:>9}  {error:10.4g} {units[var]}")

    predicted_map, truth_map = to_map(predicted), to_map(truth)
    difference = predicted_map - truth_map
    scale = np.abs(difference).max()
    panels = [
        ("HealDA-v2", predicted_map, cmaps[var], None),
        ("ERA5", truth_map, cmaps[var], None),
        (f"difference, rmse {error:.3g} {units[var]}", difference, "RdBu_r", scale),
    ]
    for col, (label, field, cmap, limit) in enumerate(panels):
        ax = axes[row, col]
        im = ax.imshow(
            field,
            transform=ccrs.PlateCarree(),
            extent=[plot_lon[0], plot_lon[0] + 360, plot_lat[-1], plot_lat[0]],
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
    f"HealDA-v2 surface analysis against ERA5 {str(analysis_time[0])[:16]} UTC",
    fontsize=18,
    y=0.99,
)
plt.tight_layout()
plt.savefig("outputs/26_healda_v2_surface.jpg", dpi=150)

# %%
# The shared upper-air fields, for comparison against the same numbers from the
# 74-channel package.

# %%
for var in shared_vars:
    error = rmse(analysis_on_hpx(var), era5_on_hpx(var))
    logger.info(f"{var:>9}  {error:10.4g} {units[var]}")
