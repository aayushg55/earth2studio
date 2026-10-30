# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

GRAVITY = 9.80665
DRY_AIR_GAS_CONSTANT = 287.053
STANDARD_LAPSE_RATE = 0.0065  # K/m


def specific_humidity_from_dewpoint(dewpoint_k, pressure_hpa):
    vapour_hpa = 6.112 * np.exp(17.67 * (dewpoint_k - 273.15) / (dewpoint_k - 29.65))
    mixing_ratio = 0.622 * vapour_hpa / (pressure_hpa - vapour_hpa)
    return mixing_ratio / (1 + mixing_ratio)


def temperature_at_height(temperature_k, height_change_m):
    return temperature_k - STANDARD_LAPSE_RATE * height_change_m


def pressure_at_height(pressure, temperature_k, height_change_m, specific_humidity=0.0):
    """Hydrostatic pressure ``height_change_m`` above ``pressure``, through a layer at
    ``temperature_k`` minus the standard lapse rate over half the height change, as
    virtual temperature for air of ``specific_humidity`` (kg/kg)."""
    layer_temperature = temperature_k - STANDARD_LAPSE_RATE * height_change_m / 2
    virtual_temperature = layer_temperature * (1 + 0.608 * specific_humidity)
    exponent = -GRAVITY * height_change_m / (DRY_AIR_GAS_CONSTANT * virtual_temperature)
    return pressure * np.exp(exponent)
