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

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from earth2studio.data.utils_bufr import (
    RAW_BUFR_SCHEMA,
    validate_bufr_batches,
)

GPSRO_SAID = 1007
GPSRO_PTID = 1050
GPSRO_QFRO = 33039
GPSRO_ELRC = 10035
GPSRO_GEODU = 10036
GPSRO_LAT = 5001
GPSRO_LON = 6001
GPSRO_YEAR = 4001
GPSRO_MONTH = 4002
GPSRO_DAY = 4003
GPSRO_HOUR = 4004
GPSRO_MIN = 4005
GPSRO_SEC = 4006
GPSRO_MEFR = 2121
GPSRO_IMPP = 7040
GPSRO_BNDA = 15037
GPSRO_HEIT = 7007
GPSRO_ARFR = 15036
GPSRO_GPHTST = 7009

_GPSRO_DESCRIPTOR_BY_FIELD_NAME = {
    "SAID": GPSRO_SAID,
    "PTID": GPSRO_PTID,
    "QFRO": GPSRO_QFRO,
    "ELRC": GPSRO_ELRC,
    "GEODU": GPSRO_GEODU,
    "CLATH": GPSRO_LAT,
    "CLONH": GPSRO_LON,
    "YEAR": GPSRO_YEAR,
    "MNTH": GPSRO_MONTH,
    "DAYS": GPSRO_DAY,
    "HOUR": GPSRO_HOUR,
    "MINU": GPSRO_MIN,
    "SECO": GPSRO_SEC,
    "MEFR": GPSRO_MEFR,
    "IMPP": GPSRO_IMPP,
    "BNDA": GPSRO_BNDA,
    "HEIT": GPSRO_HEIT,
    "ARFR": GPSRO_ARFR,
    "GPHTST": GPSRO_GPHTST,
}

_GPSRO_L1_FREQUENCY_HZ = 1_500_000_000
_GPSRO_L2_FREQUENCY_HZ = 1_200_000_000
_GPSRO_COMBINED_FREQUENCY_HZ = 0
_GPSRO_PRODUCT_BY_FREQUENCY = {
    _GPSRO_L1_FREQUENCY_HZ: "l1",
    _GPSRO_L2_FREQUENCY_HZ: "l2",
    _GPSRO_COMBINED_FREQUENCY_HZ: "combined",
}

GPSRO_BENDING_LEVEL_TYPE = pa.struct(
    [
        pa.field("source_level_index", pa.uint32(), nullable=False),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("impact_parameter", pa.float64()),
        pa.field("combined_bending_angle", pa.float64()),
        pa.field("combined_bending_angle_uncertainty", pa.float64()),
        pa.field("l1_bending_angle", pa.float64()),
        pa.field("l1_bending_angle_uncertainty", pa.float64()),
        pa.field("l2_bending_angle", pa.float64()),
        pa.field("l2_bending_angle_uncertainty", pa.float64()),
    ]
)

GPSRO_REFRACTIVITY_LEVEL_TYPE = pa.struct(
    [
        pa.field("height", pa.float64()),
        pa.field("atmospheric_refractivity", pa.float64()),
        pa.field("atmospheric_refractivity_uncertainty", pa.float64()),
    ]
)

NCEP_GPSRO_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("profile_id", pa.string(), nullable=False),
        pa.field("source_cycle", pa.timestamp("ns"), nullable=False),
        pa.field("source_profile_index", pa.uint32(), nullable=False),
        pa.field("observation_time", pa.timestamp("ns"), nullable=False),
        pa.field("receiver_said", pa.uint16()),
        pa.field("transmitter_ptid", pa.uint16()),
        pa.field("qfro", pa.uint16()),
        pa.field("earth_radius_of_curvature", pa.float64()),
        pa.field("geoid_undulation", pa.float64()),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field(
            "bending_levels",
            pa.list_(GPSRO_BENDING_LEVEL_TYPE),
            nullable=False,
        ),
        pa.field(
            "refractivity_levels",
            pa.list_(GPSRO_REFRACTIVITY_LEVEL_TYPE),
            nullable=False,
        ),
    ]
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _uint16(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0 or number > np.iinfo(np.uint16).max:
        return None
    integer = int(number)
    return integer if number == integer else None


def _profile_header(descriptors: list[int], values: list[Any]) -> dict[str, Any] | None:
    fields: dict[int, Any] = {}
    for descriptor_id, value in zip(descriptors, values, strict=True):
        if descriptor_id in (GPSRO_LAT, GPSRO_LON) and descriptor_id in fields:
            continue
        fields[descriptor_id] = value
        if descriptor_id == GPSRO_IMPP:
            break

    year = _uint16(fields.get(GPSRO_YEAR))
    month = _uint16(fields.get(GPSRO_MONTH))
    day = _uint16(fields.get(GPSRO_DAY))
    if year is None or month is None or day is None:
        return None
    second = _finite_float(fields.get(GPSRO_SEC)) or 0.0
    try:
        observation_time = datetime(
            year,
            month,
            day,
            _uint16(fields.get(GPSRO_HOUR)) or 0,
            _uint16(fields.get(GPSRO_MIN)) or 0,
        ) + timedelta(seconds=second)
    except (TypeError, ValueError, OverflowError):
        return None
    return {
        "observation_time": observation_time,
        "receiver_said": _uint16(fields.get(GPSRO_SAID)),
        "transmitter_ptid": _uint16(fields.get(GPSRO_PTID)),
        "qfro": _uint16(fields.get(GPSRO_QFRO)),
        "earth_radius_of_curvature": _finite_float(fields.get(GPSRO_ELRC)),
        "geoid_undulation": _finite_float(fields.get(GPSRO_GEODU)),
        "latitude": _finite_float(fields.get(GPSRO_LAT)),
        "longitude": _finite_float(fields.get(GPSRO_LON)),
    }


def _refractivity_levels(
    descriptors: list[int], values: list[Any]
) -> list[dict[str, float | None]]:
    levels: list[dict[str, float | None]] = []
    current: dict[str, float | None] | None = None
    arfr_slot = 0
    for descriptor_id, value in zip(descriptors, values, strict=True):
        if descriptor_id == GPSRO_HEIT:
            if current is not None:
                levels.append(current)
            current = {
                "height": _finite_float(value),
                "atmospheric_refractivity": None,
                "atmospheric_refractivity_uncertainty": None,
            }
            arfr_slot = 0
        elif current is not None and descriptor_id == GPSRO_ARFR:
            arfr_slot += 1
            field = (
                "atmospheric_refractivity"
                if arfr_slot == 1
                else "atmospheric_refractivity_uncertainty"
            )
            if arfr_slot <= 2:
                current[field] = _finite_float(value)
        elif current is not None and descriptor_id == GPSRO_GPHTST:
            break
    if current is not None:
        levels.append(current)
    return levels


def _usable_bending_level(level: dict[str, Any], radius: float | None) -> bool:
    latitude = level["latitude"]
    longitude = level["longitude"]
    impact = level["impact_parameter"]
    bending = level["combined_bending_angle"]
    return bool(
        latitude is not None
        and longitude is not None
        and impact is not None
        and radius is not None
        and bending is not None
        and abs(latitude) <= 90.0
        and -180.0 <= longitude <= 360.0
        and 0.0 < bending < 1.0e9
        and impact < 1.0e9
        and impact >= radius
    )


def _bending_levels(
    descriptors: list[int],
    values: list[Any],
    radius: float | None,
) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    frequency: float | None = None
    bnda_slot = 0
    latitude_slots = 0
    source_level_index = -1

    def finish_level() -> None:
        if current is None or not _usable_bending_level(current, radius):
            return
        levels.append(current)

    for descriptor_id, value in zip(descriptors, values, strict=True):
        if descriptor_id == GPSRO_HEIT:
            break
        if descriptor_id == GPSRO_LAT:
            latitude_slots += 1
            if latitude_slots == 1:
                continue
            finish_level()
            source_level_index += 1
            current = {
                "source_level_index": source_level_index,
                "latitude": _finite_float(value),
                "longitude": None,
                "impact_parameter": None,
                "combined_bending_angle": None,
                "combined_bending_angle_uncertainty": None,
                "l1_bending_angle": None,
                "l1_bending_angle_uncertainty": None,
                "l2_bending_angle": None,
                "l2_bending_angle_uncertainty": None,
            }
            frequency = None
            bnda_slot = 0
            continue
        if current is None:
            continue
        if descriptor_id == GPSRO_LON:
            current["longitude"] = _finite_float(value)
        elif descriptor_id == GPSRO_MEFR:
            frequency = _finite_float(value)
            bnda_slot = 0
        elif descriptor_id == GPSRO_IMPP:
            if frequency is not None and round(frequency) == 0:
                current["impact_parameter"] = _finite_float(value)
        elif descriptor_id == GPSRO_BNDA:
            bnda_slot += 1
            if frequency is None:
                continue
            product = _GPSRO_PRODUCT_BY_FREQUENCY.get(round(frequency))
            if product is None or bnda_slot > 2:
                continue
            suffix = "" if bnda_slot == 1 else "_uncertainty"
            current[f"{product}_bending_angle{suffix}"] = _finite_float(value)
    finish_level()
    return levels


def _extract_profile(
    descriptors: list[int], values: list[Any]
) -> dict[str, Any] | None:
    header = _profile_header(descriptors, values)
    if header is None:
        return None
    refractivity = _refractivity_levels(descriptors, values)
    bending = _bending_levels(
        descriptors,
        values,
        header["earth_radius_of_curvature"],
    )
    if not bending:
        return None
    return {
        **header,
        "bending_levels": bending,
        "refractivity_levels": refractivity,
    }


def _raw_element_value(element: dict[str, Any]) -> Any:
    if element["numeric_value"] is not None:
        return element["numeric_value"]
    if element["string_value"] is not None:
        return element["string_value"]
    return element["bytes_value"]


def _raw_element_descriptor(element: dict[str, Any]) -> int:
    descriptor_id = element["descriptor_id"]
    if descriptor_id is not None:
        return descriptor_id
    return _GPSRO_DESCRIPTOR_BY_FIELD_NAME.get(element["field_name"], -1)


def _field_indices(fields: Any, name: str) -> list[int]:
    return [index for index, field in enumerate(fields) if field.name == name]


def _first_field_index(fields: Any, name: str) -> int | None:
    indices = _field_indices(fields, name)
    return indices[0] if indices else None


def _list_struct_type(field: pa.Field) -> pa.StructType | None:
    if not (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)):
        return None
    value_type = field.type.value_type
    return value_type if pa.types.is_struct(value_type) else None


def _nested_profile_columns(schema: pa.Schema) -> tuple[int, int] | None:
    bending_index = None
    refractivity_index = None
    for index, field in enumerate(schema):
        value_type = _list_struct_type(field)
        if value_type is None:
            continue
        child_names = {child.name for child in value_type}
        if {"CLATH", "CLONH"}.issubset(child_names):
            bending_index = index
        elif {"HEIT", "ARFR"}.issubset(child_names):
            refractivity_index = index
    if bending_index is None or refractivity_index is None:
        return None
    return bending_index, refractivity_index


def _scalar_value(array: pa.Array, index: int) -> Any:
    scalar = array[index]
    return scalar.as_py() if scalar.is_valid else None


def _native_nested_profile(
    batch: pa.RecordBatch,
    row_index: int,
    columns: tuple[int, int],
) -> dict[str, Any] | None:
    descriptors: list[int] = []
    values: list[Any] = []
    header_fields = [
        ("SAID", GPSRO_SAID),
        ("PTID", GPSRO_PTID),
        ("QFRO", GPSRO_QFRO),
        ("YEAR", GPSRO_YEAR),
        ("MNTH", GPSRO_MONTH),
        ("DAYS", GPSRO_DAY),
        ("HOUR", GPSRO_HOUR),
        ("MINU", GPSRO_MIN),
        ("SECO", GPSRO_SEC),
        ("CLATH", GPSRO_LAT),
        ("CLONH", GPSRO_LON),
        ("ELRC", GPSRO_ELRC),
        ("GEODU", GPSRO_GEODU),
    ]
    for field_name, descriptor_id in header_fields:
        field_index = _first_field_index(batch.schema, field_name)
        if field_index is not None:
            descriptors.append(descriptor_id)
            values.append(_scalar_value(batch.column(field_index), row_index))

    bending_scalar = batch.column(columns[0])[row_index]
    refractivity_scalar = batch.column(columns[1])[row_index]
    if not bending_scalar.is_valid or not refractivity_scalar.is_valid:
        return None

    bending = bending_scalar.values
    bending_type = bending.type
    latitude_index = _first_field_index(bending_type, "CLATH")
    longitude_index = _first_field_index(bending_type, "CLONH")
    frequency_list_index = next(
        (
            index
            for index, field in enumerate(bending_type)
            if _list_struct_type(field) is not None
        ),
        None,
    )
    if (
        latitude_index is None
        or longitude_index is None
        or frequency_list_index is None
    ):
        return None

    latitudes = bending.field(latitude_index).to_pylist()
    longitudes = bending.field(longitude_index).to_pylist()
    frequency_lists = bending.field(frequency_list_index)
    frequency_type = frequency_lists.type.value_type
    frequency_index = _first_field_index(frequency_type, "MEFR")
    impact_index = _first_field_index(frequency_type, "IMPP")
    angle_indices = _field_indices(frequency_type, "BNDA")
    if frequency_index is None or impact_index is None or not angle_indices:
        return None

    offsets = frequency_lists.offsets.to_pylist()
    first_offset = offsets[0]
    offsets = [offset - first_offset for offset in offsets]
    frequencies = pc.list_flatten(frequency_lists)
    frequency_values = frequencies.field(frequency_index).to_pylist()
    impact_values = frequencies.field(impact_index).to_pylist()
    observed_angles = frequencies.field(angle_indices[0]).to_pylist()
    uncertainty_angles = (
        frequencies.field(angle_indices[1]).to_pylist()
        if len(angle_indices) > 1
        else [None] * len(frequencies)
    )
    for level_index, (latitude, longitude) in enumerate(
        zip(latitudes, longitudes, strict=True)
    ):
        descriptors.extend((GPSRO_LAT, GPSRO_LON))
        values.extend((latitude, longitude))
        for item_index in range(offsets[level_index], offsets[level_index + 1]):
            descriptors.extend((GPSRO_MEFR, GPSRO_IMPP, GPSRO_BNDA, GPSRO_BNDA))
            values.extend(
                (
                    frequency_values[item_index],
                    impact_values[item_index],
                    observed_angles[item_index],
                    uncertainty_angles[item_index],
                )
            )

    refractivity = refractivity_scalar.values
    refractivity_type = refractivity.type
    height_index = _first_field_index(refractivity_type, "HEIT")
    refractivity_indices = _field_indices(refractivity_type, "ARFR")
    if height_index is None or not refractivity_indices:
        return None
    heights = refractivity.field(height_index).to_pylist()
    observed_refractivity = refractivity.field(refractivity_indices[0]).to_pylist()
    uncertainty_refractivity = (
        refractivity.field(refractivity_indices[1]).to_pylist()
        if len(refractivity_indices) > 1
        else [None] * len(refractivity)
    )
    for height, observation, uncertainty in zip(
        heights,
        observed_refractivity,
        uncertainty_refractivity,
        strict=True,
    ):
        descriptors.extend((GPSRO_HEIT, GPSRO_ARFR, GPSRO_ARFR))
        values.extend((height, observation, uncertainty))

    return _extract_profile(descriptors, values)


def transform_gpsro_profiles(
    batches: list[pa.RecordBatch], source_cycle: datetime
) -> pa.Table:
    """Transform raw decoded BUFR subsets into GPSRO occultation profiles.

    The transformation does not apply GSI policy filters such as QFRO,
    receiver enablement, or an observation-time window. It removes only bending
    levels that cannot represent a physical observation because their combined
    angle, impact geometry, or location is missing or invalid.

    Parameters
    ----------
    batches : list[pa.RecordBatch]
        Raw decoder output preserving BUFR field names and replication structure.
    source_cycle : datetime
        Cycle represented by the source file.

    Returns
    -------
    pa.Table
        One row per usable occultation profile with
        :data:`NCEP_GPSRO_PROFILE_SCHEMA`.
    """
    validate_bufr_batches(batches)
    rows: list[dict[str, Any]] = []
    source_profile_index = 0
    for batch in batches:
        if batch.schema == RAW_BUFR_SCHEMA:
            profiles = []
            for raw_subset in batch.to_pylist():
                elements = raw_subset["elements"]
                descriptors = [_raw_element_descriptor(element) for element in elements]
                values = [_raw_element_value(element) for element in elements]
                profiles.append(_extract_profile(descriptors, values))
        else:
            columns = _nested_profile_columns(batch.schema)
            if columns is None:
                raise TypeError(
                    "BUFR decoder output does not contain a supported raw GPSRO "
                    "replication structure"
                )
            profiles = [
                _native_nested_profile(batch, row_index, columns)
                for row_index in range(batch.num_rows)
            ]

        for profile in profiles:
            if profile is not None:
                rows.append(
                    {
                        "profile_id": f"{source_cycle:%Y%m%d%H}:{source_profile_index}",
                        "source_cycle": source_cycle,
                        "source_profile_index": source_profile_index,
                        **profile,
                    }
                )
            source_profile_index += 1
    return pa.Table.from_pylist(rows, schema=NCEP_GPSRO_PROFILE_SCHEMA)
