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

"""Shared PrepBUFR parsing utilities for NCEP-format BUFR files.

This module provides common helpers for decoding NCEP PrepBUFR files
(DX table extraction, message splitting, and pybufrkit table
registration) used by :mod:`earth2studio.data.gdas` and
:mod:`earth2studio.data.nnja`.
"""

from __future__ import annotations

import contextlib
import numbers
import os
import struct
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa
from loguru import logger

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

# Shared optional-dependency key for pybufrkit. NCEP conventional sources decode
# through this module, so their public classes check this key.
BUFR_DEPENDENCY_KEY = "bufr"

RAW_BUFR_ELEMENT_TYPE = pa.struct(
    [
        pa.field("descriptor_id", pa.int32()),
        pa.field("field_name", pa.string()),
        pa.field("numeric_value", pa.float64()),
        pa.field("string_value", pa.string()),
        pa.field("bytes_value", pa.binary()),
    ]
)

RAW_BUFR_SCHEMA = pa.schema(
    [
        pa.field("message_index", pa.uint32(), nullable=False),
        pa.field("subset_index", pa.uint32(), nullable=False),
        pa.field("data_category", pa.uint8(), nullable=False),
        pa.field(
            "elements",
            pa.list_(RAW_BUFR_ELEMENT_TYPE),
            nullable=False,
        ),
    ]
)


@runtime_checkable
class BUFRDecoder(Protocol):
    """Decoder interface for raw BUFR subsets.

    Implementations decode BUFR mechanics only and preserve source field names,
    order, and replication structure in Arrow. Product-specific interpretation,
    filtering, and output schemas remain the responsibility of the caller.
    """

    def decode_file(self, path: str | Path) -> list[pa.RecordBatch]:
        """Decode a BUFR file into raw Arrow batches.

        Parameters
        ----------
        path : str | Path
            Local BUFR file.

        Returns
        -------
        list[pa.RecordBatch]
            Raw decoded batches preserving BUFR field and replication structure.
        """
        ...


def validate_bufr_batches(
    batches: list[pa.RecordBatch],
) -> list[pa.RecordBatch]:
    """Validate the container contract for decoded BUFR batches.

    Parameters
    ----------
    batches : list[pa.RecordBatch]
        Decoder output.

    Returns
    -------
    list[pa.RecordBatch]
        The validated batches.

    Raises
    ------
    TypeError
        If a decoder returns a non-batch value.
    """
    for batch in batches:
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(
                "BUFR decoder must return a list of pyarrow.RecordBatch objects"
            )
    return batches


def validate_raw_bufr_batches(
    batches: list[pa.RecordBatch],
) -> list[pa.RecordBatch]:
    """Validate batches against the default raw BUFR interchange schema.

    Parameters
    ----------
    batches : list[pa.RecordBatch]
        Default-decoder output.

    Returns
    -------
    list[pa.RecordBatch]
        The validated batches.

    Raises
    ------
    TypeError
        If a batch has an incompatible schema.
    """
    validate_bufr_batches(batches)
    for batch in batches:
        if batch.schema != RAW_BUFR_SCHEMA:
            raise TypeError(
                f"BUFR decoder returned an incompatible schema: {batch.schema}"
            )
    return batches


try:
    from pybufrkit.decoder import Decoder as BufrDecoder  # type: ignore[import-untyped]
    from pybufrkit.tables import (  # type: ignore[import-untyped]
        TableGroupCacheManager,
    )
except ImportError:
    OptionalDependencyFailure("data", BUFR_DEPENDENCY_KEY)
    BufrDecoder = None  # type: ignore[assignment,misc]
    TableGroupCacheManager = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────
# PrepBUFR descriptor ID constants
# ─────────────────────────────────────────────────────────────────────

# Header field descriptors (per-subset, scalar)
HDR_SID = 1194  # Station ID (CCITT IA5, 64 bits)
HDR_XOB = 6240  # Longitude (degrees east)
HDR_YOB = 5002  # Latitude (degrees north)
HDR_DHR = 4215  # Obs time minus cycle time (hours)
HDR_ELV = 10199  # Station elevation (m)
HDR_TYP = 55007  # Report type code
HDR_T29 = 55008  # Data dump report type code

# Observation field descriptors
OBS_CAT = 8193  # Observation category code
OBS_POB = 7245  # Pressure observation (MB)
OBS_ZOB = 10007  # Height observation (m)
OBS_TOB = 12245  # Temperature observation (DEG C)
OBS_QOB = 13245  # Specific humidity (MG/KG)
OBS_UOB = 11003  # U-wind component (M/S)
OBS_VOB = 11004  # V-wind component (M/S)
OBS_HRDR = 4218  # Profile-level time minus cycle time (hours)
OBS_XDR = 6241  # Profile-level longitude (degrees east)
OBS_YDR = 5241  # Profile-level latitude (degrees north)

# Quality mark descriptors
OBS_PQM = 7246  # Pressure quality mark
OBS_TQM = 12246  # Temperature quality mark
OBS_QQM = 13246  # Moisture quality mark
OBS_WQM = 11240  # Wind quality mark

# Set of all header descriptor IDs
HEADER_DESCR_IDS: set[int] = {
    HDR_SID,
    HDR_XOB,
    HDR_YOB,
    HDR_DHR,
    HDR_ELV,
    HDR_TYP,
    HDR_T29,
}

# Set of core observation-level descriptor IDs (obs + quality marks)
OBSERVATION_DESCR_IDS: set[int] = {
    OBS_POB,
    OBS_PQM,
    OBS_ZOB,
    OBS_TOB,
    OBS_TQM,
    OBS_QOB,
    OBS_QQM,
    OBS_UOB,
    OBS_VOB,
    OBS_WQM,
    OBS_HRDR,
    OBS_XDR,
    OBS_YDR,
}

# Lexicon mnemonic -> descriptor ID for non-wind observation fields
MNEMONIC_TO_DESCR: dict[str, int] = {
    "TOB": OBS_TOB,
    "QOB": OBS_QOB,
    "POB": OBS_POB,
    "ZOB": OBS_ZOB,
    "UOB": OBS_UOB,
    "VOB": OBS_VOB,
}

# PrepBUFR section-1 dataCategory -> NCEP message/subset family.
# Ref: NCEP PREPBUFR Table 1.a / prepobs_prep.bufrtable Table A.
# https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_1.htm
# These are PREPBUFR message families, not inner report TYP values.
PREPBUFR_OBS_TYPES: dict[int, str] = {
    102: "ADPUPA",  # Upper-air: RAOB, PIBAL, RECCO, dropsonde
    103: "AIRCAR",  # Aircraft: MDCRS ACARS
    104: "AIRCFT",  # Aircraft: AIREP, PIREP, AMDAR, TAMDAR
    105: "SATWND",  # Satellite-derived winds
    106: "PROFLR",  # Wind profiler / SODAR reports
    107: "VADWND",  # VAD/NEXRAD winds
    108: "SATEMP",  # POES/TOVS sounding/retrieval/radiance data
    109: "ADPSFC",  # Surface land: SYNOP/METAR
    110: "SFCSHP",  # Surface marine: ship/buoy/C-MAN/tide gauge
    111: "SFCBOG",  # Mean sea-level pressure bogus reports
    112: "SPSSMI",  # DMSP SSM/I retrieval products
    113: "SYNDAT",  # Synthetic bogus data
    114: "ERS1DA",  # ERS scatterometer winds
    115: "GOESND",  # GOES sounding/retrieval/radiance data
    116: "QKSWND",  # QuikSCAT scatterometer winds
    117: "MSONET",  # Mesonet surface reports
    118: "GPSIPW",  # GPS integrated precipitable water / zenith delay
    119: "RASSDA",  # RASS virtual temperature
    120: "WDSATR",  # WindSat scatterometer winds
    121: "ASCATW",  # ASCAT scatterometer winds
}

# Per-variable quality mark mapping: observation descriptor -> QM descriptor
OBS_QUALITY_MAP: dict[int, int] = {
    OBS_POB: OBS_PQM,
    OBS_TOB: OBS_TQM,
    OBS_QOB: OBS_QQM,
    OBS_UOB: OBS_WQM,
    OBS_VOB: OBS_WQM,
}


# ─────────────────────────────────────────────────────────────────────
# Silence noisy C-level stderr from pybufrkit
# ─────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def silence_bufr_noise() -> Iterator[None]:
    """Suppress chatty C-library stderr from pybufrkit.

    pybufrkit writes informational messages straight to file
    descriptor 2 (e.g. ``Cannot find sub-centre 3 nor valid default``)
    when the file uses NCEP-local descriptors. We rely on the DX
    tables embedded in each file to decode those correctly, so
    these messages are spurious and would otherwise flood the log
    with one line per BUFR message.

    The redirect only covers C-level writes; Python ``print``,
    ``logger`` and exceptions still propagate normally. We also
    flush ``sys.stderr`` first so any pending Python-side stderr
    is preserved.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
    finally:
        os.close(devnull_fd)
        os.close(saved_fd)


# ─────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────


def safe_int(v: Any) -> int:
    """Convert a value to int, handling bytes, strings, None, and floats."""
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, bytes):
        s = v.decode("ascii", errors="replace").strip()
    elif v is None:
        s = ""
    else:
        s = str(v).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def _str(v: Any) -> str:
    """Convert a value to a stripped ASCII string."""
    if isinstance(v, bytes):
        return v.decode("ascii", errors="replace").strip()
    if v is None:
        return ""
    return str(v).strip()


def fxy_to_id(f: Any, x: Any, y: Any) -> int:
    """Convert F, X, Y fields to an integer BUFR descriptor ID."""
    return safe_int(f) * 100000 + safe_int(x) * 1000 + safe_int(y)


# ─────────────────────────────────────────────────────────────────────
# DX table extraction
# ─────────────────────────────────────────────────────────────────────


def extract_dx_tables(
    flat: list[Any],
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Extract NCEP Table B and D entries from a DX-message subset.

    DX table messages (dataCategory=11) contain embedded NCEP-local BUFR
    descriptor definitions.  The flat decoded values follow the layout
    produced by the NCEP BUFRLIB DX table encoding:

    - n_table_a, [table_a entries (3 fields each)...]
    - n_table_b, [table_b entries (11 fields each)...]
    - n_table_d, [table_d entries (variable-length)...]

    Each Table B entry has 11 fields:
        F, X, Y, mnemonic(32), desc_cont(32), unit(24),
        sign_scale(1), scale(3), sign_ref(1), reference(10), width(3)

    Each Table D entry has:
        F, X, Y, mnemonic(64), n_members,
        [member F, X, Y, ...]

    Parameters
    ----------
    flat : list
        Decoded values from a single DX message subset.
    table_b : dict
        Accumulator dict to update with Table B entries (pybufrkit format).
    table_d : dict
        Accumulator dict to update with Table D entries.
    """
    n = len(flat)
    idx = 0
    if idx >= n:
        return
    n_a = safe_int(flat[idx])
    idx += 1
    idx += n_a * 3
    if idx >= n:
        return

    # Table B
    n_b = safe_int(flat[idx])
    idx += 1
    for _ in range(n_b):
        if idx + 10 >= n:
            return
        f_v = flat[idx]
        x_v = flat[idx + 1]
        y_v = flat[idx + 2]
        mnemonic = _str(flat[idx + 3])
        unit = _str(flat[idx + 5])
        sign_scale = _str(flat[idx + 6])
        scale_s = _str(flat[idx + 7])
        sign_ref = _str(flat[idx + 8])
        ref_s = _str(flat[idx + 9])
        width_s = _str(flat[idx + 10])
        idx += 11

        desc_id = fxy_to_id(f_v, x_v, y_v)
        if desc_id == 0:
            continue
        scale = safe_int(scale_s)
        if sign_scale == "-":
            scale = -scale
        reference = safe_int(ref_s)
        if sign_ref == "-":
            reference = -reference
        width = safe_int(width_s)
        table_b[desc_id] = (
            mnemonic,
            unit,
            scale,
            reference,
            width,
            unit,  # crex_unit = same as unit
            scale,  # crex_scale = same as scale
            max(1, (width + 3) // 4),  # crex_nchars approximation
        )

    # Table D
    if idx >= n:
        return
    n_d = safe_int(flat[idx])
    idx += 1
    for _ in range(n_d):
        if idx + 3 >= n:
            return
        f_v = flat[idx]
        x_v = flat[idx + 1]
        y_v = flat[idx + 2]
        seq_mnemonic = _str(flat[idx + 3])
        idx += 4
        seq_id = fxy_to_id(f_v, x_v, y_v)
        if seq_id == 0:
            continue
        if idx >= n:
            return
        n_members = safe_int(flat[idx])
        idx += 1
        members: list[str] = []
        for _ in range(n_members):
            if idx >= n:
                break
            members.append(_str(flat[idx]))
            idx += 1
        if members:
            table_d[seq_id] = (seq_mnemonic, members)


# ─────────────────────────────────────────────────────────────────────
# Table registration & decoder creation
# ─────────────────────────────────────────────────────────────────────


def register_dx_tables(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Reset pybufrkit's table cache and (re-)register NCEP DX tables.

    Parameters
    ----------
    table_b : dict
        NCEP Table B entries in pybufrkit format.
    table_d : dict
        NCEP Table D entries.
    """
    TableGroupCacheManager.clear_extra_entries()
    try:
        TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
    except AttributeError as exc:
        logger.warning(
            f"pybufrkit TableGroupCacheManager._TABLE_GROUP_CACHE not available "
            f"({exc}); skipping cache invalidation"
        )
    if table_b or table_d:
        TableGroupCacheManager.add_extra_entries(table_b, table_d)


def create_decoder(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> Any:
    """Register custom NCEP tables and create a pybufrkit decoder.

    Parameters
    ----------
    table_b : dict
        NCEP Table B entries.
    table_d : dict
        NCEP Table D entries.

    Returns
    -------
    pybufrkit.decoder.Decoder
        Configured decoder instance.
    """
    register_dx_tables(table_b, table_d)
    return BufrDecoder()


# ─────────────────────────────────────────────────────────────────────
# PrepBUFR message parsing
# ─────────────────────────────────────────────────────────────────────


def parse_prepbufr_messages(
    file_data: bytes,
    silence_noise: bool = True,
) -> tuple[
    dict[int, tuple[Any, ...]],
    dict[int, tuple[Any, ...]],
    list[tuple[bytes, int]],
]:
    """Split a PrepBUFR byte stream into messages and extract DX tables.

    The first several messages in a PrepBUFR file are DX-table messages
    (dataCategory=11) containing NCEP-local BUFR Table B and Table D
    definitions needed to decode subsequent data messages.

    Parameters
    ----------
    file_data : bytes
        Entire PrepBUFR file contents.
    silence_noise : bool
        If True (default), suppress noisy C-level stderr from pybufrkit
        during DX table decoding.

    Returns
    -------
    tuple[dict, dict, list[tuple[bytes, int]]]
        (table_b_dict, table_d_dict, data_messages) where the dicts
        are in pybufrkit ``add_extra_entries`` format and
        data_messages is a list of (message_bytes, data_category) tuples
        for all non-DX messages.
    """
    table_b: dict[int, tuple[Any, ...]] = {}
    table_d: dict[int, tuple[Any, ...]] = {}
    data_messages: list[tuple[bytes, int]] = []
    dx_messages: list[bytes] = []

    # Split into individual BUFR messages
    pos = 0
    while pos < len(file_data):
        idx = file_data.find(b"BUFR", pos)
        if idx == -1:
            break
        # BUFR edition 3/4: message length in bytes 5-7 (3 bytes, big-endian)
        msg_len = struct.unpack(">I", b"\x00" + file_data[idx + 4 : idx + 7])[0]
        if msg_len < 8:
            pos = idx + 4
            continue
        msg_bytes = file_data[idx : idx + msg_len]

        # BUFR ed3/4: section-0 = 8 bytes, section-1 octet-9 (offset 16)
        # is dataCategory
        data_cat = file_data[idx + 16] if idx + 16 < len(file_data) else 0
        if data_cat == 11:
            dx_messages.append(msg_bytes)
        else:
            data_messages.append((msg_bytes, data_cat))
        pos = idx + msg_len

    # Decode DX table messages using pybufrkit (they use standard descriptors)
    if dx_messages:
        ctx = silence_bufr_noise() if silence_noise else contextlib.nullcontext()
        with ctx:
            try:
                dx_decoder = BufrDecoder()
                for dx_bytes in dx_messages:
                    try:
                        dx_msg = dx_decoder.process(dx_bytes)
                    except Exception:  # noqa: S112
                        logger.debug("Skipping unparseable DX-table message")
                        continue
                    td = dx_msg.template_data.value
                    dvas = td.decoded_values_all_subsets
                    if not dvas:
                        continue
                    extract_dx_tables(dvas[0], table_b, table_d)
            except Exception as e:
                logger.warning(f"Failed to extract DX tables: {e}")

    return table_b, table_d, data_messages


# ─────────────────────────────────────────────────────────────────────
# Process-pool worker initialization
# ─────────────────────────────────────────────────────────────────────

# Module-level decoder for worker processes, set by init_decode_worker.
_worker_decoder: Any = None


def init_decode_worker(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Initializer for process pool workers.

    Registers NCEP-local descriptor tables with pybufrkit in each
    worker process and creates a reusable decoder instance stored
    as a module-level global (``_worker_decoder``).

    Parameters
    ----------
    table_b : dict
        NCEP Table B entries in pybufrkit format.
    table_d : dict
        NCEP Table D entries.
    """
    global _worker_decoder  # noqa: PLW0603
    register_dx_tables(table_b, table_d)
    _worker_decoder = BufrDecoder()


def get_worker_decoder() -> Any:
    """Return the module-level worker decoder instance.

    Returns
    -------
    pybufrkit.decoder.Decoder
        The decoder created by :func:`init_decode_worker`.
    """
    return _worker_decoder


def _raw_element(descriptor: Any, value: Any) -> dict[str, Any]:
    descriptor_name = getattr(descriptor, "name", None)
    name = (
        str(descriptor_name).strip().split(maxsplit=1)[0] if descriptor_name else None
    )
    element: dict[str, Any] = {
        "descriptor_id": int(descriptor.id),
        "field_name": name,
        "numeric_value": None,
        "string_value": None,
        "bytes_value": None,
    }
    if isinstance(value, bytes):
        element["bytes_value"] = value
    elif isinstance(value, str):
        element["string_value"] = value
    elif isinstance(value, numbers.Real):
        element["numeric_value"] = float(value)
    elif value is not None:
        element["string_value"] = str(value)
    return element


def _decode_raw_message(
    work: tuple[int, bytes, int],
) -> list[dict[str, Any]]:
    message_index, message_bytes, data_category = work
    with silence_bufr_noise():
        message = _worker_decoder.process(message_bytes)
    if not message.n_subsets.value:
        return []

    template = message.template_data.value
    rows: list[dict[str, Any]] = []
    for subset_index, (descriptors, values) in enumerate(
        zip(
            template.decoded_descriptors_all_subsets,
            template.decoded_values_all_subsets,
            strict=True,
        )
    ):
        rows.append(
            {
                "message_index": message_index,
                "subset_index": subset_index,
                "data_category": data_category,
                "elements": [
                    _raw_element(descriptor, value)
                    for descriptor, value in zip(descriptors, values, strict=True)
                ],
            }
        )
    return rows


@check_optional_dependencies(BUFR_DEPENDENCY_KEY)
class PyBufrKitDecoder:
    """Decode BUFR files into the canonical raw Arrow representation.

    Parameters
    ----------
    decode_workers : int, optional
        Number of parallel message-decoding processes, by default 8.
    """

    def __init__(self, decode_workers: int = 8) -> None:
        self.decode_workers = max(1, decode_workers)

    def decode_file(self, path: str | Path) -> list[pa.RecordBatch]:
        """Decode one BUFR file without product-specific interpretation.

        Parameters
        ----------
        path : str | Path
            Local BUFR file.

        Returns
        -------
        list[pa.RecordBatch]
            Raw decoded subsets with :data:`RAW_BUFR_SCHEMA`.
        """
        table_b, table_d, messages = parse_prepbufr_messages(
            Path(path).read_bytes(), silence_noise=True
        )
        work = [
            (message_index, message_bytes, data_category)
            for message_index, (message_bytes, data_category) in enumerate(messages)
        ]
        rows: list[dict[str, Any]] = []
        if self.decode_workers == 1:
            init_decode_worker(table_b, table_d)
            for message_rows in map(_decode_raw_message, work):
                rows.extend(message_rows)
        else:
            with ProcessPoolExecutor(
                max_workers=self.decode_workers,
                initializer=init_decode_worker,
                initargs=(table_b, table_d),
            ) as pool:
                for message_rows in pool.map(_decode_raw_message, work, chunksize=1):
                    rows.extend(message_rows)

        if not rows:
            return []
        table = pa.Table.from_pylist(rows, schema=RAW_BUFR_SCHEMA)
        return validate_raw_bufr_batches(table.to_batches(max_chunksize=256))
