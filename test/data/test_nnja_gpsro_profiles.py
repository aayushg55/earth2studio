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

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from earth2studio.data.nnja import NNJAGPSRO, _NNJAGPSROProfileTask
from earth2studio.data.utils_bufr import (
    RAW_BUFR_SCHEMA,
    BUFRDecoder,
    PyBufrKitDecoder,
    validate_raw_bufr_batches,
)
from earth2studio.data.utils_ncep_gpsro import (
    GPSRO_ARFR,
    GPSRO_BNDA,
    GPSRO_DAY,
    GPSRO_ELRC,
    GPSRO_GEODU,
    GPSRO_HEIT,
    GPSRO_HOUR,
    GPSRO_IMPP,
    GPSRO_LAT,
    GPSRO_LON,
    GPSRO_MEFR,
    GPSRO_MIN,
    GPSRO_MONTH,
    GPSRO_PTID,
    GPSRO_QFRO,
    GPSRO_SAID,
    GPSRO_SEC,
    GPSRO_YEAR,
    NCEP_GPSRO_PROFILE_SCHEMA,
    transform_gpsro_profiles,
)


def _element(descriptor_id, value):
    return {
        "descriptor_id": descriptor_id,
        "field_name": None,
        "numeric_value": value,
        "string_value": None,
        "bytes_value": None,
    }


def _raw_batch(stream):
    return pa.RecordBatch.from_pylist(
        [
            {
                "message_index": 7,
                "subset_index": 2,
                "data_category": 3,
                "elements": [
                    _element(descriptor_id, value) for descriptor_id, value in stream
                ],
            }
        ],
        schema=RAW_BUFR_SCHEMA,
    )


def _profile_stream():
    return [
        (GPSRO_SAID, 3),
        (GPSRO_PTID, 27),
        (GPSRO_QFRO, 2048),
        (GPSRO_YEAR, 2022),
        (GPSRO_MONTH, 1),
        (GPSRO_DAY, 1),
        (GPSRO_HOUR, 3),
        (GPSRO_MIN, 30),
        (GPSRO_SEC, 15.25),
        (GPSRO_LAT, -10.5),
        (GPSRO_LON, -70.25),
        (GPSRO_ELRC, 6_371_000.0),
        (GPSRO_GEODU, 8.25),
        (GPSRO_LAT, -9.75),
        (GPSRO_LON, -69.5),
        (GPSRO_MEFR, 1_500_000_000.0),
        (GPSRO_IMPP, 6_373_000.0),
        (GPSRO_BNDA, 0.00999),
        (GPSRO_BNDA, 0.00088),
        (GPSRO_MEFR, 1_200_000_000.0),
        (GPSRO_IMPP, 6_373_000.0),
        (GPSRO_BNDA, 0.00777),
        (GPSRO_BNDA, 0.00066),
        (GPSRO_MEFR, 0.0),
        (GPSRO_IMPP, 6_373_000.0),
        (GPSRO_BNDA, 0.00123),
        (GPSRO_BNDA, 0.00045),
        (GPSRO_LAT, 91.0),
        (GPSRO_LON, 21.0),
        (GPSRO_MEFR, 0.0),
        (GPSRO_IMPP, 6_374_000.0),
        (GPSRO_BNDA, 0.00234),
        (GPSRO_BNDA, 0.00056),
        (GPSRO_HEIT, 1_950.0),
        (GPSRO_ARFR, 250.0),
        (GPSRO_ARFR, 2.5),
        (GPSRO_HEIT, 2_050.0),
        (GPSRO_ARFR, None),
        (GPSRO_ARFR, 2.0),
    ]


def _native_nested_batch():
    frequency_type = pa.struct(
        [
            pa.field("MEFR", pa.float64()),
            pa.field("IMPP", pa.float64()),
            pa.field("BNDA", pa.float64()),
            pa.field("BNDA", pa.float64()),
        ]
    )
    frequencies = pa.StructArray.from_arrays(
        [
            pa.array(
                [
                    1_500_000_000.0,
                    1_200_000_000.0,
                    0.0,
                    0.0,
                ]
            ),
            pa.array([6_373_000.0, 6_373_000.0, 6_373_000.0, 6_374_000.0]),
            pa.array([0.00999, 0.00777, 0.00123, 0.00234]),
            pa.array([0.00088, 0.00066, 0.00045, 0.00056]),
        ],
        fields=frequency_type,
    )
    frequency_lists = pa.ListArray.from_arrays(
        pa.array([0, 3, 4], type=pa.int32()), frequencies
    )
    bending_type = pa.struct(
        [
            pa.field("CLATH", pa.float64()),
            pa.field("CLONH", pa.float64()),
            pa.field("DRF8BIT", frequency_lists.type),
        ]
    )
    bending = pa.StructArray.from_arrays(
        [
            pa.array([-9.75, 91.0]),
            pa.array([-69.5, 21.0]),
            frequency_lists,
        ],
        fields=bending_type,
    )
    bending_profiles = pa.ListArray.from_arrays(
        pa.array([0, 2], type=pa.int32()), bending
    )

    refractivity_type = pa.struct(
        [
            pa.field("HEIT", pa.float64()),
            pa.field("ARFR", pa.float64()),
            pa.field("ARFR", pa.float64()),
        ]
    )
    refractivity = pa.StructArray.from_arrays(
        [
            pa.array([1_950.0, 2_050.0]),
            pa.array([250.0, None]),
            pa.array([2.5, 2.0]),
        ],
        fields=refractivity_type,
    )
    refractivity_profiles = pa.ListArray.from_arrays(
        pa.array([0, 2], type=pa.int32()), refractivity
    )

    columns = [
        pa.array([3.0]),
        pa.array([27.0]),
        pa.array([2048.0]),
        pa.array([2022.0]),
        pa.array([1.0]),
        pa.array([1.0]),
        pa.array([3.0]),
        pa.array([30.0]),
        pa.array([15.25]),
        pa.array([-10.5]),
        pa.array([-70.25]),
        pa.array([6_371_000.0]),
        pa.array([8.25]),
        bending_profiles,
        refractivity_profiles,
        pa.array([3.0]),
    ]
    names = [
        "SAID",
        "PTID",
        "QFRO",
        "YEAR",
        "MNTH",
        "DAYS",
        "HOUR",
        "MINU",
        "SECO",
        "CLATH",
        "CLONH",
        "ELRC",
        "GEODU",
        "DRF16BIT",
        "DRF16BIT_2",
        "_data_category",
    ]
    return pa.RecordBatch.from_arrays(columns, names)


def test_transform_gpsro_profiles_preserves_raw_policy_inputs():
    table = transform_gpsro_profiles(
        [_raw_batch(_profile_stream())], datetime(2022, 1, 1)
    )

    assert table.schema == NCEP_GPSRO_PROFILE_SCHEMA
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["profile_id"] == "2022010100:0"
    assert row["source_profile_index"] == 0
    assert row["observation_time"] == datetime(2022, 1, 1, 3, 30, 15, 250000)
    assert row["qfro"] == 2048
    assert row["longitude"] == pytest.approx(-70.25)

    assert len(row["bending_levels"]) == 1
    bending = row["bending_levels"][0]
    assert bending["source_level_index"] == 0
    assert bending["latitude"] == pytest.approx(-9.75)
    assert bending["combined_bending_angle"] == pytest.approx(0.00123)
    assert bending["combined_bending_angle_uncertainty"] == pytest.approx(0.00045)
    assert bending["l1_bending_angle"] == pytest.approx(0.00999)
    assert bending["l2_bending_angle"] == pytest.approx(0.00777)

    assert row["refractivity_levels"] == [
        {
            "height": 1_950.0,
            "atmospheric_refractivity": 250.0,
            "atmospheric_refractivity_uncertainty": 2.5,
        },
        {
            "height": 2_050.0,
            "atmospheric_refractivity": None,
            "atmospheric_refractivity_uncertainty": 2.0,
        },
    ]


def test_transform_gpsro_profiles_accepts_mnemonic_backend_identity():
    field_names = {
        GPSRO_SAID: "SAID",
        GPSRO_PTID: "PTID",
        GPSRO_QFRO: "QFRO",
        GPSRO_YEAR: "YEAR",
        GPSRO_MONTH: "MNTH",
        GPSRO_DAY: "DAYS",
        GPSRO_HOUR: "HOUR",
        GPSRO_MIN: "MINU",
        GPSRO_SEC: "SECO",
        GPSRO_LAT: "CLATH",
        GPSRO_LON: "CLONH",
        GPSRO_ELRC: "ELRC",
        GPSRO_GEODU: "GEODU",
        GPSRO_MEFR: "MEFR",
        GPSRO_IMPP: "IMPP",
        GPSRO_BNDA: "BNDA",
        GPSRO_HEIT: "HEIT",
        GPSRO_ARFR: "ARFR",
    }
    elements = [
        {
            "descriptor_id": None,
            "field_name": field_names[descriptor_id],
            "numeric_value": value,
            "string_value": None,
            "bytes_value": None,
        }
        for descriptor_id, value in _profile_stream()
    ]
    batch = pa.RecordBatch.from_pylist(
        [
            {
                "message_index": 0,
                "subset_index": 0,
                "data_category": 3,
                "elements": elements,
            }
        ],
        schema=RAW_BUFR_SCHEMA,
    )

    by_id = transform_gpsro_profiles(
        [_raw_batch(_profile_stream())], datetime(2022, 1, 1)
    )
    by_name = transform_gpsro_profiles([batch], datetime(2022, 1, 1))

    assert by_name.equals(by_id)


def test_transform_gpsro_profiles_nested_and_element_layouts_are_identical():
    by_element = transform_gpsro_profiles(
        [_raw_batch(_profile_stream())], datetime(2022, 1, 1)
    )
    by_nested = transform_gpsro_profiles([_native_nested_batch()], datetime(2022, 1, 1))

    assert by_nested.equals(by_element)


def test_transform_gpsro_profiles_omits_profile_without_usable_bending_level():
    stream = _profile_stream()
    stream[25] = (GPSRO_BNDA, 0.0)

    table = transform_gpsro_profiles([_raw_batch(stream)], datetime(2022, 1, 1))

    assert table.schema == NCEP_GPSRO_PROFILE_SCHEMA
    assert table.num_rows == 0


def test_nnja_gpsro_routes_injected_decoder_through_shared_transform(tmp_path):
    batch = _raw_batch(_profile_stream())

    class FakeDecoder:
        def __init__(self):
            self.paths = []

        def decode_file(self, path):
            self.paths.append(Path(path))
            return [batch]

    decoder = FakeDecoder()
    assert isinstance(decoder, BUFRDecoder)
    source = NNJAGPSRO(decoder=decoder, cache=False, verbose=False)
    task = _NNJAGPSROProfileTask("s3://example/source.bufr", datetime(2022, 1, 1))
    source_path = tmp_path / "source.bufr"

    table = source._decode_file(source_path, task)

    assert decoder.paths == [source_path]
    assert table.schema == NCEP_GPSRO_PROFILE_SCHEMA
    assert table.num_rows == 1


def test_validate_raw_bufr_batches_rejects_backend_contract_drift():
    wrong = pa.RecordBatch.from_pydict({"value": [1.0]})

    with pytest.raises(TypeError, match="incompatible schema"):
        validate_raw_bufr_batches([wrong])


def test_pybufrkit_decoder_emits_canonical_raw_batches(monkeypatch, tmp_path):
    class Descriptor:
        def __init__(self, descriptor_id):
            self.id = descriptor_id

    template = SimpleNamespace(
        decoded_descriptors_all_subsets=[
            [Descriptor(1007), Descriptor(1194), Descriptor(4001)]
        ],
        decoded_values_all_subsets=[[3.0, b"station", None]],
    )
    message = SimpleNamespace(
        n_subsets=SimpleNamespace(value=1),
        template_data=SimpleNamespace(value=template),
    )

    class Decoder:
        def process(self, message_bytes):
            assert message_bytes == b"message"
            return message

    from earth2studio.data import utils_bufr

    monkeypatch.setattr(
        utils_bufr,
        "parse_prepbufr_messages",
        lambda data, silence_noise: ({}, {}, [(b"message", 3)]),
    )
    monkeypatch.setattr(utils_bufr, "init_decode_worker", lambda table_b, table_d: None)
    monkeypatch.setattr(utils_bufr, "_worker_decoder", Decoder())
    path = tmp_path / "source.bufr"
    path.write_bytes(b"source")

    batches = PyBufrKitDecoder(decode_workers=1).decode_file(path)

    assert len(batches) == 1
    assert batches[0].schema == RAW_BUFR_SCHEMA
    row = batches[0].to_pylist()[0]
    assert row["elements"] == [
        {
            "descriptor_id": 1007,
            "field_name": None,
            "numeric_value": 3.0,
            "string_value": None,
            "bytes_value": None,
        },
        {
            "descriptor_id": 1194,
            "field_name": None,
            "numeric_value": None,
            "string_value": None,
            "bytes_value": b"station",
        },
        {
            "descriptor_id": 4001,
            "field_name": None,
            "numeric_value": None,
            "string_value": None,
            "bytes_value": None,
        },
    ]
