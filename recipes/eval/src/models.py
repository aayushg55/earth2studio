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

from __future__ import annotations

from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, open_dict

from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel

from .distributed import run_on_rank0_first


def resolve_member_load_args(
    node: DictConfig, member_ids: tuple[int, ...], ensemble_size: int
) -> None:
    """Fold ``load_args_by_member`` into ``load_args`` for this rank's members.

    ``load_args_by_member`` maps a ``load_model`` argument to a list of values;
    the ensemble's members are split into equal consecutive blocks, one per
    value, so with ``{checkpoint_member: [1, 2, 3, 4]}`` and 16 members,
    members 0-3 load checkpoint 1, members 4-7 checkpoint 2, and so on. A rank
    loads one model, so its members must fall in one block. Removes
    ``load_args_by_member`` from ``node`` once resolved.
    """
    by_member = node.get("load_args_by_member")
    if not by_member:
        return
    resolved = {}
    for name, values in by_member.items():
        values = list(values)
        if ensemble_size % len(values):
            raise ValueError(
                f"load_args_by_member.{name}: {len(values)} values do not split "
                f"{ensemble_size} members evenly"
            )
        blocks = {m * len(values) // ensemble_size for m in member_ids}
        if len(blocks) != 1:
            raise ValueError(
                f"load_args_by_member.{name}: members {list(member_ids)} of this "
                f"rank span blocks {sorted(blocks)}; choose an ensemble layout "
                "whose ranks hold members of one block"
            )
        resolved[name] = values[blocks.pop()]
    with open_dict(node):
        node.load_args = {**dict(node.get("load_args") or {}), **resolved}
        del node["load_args_by_member"]
    logger.info(f"members {list(member_ids)}: load_args {resolved}")


def load_prognostic(
    cfg: DictConfig,
    node: DictConfig | None = None,
) -> PrognosticModel:
    """Load a prognostic model from the Hydra config.

    The model config node must contain an ``architecture`` key whose
    value is the fully-qualified class name of a prognostic model (e.g.
    ``earth2studio.models.px.DLWP``).  The class is expected to expose the
    standard ``load_default_package`` / ``load_model`` classmethods from the
    ``AutoModelMixin`` protocol.

    Any extra keyword arguments under ``load_args`` are forwarded to
    ``load_model``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with a ``model`` section.
    node : DictConfig | None
        Model config node to load from.  Defaults to ``cfg.model``;
        pipelines whose model config nests the prognostic (e.g.
        ``cfg.model.forecast`` for DA-initialized forecasts) pass the
        nested node explicitly.

    Returns
    -------
    PrognosticModel
        Loaded (but not yet device-placed) prognostic model.
    """
    model_cfg = node if node is not None else cfg.model
    cls = hydra.utils.get_class(model_cfg.architecture)

    if model_cfg.get("package_path"):
        from earth2studio.models.auto import Package

        pkg = Package(model_cfg.package_path)
    else:
        pkg = run_on_rank0_first(cls.load_default_package)

    load_kwargs: dict[str, Any] = dict(model_cfg.get("load_args", {}))
    # Unresolved (a model loaded only for its coordinates): the first value.
    for name, values in (model_cfg.get("load_args_by_member") or {}).items():
        load_kwargs.setdefault(name, list(values)[0])
    model: PrognosticModel = cls.load_model(package=pkg, **load_kwargs)

    logger.success(f"Loaded prognostic model: {cls.__name__}")
    return model


def load_diagnostics(cfg: DictConfig) -> list[DiagnosticModel]:
    """Load diagnostic models listed in the config.

    Each entry under ``diagnostics`` should be a Hydra-instantiable object
    (``_target_`` style) or have an ``architecture`` key following the same
    pattern as the prognostic model.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with an optional ``diagnostics`` section.

    Returns
    -------
    list[DiagnosticModel]
        Loaded diagnostic models (may be empty).
    """
    if "diagnostics" not in cfg:
        return []

    models: list[DiagnosticModel] = []
    for name, dx_cfg in cfg.diagnostics.items():
        if "_target_" in dx_cfg:
            dx = hydra.utils.instantiate(dx_cfg)
        elif "architecture" in dx_cfg:
            cls = hydra.utils.get_class(dx_cfg.architecture)
            pkg = run_on_rank0_first(cls.load_default_package)
            load_kwargs: dict[str, Any] = dict(dx_cfg.get("load_args", {}))
            dx = cls.load_model(package=pkg, **load_kwargs)
        else:
            raise ValueError(
                f"Diagnostic '{name}' must have '_target_' or 'architecture'. "
                f"Got keys: {list(dx_cfg.keys())}"
            )
        models.append(dx)
        logger.success(f"Loaded diagnostic model: {name}")

    return models
