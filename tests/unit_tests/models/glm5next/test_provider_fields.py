# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""GLM-5.3-Flash provider configuration contract.

``provider_bridge`` configures the model by assigning attributes on the provider dataclass
(``provider.foo = ...``).  A name the dataclass does not declare is *not* an error: the
instance silently keeps it as a stray attribute and the value never reaches the
``TransformerConfig`` built from the provider's fields.  That is exactly how
``provider.enable_mhc_connections`` (MCore's field is ``enable_hyper_connections``) left the
mHC wrapper disabled on GLM-5.3-Flash, which in turn made every mapping-registry pattern
unreachable — all of them are written with the wrapper's ``inner_layer.`` level — and failed
HF->Megatron conversion with 3909 unmapped parameters.

These tests pin the contract so a misspelled field can never no-op again.
"""

import ast
import inspect
import pathlib

from megatron.bridge.models.glm5next.glm53flash_bridge import (
    GLM53FlashBridge,
    GLM53FlashModelProvider,
)

# Assignment targets deliberately not provider fields.  Empty by design: every assignment in
# ``provider_bridge`` is expected to be a real, consumed configuration field.
_NON_FIELD_ASSIGNMENTS: frozenset = frozenset()

_TEXT_CONFIG_ASSIGNMENTS = {"vision_config", "spatial_merge_size", "scatter_embedding_sequence_parallel"}


def _provider_assignments() -> dict:
    """Map every ``provider.<attr>`` name assigned in ``GLM53FlashBridge.provider_bridge`` to
    its line number, by parsing the module that actually defines the class."""
    module_path = inspect.getsourcefile(GLM53FlashBridge)
    assert module_path, "cannot locate the module defining GLM53FlashBridge"
    tree = ast.parse(pathlib.Path(module_path).read_text())

    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == GLM53FlashBridge.__name__:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "provider_bridge":
                    method = item
    assert method is not None, "GLM53FlashBridge.provider_bridge() was not found"

    assignments = {}
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "provider"
            ):
                assignments.setdefault(target.attr, node.lineno)
    return assignments


def test_provider_bridge_only_assigns_declared_fields():
    """No assignment may invent an attribute: a stray name silently drops the configuration."""
    declared = set(GLM53FlashModelProvider.__dataclass_fields__)
    assigned = _provider_assignments()
    assert assigned, "provider_bridge() assigns no provider fields — the introspection broke"

    unknown = sorted(set(assigned) - declared - _NON_FIELD_ASSIGNMENTS)
    assert not unknown, (
        "provider_bridge() assigns attributes that are not declared fields of "
        f"{GLM53FlashModelProvider.__name__}: "
        + ", ".join(f"provider.{name} (line {assigned[name]})" for name in unknown)
        + ". A misspelled field silently no-ops and the value never reaches the "
        "TransformerConfig. Use the real MCore field name, or — if the value genuinely has no "
        "field — drop the assignment and explain why in a comment."
    )


def test_mhc_uses_the_mcore_field_names():
    """The mHC switch must use MCore's names, and mHC must be on for this architecture.

    The GLM-5.3-Flash checkpoint carries ``hc_{attn,ffn}_{fn,base,scale}`` on all 45 layers, so
    the hyper-connection wrapper is part of the architecture: with mHC off the model builds a
    different parameter tree than the mapping registry describes.
    """
    declared = set(GLM53FlashModelProvider.__dataclass_fields__)
    assert "enable_hyper_connections" in declared, "MCore's mHC switch disappeared from the provider"
    assert "num_residual_streams" in declared, "MCore's residual-stream count disappeared"

    assigned = _provider_assignments()
    assert "enable_hyper_connections" in assigned, "provider_bridge() no longer enables mHC"
    assert "num_residual_streams" in assigned, "provider_bridge() no longer sets the stream count"

    for stale in ("enable_mhc_connections", "mhc_num_residual_streams"):
        assert stale not in assigned, (
            f"provider.{stale} is not an MCore field (it silently no-ops); use the MCore name"
        )
