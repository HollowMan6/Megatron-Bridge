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
"""GLM-5.3-Flash builder/provider kwargs must reach declared Core configuration fields."""

from dataclasses import fields

import pytest

from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashBridge, GLM53FlashModelProvider
from megatron.bridge.models.glm5next.glm53flash_config import GLM53FlashModelConfig, GLM53FlashTransformerConfig


def _hf_config():
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    return config_module.Glm5NextConfig(
        text_config={
            "scoring_func": "sigmoid",
            "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4},
        }
    )


@pytest.mark.parametrize("path", ["provider", "model_config"])
def test_config_conversion_only_sets_declared_fields(path):
    """A misspelled flat kwarg must fail even when a generic constructor would filter it out."""
    bridge = GLM53FlashBridge()
    if path == "provider":
        kwargs = bridge.hf_config_to_provider_kwargs(_hf_config())
        declared = {field.name for field in fields(GLM53FlashModelProvider)}
    else:
        kwargs = bridge.hf_config_to_model_config_kwargs(_hf_config())
        declared = {field.name for field in fields(GLM53FlashModelConfig)} | {
            field.name for field in fields(GLM53FlashTransformerConfig)
        }
    unknown = sorted(set(kwargs) - declared)
    assert not unknown, f"Undeclared {path} configuration keys: {unknown}"


@pytest.mark.parametrize("path", ["provider", "model_config"])
def test_mhc_uses_the_mcore_field_names(path):
    """Core902 uses enable_mhc_connections/mhc_num_residual_streams on both paths."""
    bridge = GLM53FlashBridge()
    hf_config = _hf_config()
    if path == "provider":
        kwargs = bridge.hf_config_to_provider_kwargs(hf_config)
        declared = {field.name for field in fields(GLM53FlashModelProvider)}
    else:
        kwargs = bridge.hf_config_to_model_config_kwargs(hf_config)
        declared = {field.name for field in fields(GLM53FlashTransformerConfig)}
    assert {"enable_mhc_connections", "mhc_num_residual_streams"} <= declared
    assert kwargs["enable_mhc_connections"] is True
    assert kwargs["mhc_num_residual_streams"] == hf_config.text_config.hc_mult
    assert "enable_hyper_connections" not in kwargs
    assert "num_residual_streams" not in kwargs
