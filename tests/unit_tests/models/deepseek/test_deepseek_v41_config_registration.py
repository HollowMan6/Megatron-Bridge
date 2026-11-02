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

"""DeepSeek-V4.1 resolves through transformers alone.

The property under test is not "the class exists" but "`AutoConfig.from_pretrained`
works on a real checkpoint layout with nothing but transformers installed". That is the
case that used to fail with `KeyError: 'deepseek_v41'` and push consumers onto a serving
library's config reader, so the test builds the composite `config.json` shape on disk and
resolves it through the public entry point.
"""

import json

import pytest
from transformers import AutoConfig

import megatron.bridge.models.deepseek.deepseek_v41_bridge  # noqa: F401  (import registers the family)
from megatron.bridge.models.deepseek.configuration_deepseek_v41 import (
    DeepSeekV41Config,
    DeepSeekV41TextConfig,
    DeepSeekV41VisionConfig,
)


def _write_checkpoint_config(directory):
    """Write the composite config.json the published checkpoint carries."""
    payload = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 129_343,
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 129_280,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "moe_intermediate_size": 2304,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 512,
            "max_position_embeddings": 1_048_576,
            "rope_theta": 10_000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65_536,
            },
        },
        "vision_config": {"model_type": "deepseek_v41_vision", "hidden_size": 1024},
    }
    (directory / "config.json").write_text(json.dumps(payload))
    return directory


def test_auto_config_resolves_the_composite_checkpoint(tmp_path):
    config = AutoConfig.from_pretrained(_write_checkpoint_config(tmp_path))

    # `deepseek_v41` resolves to the language-model config (flattened), which is the shape
    # the previous reader produced and the one the Megatron geometry is built from.
    assert type(config) is DeepSeekV41Config
    assert config.model_type == "deepseek_v41"
    assert config.hidden_size == 5120
    assert config.num_hidden_layers == 40


def test_text_config_keeps_the_fields_consumers_read(tmp_path):
    """The nested block must survive as typed attributes, not a raw dict.

    `HFModelConfig` reads these before any weight is touched, so a dict here would move
    the failure to a later, more expensive place.
    """
    config = AutoConfig.from_pretrained(_write_checkpoint_config(tmp_path))
    text = config

    assert text.hidden_size == 5120
    assert text.num_hidden_layers == 40
    assert text.num_attention_heads == 64
    assert text.vocab_size == 129_280
    # Keys this bridge does not name still travel as attributes: the config a consumer
    # sees has to match the file on disk, not a curated subset of it.
    assert text.index_topk == 512
    assert text.index_n_heads == 32


def test_top_level_config_exposes_the_language_model_fields(tmp_path):
    """The conversion path reads the LM geometry off the object it is handed.

    Measured: `KeyError: 'num_hidden_layers'` inside the Megatron config construction when
    only the nested form is available, because the reader that used to supply this config
    flattens a composite checkpoint. Both forms have to answer.
    """
    config = AutoConfig.from_pretrained(_write_checkpoint_config(tmp_path))

    assert config.num_hidden_layers == 40
    assert config.hidden_size == 5120
    assert config.max_position_embeddings == 1_048_576


def test_position_and_rope_fields_survive(tmp_path):
    """The fields a loader reads before touching weights.

    `max_position_embeddings` is not a `PretrainedConfig` field in this transformers line;
    it reaches the object only because the class names it. It was measured missing on a
    real rank (`AttributeError: ... has no attribute 'max_position_embeddings'`), which is
    why it is pinned here rather than assumed.
    """
    config = AutoConfig.from_pretrained(_write_checkpoint_config(tmp_path))
    text = config

    assert text.max_position_embeddings == 1_048_576
    assert text.rope_theta == 10000
    assert text.rope_scaling is not None


def test_text_config_alone_resolves():
    """A bare text config resolves too — some consumers load only the LM half."""
    config = AutoConfig.for_model("deepseek_v41_text", hidden_size=7168, num_hidden_layers=61)

    assert type(config) is DeepSeekV41TextConfig
    assert config.hidden_size == 7168
    assert config.num_hidden_layers == 61


def test_registration_does_not_require_vllm(monkeypatch):
    """Resolution works with the serving stack absent.

    This is the regression the registration exists for: the previous path fell through to
    `vllm.transformers_utils.config` when transformers could not resolve `deepseek_v41`.
    """
    import sys

    monkeypatch.setitem(sys.modules, "vllm", None)
    config = AutoConfig.for_model("deepseek_v41")
    assert type(config) is DeepSeekV41Config
    assert config.hidden_size == 5120
