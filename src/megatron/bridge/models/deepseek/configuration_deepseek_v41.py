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
"""Transformers-side configuration for DeepSeek-V4.1-Flash.

The checkpoint's ``config.json`` is a composite: ``model_type = "deepseek_v41"``
carries the language model under ``text_config`` (``deepseek_v41_text``) and a
projector-side under ``vision_config`` (``deepseek_v41_vision``).

Registering these three types with ``AutoConfig`` is what lets
``AutoConfig.from_pretrained(<checkpoint>)`` resolve the family with nothing but
``transformers`` installed. Without it that call raises ``KeyError:
'deepseek_v41'``, and every consumer that only needs the config — the training
stack's ``HFModelConfig``, conversion scripts, anything that reads
``hidden_size`` before touching weights — has to reach for a serving library
instead. The sibling families in this package (``step3p5``, ``step3p7``,
``bailing_moe_v2``, ``bagel``, ``qwen3_asr``) already register themselves; this
file closes the same gap for DeepSeek-V4.1.

``model_type`` spellings are *external identifiers*: they come from the published
``config.json``. They are deliberately not renamed to the Python class spelling,
or ``from_pretrained`` on a real checkpoint would route elsewhere.
"""

from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig

__all__ = [
    "DeepSeekV41Config",
    "DeepSeekV41TextConfig",
    "DeepSeekV41VisionConfig",
]


class DeepSeekV41TextConfig(PretrainedConfig):
    """The language-model half of DeepSeek-V4.1-Flash (``deepseek_v41_text``).

    Only the fields something outside the bridge reads are named explicitly; the
    rest of the checkpoint's 59 text keys are kept as attributes through
    ``**kwargs`` so a config read here matches the file on disk field for field.
    ``attribute_map`` carries the transformer-library spelling of two names that
    the checkpoint writes in the model's own vocabulary.
    """

    model_type = "deepseek_v41_text"
    # No `attribute_map`: it renames attributes at construction time (the aliased key is
    # dropped and only its target is set), which silently removed `num_hidden_layers` from
    # the config object and made the Megatron config construction fail on
    # `KeyError: 'num_hidden_layers'`. The checkpoint's own spellings are the ones consumers
    # read, so they stay as they are.

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        moe_intermediate_size: int = 2304,
        max_position_embeddings: int = 1_048_576,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.moe_intermediate_size = moe_intermediate_size
        # `max_position_embeddings` is not a `PretrainedConfig` field in this transformers
        # line, so it survives only because it is named here — and consumers read it before
        # any weight is touched (measured: a rank died on
        # `AttributeError: ... has no attribute 'max_position_embeddings'`).
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Every remaining field the checkpoint carries stays addressable: the config a
        # consumer sees has to match the file on disk, not the subset this class names.
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class DeepSeekV41VisionConfig(PretrainedConfig):
    """The projector-side half (``deepseek_v41_vision``).

    No bridge code reads it — the Megatron-side model is text-only — but the
    composite ``config.json`` declares it, so it is registered as a plain
    pass-through rather than left to fail resolution when a consumer walks the
    sub-configs.
    """

    model_type = "deepseek_v41_vision"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class DeepSeekV41Config(DeepSeekV41TextConfig):
    """The checkpoint's top-level config (``model_type = "deepseek_v41"``).

    It is the language-model config **flattened**, not a composite: the reader this lane
    used before registration resolved a composite checkpoint to the text config and built
    the Megatron geometry from that shape, so the class keeps the same field set under a
    different ``model_type``. Returning a nested view instead produced a model whose
    projections did not match the checkpoint's FP8 scale grids.

    The nested halves remain resolvable on their own (``deepseek_v41_text``,
    ``deepseek_v41_vision``) for callers that walk them.
    """

    model_type = "deepseek_v41"

    def __init__(self, text_config: Optional[dict[str, Any]] = None, **kwargs: Any):
        # The published `config.json` is a composite: the language model lives under
        # `text_config`. This class is the flattened view, so the nested block is merged in
        # before construction — with any explicitly passed field winning over it, which is
        # what makes `for_model(..., hidden_size=...)` behave for programmatic callers.
        if isinstance(text_config, dict):
            kwargs = {**text_config, **kwargs}
        super().__init__(**kwargs)
