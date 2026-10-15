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
"""GLM-5.3-Flash forward wrapper contract.

``GLM53FlashModel.forward`` is a thin adapter: it packs BSHD inputs into THD layout and
forwards to ``HybridModel.forward``.  A keyword the wrapped model does not declare is not
noticed at import time — it raises ``TypeError`` on the *first* ``forward_backward``, which
on this model costs a full checkpoint load (tens of minutes) to reach.  ``compute_mtp_loss``
was exactly that: the name exists nowhere under ``megatron/core`` on this line, where the MTP
auxiliary loss is produced inside the model by ``process_mtp_loss``
(``megatron/core/models/hybrid/hybrid_model.py:673``).  This test pins the contract so the
next foreign kwarg fails in a unit test rather than after a load.
"""

import ast
import inspect
import pathlib

from megatron.core.models.hybrid.hybrid_model import HybridModel

from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashModel


def _forwarded_kwarg_names() -> list:
    """Keyword names ``GLM53FlashModel.forward`` passes to ``self.language_model.forward``."""
    module_path = inspect.getsourcefile(GLM53FlashModel)
    assert module_path, "cannot locate the module defining GLM53FlashModel"
    tree = ast.parse(pathlib.Path(module_path).read_text())

    cls = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == GLM53FlashModel.__name__),
        None,
    )
    assert cls is not None, "GLM53FlashModel was not found in its own module"
    method = next(
        (item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "forward"),
        None,
    )
    assert method is not None, "GLM53FlashModel.forward() was not found"

    names = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match exactly `self.language_model.forward(...)`.
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "forward"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "language_model"
        ):
            continue
        names.extend(keyword.arg for keyword in node.keywords if keyword.arg)
    return names


def test_forward_wrapper_only_passes_kwargs_the_wrapped_model_declares():
    accepted = set(inspect.signature(HybridModel.forward).parameters)
    passed = _forwarded_kwarg_names()
    assert passed, "no forwarded keywords found — the introspection broke, fix the test"

    unknown = sorted(set(passed) - accepted)
    assert not unknown, (
        "GLM53FlashModel.forward() passes keywords that HybridModel.forward() does not "
        f"declare: {unknown}. That raises TypeError on the first forward_backward, i.e. after "
        "a full checkpoint load. Drop the keyword or map it onto this line's mechanism (the "
        "MTP auxiliary loss here is produced inside the model by process_mtp_loss)."
    )


def test_forward_wrapper_keeps_the_output_processor_path():
    """verl drives training through ``output_processor``; it must stay wired."""
    passed = set(_forwarded_kwarg_names())
    assert "output_processor" in passed
    assert "output_processor_context" in passed
