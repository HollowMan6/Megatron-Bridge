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

"""Unit tests for ``ExpertSlotRelay.derive_perm1_from_fused_map``.

The helper's fail-fast guards read device scalars back to the host, which
stalls the MoE dispatch path (the helper runs once per MoE layer per pass,
called ahead of the dispatch all-to-all), so they are opt-in via the module
constant ``megatron.bridge.peft.batched.lora._VERIFY_RELAY_DEVICE_GUARDS``.
These tests use CPU tensors and pin both states of that switch: the guarded
branches still raise the original failure text when it is on, the returned
slot vector is unchanged when it is off, and the default path performs no
host-side scalar read at all.
"""

import pytest
import torch

from megatron.bridge.peft.batched import lora as batched_lora
from megatron.bridge.peft.batched.lora import ExpertSlotRelay


pytestmark = pytest.mark.unit

NUM_EXPERTS = 2
NUM_OUT = 6
EXPECTED_PERM1 = [10, 10, 20, 20, 30, 30]


def _local_slots():
    """This rank's CP-chunk slots, one per input row of the routed surface."""
    return torch.tensor([10, 20, 30], dtype=torch.int64)


def _row_id_map(n_routed_per_token, rows_per_token):
    """Fused TE row_id_map [S, 2E+1] int32 (see the helper's docstring).

    Per token: ``[t, 2E]`` routed count, ``[t, :n_routed]`` destination rows,
    ``[t, E:E+n_routed]`` expert ids. Only the first two are read by the
    helper under test.
    """
    width = 2 * NUM_EXPERTS + 1
    row_id_map = torch.zeros((len(n_routed_per_token), width), dtype=torch.int32)
    for token, (n_routed, rows) in enumerate(zip(n_routed_per_token, rows_per_token)):
        row_id_map[token, 2 * NUM_EXPERTS] = n_routed
        row_id_map[token, :n_routed] = torch.tensor(rows[:n_routed], dtype=torch.int32)
    return row_id_map


def _valid_row_id_map():
    """Three tokens x topk 2 -> destination rows 0..5, one routed pair each."""
    return _row_id_map([2, 2, 2], [[0, 1], [2, 3], [4, 5]])


# The three guarded reads all end in a Python truth test on a 0-dim device
# tensor: ``rows.min() < 0``, ``rows.max() >= num_out`` (both via the ``or``
# short-circuit) and ``bool((perm1 < 0).any())``. Counting
# ``Tensor.__bool__`` therefore counts exactly the host-side device reads,
# identically on CPU and on CUDA (where each one is a blocking D2H copy).
class _HostScalarReadCounter:
    def __init__(self):
        self.count = 0

    def __enter__(self):
        self._original = torch.Tensor.__bool__

        def _counting_bool(tensor):
            self.count += 1
            return self._original(tensor)

        torch.Tensor.__bool__ = _counting_bool
        return self

    def __exit__(self, *exc_info):
        torch.Tensor.__bool__ = self._original
        return False


def test_guards_are_off_by_default():
    assert batched_lora._VERIFY_RELAY_DEVICE_GUARDS is False


def test_default_path_builds_perm1_without_host_scalar_reads(monkeypatch):
    monkeypatch.setattr(batched_lora, "_VERIFY_RELAY_DEVICE_GUARDS", False)
    with _HostScalarReadCounter() as counter:
        perm1 = ExpertSlotRelay.derive_perm1_from_fused_map(_valid_row_id_map(), _local_slots(), NUM_OUT)
    assert counter.count == 0
    assert perm1.dtype == torch.int64
    assert perm1.tolist() == EXPECTED_PERM1


def test_guarded_path_keeps_the_valid_result(monkeypatch):
    monkeypatch.setattr(batched_lora, "_VERIFY_RELAY_DEVICE_GUARDS", True)
    with _HostScalarReadCounter() as counter:
        perm1 = ExpertSlotRelay.derive_perm1_from_fused_map(_valid_row_id_map(), _local_slots(), NUM_OUT)
    # rows.min(), rows.max(), (perm1 < 0).any() — the three reads the guards add.
    assert counter.count == 3
    assert perm1.tolist() == EXPECTED_PERM1


@pytest.mark.parametrize(
    "row_id_map, message",
    [
        # fewer routed pairs than permuted rows
        (_row_id_map([2, 2, 1], [[0, 1], [2, 3], [4]]), "does not describe a valid permutation"),
        # destination row outside [0, num_out)
        (_row_id_map([2, 2, 2], [[0, 1], [2, 3], [4, 99]]), "does not describe a valid permutation"),
        # in-range rows that leave a permuted row uncovered
        (_row_id_map([2, 2, 2], [[0, 1], [2, 2], [4, 5]]), "does not cover all"),
    ],
)
def test_guarded_path_keeps_the_failure_text(row_id_map, message, monkeypatch):
    monkeypatch.setattr(batched_lora, "_VERIFY_RELAY_DEVICE_GUARDS", True)
    with pytest.raises(RuntimeError, match=message):
        ExpertSlotRelay.derive_perm1_from_fused_map(row_id_map, _local_slots(), NUM_OUT)


def test_default_path_does_not_raise_on_the_maps_the_guards_reject(monkeypatch):
    """The guards are diagnostics, so turning them off deliberately drops the raise.

    Pinned so the trade-off stays explicit: an uncovered row now reaches the
    caller as the ``-1`` sentinel instead of failing fast in the helper (the
    downstream row-count check in ``take_current`` is host-side metadata and
    still runs).
    """
    monkeypatch.setattr(batched_lora, "_VERIFY_RELAY_DEVICE_GUARDS", False)
    perm1 = ExpertSlotRelay.derive_perm1_from_fused_map(
        _row_id_map([2, 2, 2], [[0, 1], [2, 2], [4, 5]]), _local_slots(), NUM_OUT
    )
    assert perm1.tolist() == [10, 10, 20, -1, 30, 30]
