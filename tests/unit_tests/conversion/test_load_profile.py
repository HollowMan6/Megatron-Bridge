# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for the opt-in MINT_LOAD_PROFILE phase timer.

The module is loaded from its file path (not through ``megatron.bridge``) so
these tests exercise the profiler without importing the model stack, which is
slow and unnecessary here. A fresh module object per test keeps its module
level counters independent.
"""

import importlib.util
import itertools
import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "megatron" / "bridge" / "models" / "conversion"
_MODULE_PATH = _MODULE_PATH / "load_profile.py"
_STATE_PATH = _MODULE_PATH.parents[1] / "hf_pretrained" / "state.py"

_SUMMARY_PATTERN = re.compile(
    r"^LOAD_PROFILE: rank \d+ \| \d+ params"
    r" \| read \d+\.\ds \(\d+\.\d%\)"
    r" \| dequant \d+\.\ds"
    r" \| map \d+\.\ds"
    r" \| copy \d+\.\ds"
    r" \| other \d+\.\ds"
    r" \| total \d+\.\ds"
    r" \| \d+\.\d MB/s$"
)

_names = itertools.count()


@pytest.fixture()
def profiler():
    """An isolated copy of the profiler module."""
    spec = importlib.util.spec_from_file_location(f"load_profile_under_test_{next(_names)}", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    """Deterministic ``perf_counter`` stand-in returning a queued timeline."""

    def __init__(self, *timestamps):
        self.timestamps = list(timestamps)

    def __call__(self):
        assert self.timestamps, "perf_counter called more often than the fake timeline allows"
        return self.timestamps.pop(0)


class FakeTensor:
    """Just enough of ``torch.Tensor`` for the byte accounting."""

    def __init__(self, numel, element_size=2):
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakeMapping:
    """A tiny HF->Megatron mapping: fetch, transform, write."""

    def __init__(self, tensors):
        self.tensors = tensors

    def fetch(self, name):
        return list(self.tensors[name])

    def convert(self, values):
        return [value * 2.0 for value in reversed(values)]


def run_loop_body(profiler, mapping, names):
    """Mirror the instrumented per-parameter loop body on plain lists."""
    loaded = {}
    for name in names:
        profiler.count_param()
        with profiler.phase("dequant"):
            with profiler.phase("read"):
                source = mapping.fetch(name)
                profiler.add_read_tensors({name: FakeTensor(len(source))})
        with profiler.phase("map"):
            converted = mapping.convert(source)
        destination = [0.0] * len(converted)
        with profiler.phase("copy"):
            destination[:] = converted
        loaded[name] = destination
    return loaded


def test_unset_env_is_a_strict_no_op(profiler, monkeypatch, capsys):
    monkeypatch.delenv(profiler.ENV_VAR, raising=False)
    profiler.reset()

    assert profiler.is_enabled() is False
    with profiler.phase("read"):
        pass
    profiler.add_read_bytes(1 << 20)
    profiler.add_read_tensors({"w": FakeTensor(8)})
    profiler.count_param()
    token = profiler.begin_load()
    assert token is False
    assert profiler.end_load(token) is None
    assert profiler.report() is None

    snap = profiler.snapshot()
    assert snap.total == 0.0
    assert snap.other == 0.0
    assert snap.read_bytes == 0
    assert snap.params == 0
    assert all(seconds == 0.0 for seconds in snap.seconds.values())
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_truthy_env_values_enable_profiling(profiler, monkeypatch, value):
    monkeypatch.setenv(profiler.ENV_VAR, value)
    profiler.reset()
    assert profiler.is_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "off", "nope"])
def test_falsey_env_values_keep_profiling_off(profiler, monkeypatch, value):
    monkeypatch.setenv(profiler.ENV_VAR, value)
    profiler.reset()
    assert profiler.is_enabled() is False


def test_buckets_accumulate_and_report_one_line(profiler, monkeypatch, capsys):
    monkeypatch.setenv(profiler.ENV_VAR, "1")
    monkeypatch.setattr(profiler, "current_rank", lambda: 7)
    profiler.reset()
    # One parameter: the read (1 s) is nested in the fetch/dequantize step, so
    # dequantize keeps only its own 3 s out of the 4 s the step took.
    monkeypatch.setattr(
        profiler,
        "perf_counter",
        FakeClock(
            0.0,  # begin_load
            0.0,  # enter dequant
            0.0,  # enter read
            1.0,  # exit read
            4.0,  # exit dequant
            4.0,  # enter map
            9.0,  # exit map
            9.0,  # enter copy
            10.0,  # exit copy
            20.0,  # end_load
        ),
    )

    token = profiler.begin_load()
    assert token is True
    profiler.count_param()
    with profiler.phase("dequant"):
        with profiler.phase("read"):
            profiler.add_read_bytes(100_000_000)
    with profiler.phase("map"):
        pass
    with profiler.phase("copy"):
        pass
    line = profiler.end_load(token)

    snap = profiler.snapshot()
    assert snap.seconds == {"read": 1.0, "dequant": 3.0, "map": 5.0, "copy": 1.0}
    assert snap.total == 20.0
    assert snap.other == 10.0
    assert snap.params == 1
    assert line == (
        "LOAD_PROFILE: rank 7 | 1 params | read 1.0s (5.0%) | dequant 3.0s | map 5.0s"
        " | copy 1.0s | other 10.0s | total 20.0s | 5.0 MB/s"
    )
    assert _SUMMARY_PATTERN.match(line)
    assert capsys.readouterr().out.splitlines() == [line]


def test_summary_reports_read_percentage_of_total(profiler):
    snap = profiler.LoadProfileSnapshot(
        seconds={"read": 2.5, "dequant": 1.5, "map": 3.0, "copy": 3.0},
        total=10.0,
        read_bytes=4_000_000,
        params=10,
    )
    line = profiler.format_summary(snap, rank=3)
    assert line == (
        "LOAD_PROFILE: rank 3 | 10 params | read 2.5s (25.0%) | dequant 1.5s | map 3.0s"
        " | copy 3.0s | other 0.0s | total 10.0s | 0.4 MB/s"
    )
    assert _SUMMARY_PATTERN.match(line)


def test_nested_load_scope_reports_once(profiler, monkeypatch, capsys):
    monkeypatch.setenv(profiler.ENV_VAR, "1")
    profiler.reset()
    monkeypatch.setattr(profiler, "perf_counter", FakeClock(0.0, 4.0))

    with profiler.load_scope():
        assert profiler.begin_load() is True
        profiler.count_param()
        assert profiler.end_load(True) is None

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("LOAD_PROFILE: rank ")
    assert "| 1 params |" in lines[0]
    assert "| total 4.0s |" in lines[0]


def test_unknown_phase_is_rejected_when_enabled(profiler, monkeypatch):
    monkeypatch.setenv(profiler.ENV_VAR, "1")
    profiler.reset()
    with pytest.raises(ValueError, match="unknown load phase"):
        profiler.phase("bogus")


def test_instrumentation_does_not_change_loaded_values(profiler, monkeypatch):
    tensors = {"a": [1.0, 2.0, 3.0], "b": [-4.0, 0.5]}
    names = ["a", "b"]

    monkeypatch.delenv(profiler.ENV_VAR, raising=False)
    profiler.reset()
    plain = run_loop_body(profiler, FakeMapping(tensors), names)
    disabled = profiler.snapshot()

    monkeypatch.setenv(profiler.ENV_VAR, "1")
    profiler.reset()
    with profiler.load_scope():
        instrumented = run_loop_body(profiler, FakeMapping(tensors), names)
    enabled = profiler.snapshot()

    # The same bytes reach the destination with the timer off and on.
    assert instrumented == plain == {"a": [6.0, 4.0, 2.0], "b": [1.0, -8.0]}

    # And the only difference is what the profiler observed.
    assert disabled.params == 0
    assert disabled.read_bytes == 0
    assert disabled.total == 0.0
    assert enabled.params == len(names)
    assert enabled.read_bytes == (3 + 2) * 2
    assert enabled.total > 0.0


def test_state_dict_read_hook_returns_the_same_tensors(profiler, monkeypatch):
    """The state-provider hook only observes: same tensors, bytes counted."""
    torch = pytest.importorskip("torch", reason="the state provider imports torch at module level")
    spec = importlib.util.spec_from_file_location(f"state_under_test_{next(_names)}", _STATE_PATH)
    assert spec is not None and spec.loader is not None
    state = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state)

    tensors = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    expected = tensors["w"].clone()
    source = state.StateDict(tensors)

    monkeypatch.delenv(profiler.ENV_VAR, raising=False)
    profiler.reset()
    monkeypatch.setattr(state, "_LOAD_PROFILE", profiler)
    assert torch.equal(source._load_tensors(["w"])["w"], expected)
    assert profiler.snapshot().read_bytes == 0
    assert profiler.snapshot().seconds["read"] == 0.0

    monkeypatch.setenv(profiler.ENV_VAR, "1")
    profiler.reset()
    monkeypatch.setattr(profiler, "perf_counter", FakeClock(0.0, 1.0))
    loaded = source._load_tensors(["w"])
    assert torch.equal(loaded["w"], expected)
    assert profiler.snapshot().read_bytes == expected.numel() * expected.element_size()
    assert profiler.snapshot().seconds["read"] == 1.0
