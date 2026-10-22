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

"""Opt-in phase timing for HuggingFace -> Megatron weight loads.

Set ``MINT_LOAD_PROFILE=1`` (or ``true``/``yes``/``on``) to measure where a
checkpoint load spends its wall time and print one greppable line per rank at
the end of the load::

    LOAD_PROFILE: rank 0 | 3000 params | read 12.3s (27.6%) | dequant 4.5s | \
map 20.1s | copy 7.0s | other 0.4s | total 44.3s | 361.2 MB/s

When the variable is unset every entry point here is a strict no-op: no clock
reads, no accumulation, no output. The module deliberately imports nothing
from ``megatron.bridge`` (and imports ``torch`` only when reporting), so it is
safe to import from both the loader and the state provider without changing
their import order.

Accounting
----------
Time is charged to the *innermost* open phase. A phase nested in another one
(the tensor read inside the fetch/dequantize step) therefore contributes its
time to exactly one bucket and is subtracted from its parent, so the buckets
partition the load instead of overlapping:

* ``read``    materializing HF tensor(s) from the state source,
* ``dequant`` fetch/dequantize step, excluding the nested ``read`` time,
* ``map``     the mapping transform (cast/reshape/TP-EP slice),
* ``copy``    the write into the destination parameter (including H2D),
* ``other``   the remainder of the load wall time (computed, not accumulated).

``total`` is the measured load wall time and ``MB/s`` is the summed bytes of
the tensors actually materialized divided by it.
"""

import functools
import os
import threading
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterator, Mapping, Optional, Tuple


#: Environment variable that turns load profiling on.
ENV_VAR = "MINT_LOAD_PROFILE"

#: Values accepted as "on" for :data:`ENV_VAR` (case-insensitive, stripped).
TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

#: Phases whose time is accumulated directly. ``other`` is the remainder of the
#: load and is derived at report time so the buckets never double count.
TIMED_PHASES: Tuple[str, ...] = ("read", "dequant", "map", "copy")

#: Every bucket in the summary line, in report order.
PHASES: Tuple[str, ...] = TIMED_PHASES + ("other",)

# Enabled flag: ``None`` means "not resolved yet", so the environment is read
# once per process (and again after :func:`reset`).
_enabled: Optional[bool] = None

_totals: Dict[str, float] = dict.fromkeys(TIMED_PHASES, 0.0)
_read_bytes = 0
_params = 0
_depth = 0
_start = 0.0
_total: Optional[float] = None

# Nested phase frames are per thread; the load itself is single-threaded.
_frames = threading.local()

# Shared stateless context manager returned while profiling is off.
_NOOP = nullcontext()


@dataclass(frozen=True)
class LoadProfileSnapshot:
    """Immutable view of the current counters, in seconds where applicable."""

    #: Accumulated seconds per :data:`TIMED_PHASES` name.
    seconds: Mapping[str, float]
    #: Measured load wall time (0.0 when no load has closed yet).
    total: float
    #: Bytes of tensors materialized from the checkpoint.
    read_bytes: int
    #: Parameters processed by the per-parameter loop.
    params: int

    @property
    def other(self) -> float:
        """The part of the load no phase accounted for."""
        remainder = self.total - sum(self.seconds[name] for name in TIMED_PHASES)
        return remainder if remainder > 0.0 else 0.0


def is_enabled() -> bool:
    """Whether load profiling is enabled for this process."""
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get(ENV_VAR, "").strip().lower() in TRUTHY_VALUES
    return _enabled


def reset() -> None:
    """Drop every counter and re-read :data:`ENV_VAR` on the next query.

    Callers that toggle the environment at runtime (tests) use this to make the
    gate take effect; a normal process sets the variable before importing.
    """
    global _enabled, _read_bytes, _params, _depth, _start, _total
    _enabled = None
    _read_bytes = 0
    _params = 0
    _depth = 0
    _start = 0.0
    _total = None
    for name in TIMED_PHASES:
        _totals[name] = 0.0
    _frames.stack = []


def _clear_accumulators() -> None:
    """Zero the counters but keep the resolved enabled flag."""
    global _read_bytes, _params, _total
    _read_bytes = 0
    _params = 0
    _total = None
    for name in TIMED_PHASES:
        _totals[name] = 0.0


def _frame_stack() -> list:
    """Per-thread stack of open phases."""
    stack = getattr(_frames, "stack", None)
    if stack is None:
        stack = []
        _frames.stack = stack
    return stack


class _Phase:
    """One timed interval, charged to itself minus the intervals nested in it."""

    __slots__ = ("name", "start", "nested")

    def __init__(self, name: str) -> None:
        self.name = name
        self.start = 0.0
        self.nested = 0.0

    def __enter__(self) -> "_Phase":
        self.start = perf_counter()
        self.nested = 0.0
        _frame_stack().append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = perf_counter() - self.start
        stack = _frame_stack()
        if stack and stack[-1] is self:
            stack.pop()
        elif self in stack:
            # Defensive: a phase out of order must not corrupt the stack.
            stack.remove(self)
        own = elapsed - self.nested
        _totals[self.name] += own if own > 0.0 else 0.0
        if stack:
            stack[-1].nested += elapsed
        return False


def phase(name: str) -> AbstractContextManager[None]:
    """Time one phase of the per-parameter load loop.

    Args:
        name: One of :data:`TIMED_PHASES`.

    Returns:
        A context manager. While profiling is off this is a shared null context
        manager and no clock is read or recorded.
    """
    if not is_enabled():
        return _NOOP
    if name not in TIMED_PHASES:
        raise ValueError(f"unknown load phase {name!r}; expected one of {TIMED_PHASES}")
    return _Phase(name)


def add_read_bytes(nbytes: int) -> None:
    """Count ``nbytes`` of checkpoint payload as read (ignored when off)."""
    global _read_bytes
    if not is_enabled() or nbytes <= 0:
        return
    _read_bytes += int(nbytes)


def add_read_tensors(tensors: Mapping[str, object]) -> None:
    """Count the bytes of materialized ``tensors`` (ignored when off)."""
    if not is_enabled():
        return
    nbytes = 0
    for tensor in tensors.values():
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            nbytes += int(numel()) * int(element_size())
    add_read_bytes(nbytes)


def count_param(n: int = 1) -> None:
    """Count ``n`` parameters processed by the load loop (ignored when off)."""
    global _params
    if not is_enabled():
        return
    _params += int(n)


def begin_load() -> bool:
    """Open a load-timing window.

    Returns:
        A token for :func:`end_load`. ``False`` while profiling is off, in which
        case the paired :func:`end_load` is a no-op too.
    """
    global _depth, _start
    if not is_enabled():
        return False
    if _depth == 0:
        _clear_accumulators()
        _start = perf_counter()
        _frame_stack().clear()
    _depth += 1
    return True


def end_load(token: bool = True) -> Optional[str]:
    """Close a load-timing window, reporting when the outermost one closes."""
    global _depth, _total
    if not token or not is_enabled() or _depth == 0:
        return None
    _depth -= 1
    if _depth > 0:
        # A nested load driver reports through the scope that owns the load.
        return None
    _total = perf_counter() - _start
    return report()


@contextmanager
def load_scope() -> Iterator[None]:
    """Context-manager form of :func:`begin_load` / :func:`end_load`."""
    token = begin_load()
    try:
        yield
    finally:
        end_load(token)


def profiled_load(fn: Callable) -> Callable:
    """Decorate a load driver so one summary line is emitted per load.

    Re-entrant through :func:`begin_load`: a decorated driver called from an
    already open scope neither resets the counters nor reports.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with load_scope():
            return fn(*args, **kwargs)

    return wrapper


def snapshot() -> LoadProfileSnapshot:
    """Capture the current counters."""
    seconds = {name: _totals[name] for name in TIMED_PHASES}
    total = _total if _total is not None else sum(seconds.values())
    return LoadProfileSnapshot(
        seconds=seconds,
        total=total if total > 0.0 else 0.0,
        read_bytes=_read_bytes,
        params=_params,
    )


def current_rank() -> int:
    """Process rank when a distributed job is running, else 0."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # pragma: no cover - torch missing or misbehaving
        pass
    return 0


def _synchronize_device() -> None:
    """Wait for queued device work once, at report time only."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover - torch missing or no usable device
        pass


def format_summary(
    snap: Optional[LoadProfileSnapshot] = None,
    rank: Optional[int] = None,
) -> str:
    """Render the one-line ``LOAD_PROFILE:`` summary for ``snap``.

    Args:
        snap: Counters to render; captured from the live state when omitted.
        rank: Rank to print; resolved from torch.distributed when omitted.
    """
    if snap is None:
        snap = snapshot()
    if rank is None:
        rank = current_rank()
    total = snap.total
    read = snap.seconds["read"]
    share = 100.0 * read / total if total > 0.0 else 0.0
    megabytes_per_second = snap.read_bytes / total / 1e6 if total > 0.0 else 0.0
    dequant = snap.seconds["dequant"]
    mapping = snap.seconds["map"]
    copy = snap.seconds["copy"]
    return (
        f"LOAD_PROFILE: rank {rank} | {snap.params} params"
        f" | read {read:.1f}s ({share:.1f}%)"
        f" | dequant {dequant:.1f}s"
        f" | map {mapping:.1f}s"
        f" | copy {copy:.1f}s"
        f" | other {snap.other:.1f}s"
        f" | total {total:.1f}s"
        f" | {megabytes_per_second:.1f} MB/s"
    )


def report() -> Optional[str]:
    """Synchronize once, print the summary line and return it.

    Returns:
        The printed line, or ``None`` while profiling is off.
    """
    if not is_enabled():
        return None
    _synchronize_device()
    line = format_summary(snapshot())
    print(line, flush=True)
    return line
