"""A processing-stage chain, driven as the host (`processing_stages.md`).

vizi drives a chain of one stage (an applied filter) and no chain of
several; this drives chains of several.
A `Chain` coordinates several single-session `Host`s -- one per plugin
process, each with its own session epoch -- and plays the host's half of
every section of the spec:

- §3: `PlanStage` to each position in order, each answer the next one's
  input; the channel indices of `adds` assigned above the source's channel
  count; §3.1's backward walk for the source channels to load; §3.2's
  `plane_cap` and `capacity`;
- §4: ONE `S` ring granted to every session in the chain under ONE id, minted
  from an epoch no session uses; `STAGED` slots at 64-byte offsets; then
  `StageArm` per position, awaiting its `StageArmAck`;
- §5: the slot header and packed planes, with a codec of its own written from
  §5.1's table (not the SDK's: a host that shared the plugin's codec would
  agree with it by construction);
- §6: the `u64` turn word -- the host fills a slot and stores `position =
  first`; a stage stores `position + 1` or `-position`; a positive position
  no stage is armed at is the host's, which is how it inspects between two
  stages; `D` goes to the stage the slot is at, and the host RELAYS each
  stage's hint to the next;
- §7: the host's deadline, and a slot that misses it quarantined -- never
  reused -- until the turn comes back to a host position.

The algorithm is the document's; `atomicshm` supplies the atomics on the
host's own mapping, as it does for `PulsedLane`.
"""

from __future__ import annotations

import itertools
import math
import mmap
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import (
    PERM_HOST_READ,
    PERM_HOST_WRITE,
    PERM_PLUGIN_READ,
    PERM_PLUGIN_WRITE,
    Dataset,
    Host,
    Segment,
    pb2,
)

__all__ = [
    "Chain",
    "ChainError",
    "Pass",
    "SlotHeader",
    "Step",
    "StageRing",
    "encode_turn",
    "decode_turn",
    "header_bytes",
    "DEADLINE_S",
]

#: §5.1: 256 fixed bytes, then one `u16` per allocated plane, rounded up to 64.
HEADER_FIXED = 256
ALIGN = 64
#: §4: `position` is 1..32 766, `pipeline` 1..65 535, at most 512 slots.
MAX_POSITION = 32766
MAX_PIPELINE = 65535
MAX_SLOTS = 512
#: §4 / §5.1: channel indices are `u16`, so at most 65 536 planes.
MAX_PLANES = 65536
#: §6.3's outcomes.
OUTCOME_NONE, OUTCOME_PROCESSED, OUTCOME_REFUSED, OUTCOME_FAILED = 0, 1, 2, 3
#: §7 leaves the deadline's value to the host; this is the emulator's.
DEADLINE_S = 2.0
#: §6.2: how often the host re-checks turn words on its own schedule, so a
#: lost `D` delays a turn and never loses one.
RECHECK_S = 0.02

#: §5.1 from offset 8 (the turn word at 0 is only ever touched atomically):
#: outcome, reserved, t, done_through, reason, dtype, planes, plane_cap,
#: z/y/x_alloc, reserved, capacity, the extent, origin, pose_r, pose_t,
#: open_generation, 40 reserved bytes.
_FIXED = struct.Struct("<11IIQ6I3d9d3dQ40s")
assert 8 + _FIXED.size == HEADER_FIXED

_ITEMSIZE = {pb2.DTYPE_U8: 1, pb2.DTYPE_U16: 2, pb2.DTYPE_F32: 4, pb2.DTYPE_F64: 8}
_NUMPY = {
    pb2.DTYPE_U8: np.dtype("<u1"),
    pb2.DTYPE_U16: np.dtype("<u2"),
    pb2.DTYPE_F32: np.dtype("<f4"),
    pb2.DTYPE_F64: np.dtype("<f8"),
}
_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _round_up(n: int, to: int = ALIGN) -> int:
    return -(-n // to) * to


def header_bytes(plane_cap: int) -> int:
    """§5.1: where the data begins."""
    return _round_up(HEADER_FIXED + 2 * plane_cap)


def encode_turn(iteration: int, pipeline: int, position: int) -> int:
    """§6.1: bits 0-31 iteration, 32-47 pipeline, 48-63 position SIGNED."""
    if not 0 <= iteration <= 0xFFFFFFFF or not 0 <= pipeline <= 0xFFFF or not -0x8000 <= position <= 0x7FFF:
        raise ValueError(f"turn ({iteration}, {pipeline}, {position}) does not fit §6.1's word")
    return iteration | pipeline << 32 | (position & 0xFFFF) << 48


def decode_turn(word: int) -> tuple[int, int, int]:
    """``(iteration, pipeline, position)``; ``position`` signed."""
    position = word >> 48 & 0xFFFF
    if position >= 0x8000:
        position -= 0x10000
    return word & 0xFFFFFFFF, word >> 32 & 0xFFFF, position


@dataclass
class SlotHeader:
    """§5.1's header, every field, reserved ones included so a check can see
    them."""

    turn: int
    outcome: int
    reserved12: int
    t: int
    done_through: int
    reason: int
    dtype: int
    planes: int
    plane_cap: int
    alloc: tuple[int, int, int]
    reserved52: int
    capacity: int
    extent: tuple[int, int, int, int, int, int]
    origin: tuple[float, float, float]
    pose_r: tuple[float, ...]
    pose_t: tuple[float, float, float]
    open_generation: int
    reserved216: bytes
    channels: tuple[int, ...]

    @property
    def header_bytes(self) -> int:
        return header_bytes(self.plane_cap)

    @property
    def plane_bytes(self) -> int:
        z, y, x = self.alloc
        return z * y * x * _ITEMSIZE.get(self.dtype, 0)

    @property
    def position(self) -> int:
        return decode_turn(self.turn)[2]


def read_header(buf, turn: int) -> SlotHeader:
    """The header of a slot whose turn word was ``turn`` (loaded atomically
    by the caller, BEFORE this read)."""
    f = _FIXED.unpack_from(buf, 8)
    outcome, r12, t, done, reason, dtype, planes, plane_cap, z, y, x = f[0:11]
    plane_cap_read = min(plane_cap, MAX_PLANES, max(0, (len(buf) - HEADER_FIXED) // 2))
    channels = struct.unpack_from(f"<{plane_cap_read}H", buf, HEADER_FIXED) if plane_cap_read else ()
    return SlotHeader(
        turn=turn,
        outcome=outcome,
        reserved12=r12,
        t=t,
        done_through=done,
        reason=reason,
        dtype=dtype,
        planes=planes,
        plane_cap=plane_cap,
        alloc=(z, y, x),
        reserved52=f[11],
        capacity=f[12],
        extent=tuple(f[13:19]),
        origin=tuple(f[19:22]),
        pose_r=tuple(f[22:31]),
        pose_t=tuple(f[31:34]),
        open_generation=f[34],
        reserved216=f[35],
        channels=tuple(channels),
    )


def write_header(buf, h: SlotHeader) -> None:
    """Every field but the turn word, and the whole channel array."""
    _FIXED.pack_into(
        buf,
        8,
        h.outcome,
        h.reserved12,
        h.t,
        h.done_through,
        h.reason,
        h.dtype,
        h.planes,
        h.plane_cap,
        *h.alloc,
        h.reserved52,
        h.capacity,
        *h.extent,
        *h.origin,
        *h.pose_r,
        *h.pose_t,
        h.open_generation,
        h.reserved216,
    )
    channels = list(h.channels) + [0] * (h.plane_cap - len(h.channels))
    struct.pack_into(f"<{h.plane_cap}H", buf, HEADER_FIXED, *channels)


def header_problems(h: SlotHeader, extent: int) -> list[str]:
    """§7's minimum, as the host checks a slot it gets back.  (A stage also
    checks the slot against its arm; the host checks against nothing but the
    format.)"""
    problems = []
    if h.plane_cap > MAX_PLANES:
        problems.append(f"plane_cap {h.plane_cap} > {MAX_PLANES}")
        return problems
    if h.header_bytes + h.capacity > extent:
        problems.append(f"header {h.header_bytes} + capacity {h.capacity} exceeds the region's {extent}")
    if h.dtype not in _ITEMSIZE:
        problems.append(f"dtype {h.dtype} is not a DType")
    if h.planes > h.plane_cap:
        problems.append(f"planes {h.planes} > plane_cap {h.plane_cap}")
    elif h.planes * h.plane_bytes > h.capacity:
        problems.append(f"{h.planes} planes of {h.plane_bytes} bytes exceed capacity {h.capacity}")
    used = h.channels[: h.planes]
    if len(set(used)) != len(used):
        problems.append(f"channels {list(used)} are not distinct")
    z0, zn, y0, yn, x0, xn = h.extent
    z, y, x = h.alloc
    if not (z0 <= zn <= z and y0 <= yn <= y and x0 <= xn <= x):
        problems.append(f"extent {h.extent} is not within the allocation {h.alloc}")
    if not all(math.isfinite(v) for v in (*h.origin, *h.pose_r, *h.pose_t)):
        problems.append("origin or pose is not finite")
    if h.reserved12 or h.reserved52 or any(h.reserved216):
        problems.append("a reserved field is non-zero")
    return problems


class StagedSlot:
    """One `STAGED` region: one slot of the ring (§4)."""

    def __init__(self, index: int, offset: int, extent: int) -> None:
        if offset % ALIGN:
            raise ValueError("§4: each slot's offset is a multiple of 64")
        self.index = index
        self.offset = offset
        self.extent = extent

    def region(self) -> pb2.Region:
        # §4: STAGED, producer HOST; `buffers` and `slot_bytes` zero and
        # `epoch_offset` absent -- the slot names its volume in its header.
        return pb2.Region(
            index=self.index,
            offset=self.offset,
            extent=self.extent,
            discipline=pb2.RegionDiscipline.REGION_DISCIPLINE_STAGED,
            producer=pb2.RegionProducer.REGION_PRODUCER_HOST,
        )


class StageRing(Segment):
    """The `S` segment: ``K`` slots, one mapping granted to every session in
    the chain under one id (§4).  ``shared``: no one `Host` closes it --
    the chain does, once, with `dispose`."""

    shared = True

    def __init__(self, segment: int, slots: int, slot_extent: int, directory: Path) -> None:
        if not 1 <= slots <= MAX_SLOTS:
            raise ValueError(f"§4: a ring has 1..{MAX_SLOTS} slots")
        size = ALIGN + slots * slot_extent
        # §4: read-write for the plugin, since every stage writes; on Linux
        # no future-write seal (`Segment` adds only shrink and grow).
        super().__init__(
            segment,
            "S",
            size,
            directory,
            perms=PERM_PLUGIN_READ | PERM_PLUGIN_WRITE | PERM_HOST_READ | PERM_HOST_WRITE,
        )
        for i in range(slots):
            self.regions.append(StagedSlot(i, ALIGN + i * slot_extent, slot_extent))
        import atomicshm  # the plugin environment's, as `PulsedLane`'s is

        self.map = mmap.mmap(self.fd, size)
        self._view = atomicshm.AtomicView(self.map)
        self._atomics = atomicshm
        self._disposed = False

    def slot(self, index: int) -> StagedSlot:
        return self.regions[index]

    def buf(self, index: int) -> memoryview:
        s = self.regions[index]
        return memoryview(self.map)[s.offset : s.offset + s.extent]

    def load_turn(self, index: int) -> int:
        return int(self._view.load_u64(self.regions[index].offset, self._atomics.ACQUIRE))

    def store_turn(self, index: int, word: int) -> None:
        self._view.store_u64(self.regions[index].offset, word, self._atomics.RELEASE)

    def close(self) -> None:
        """Not a `Host`'s to close: retiring it in one session leaves it
        mapped in the others."""

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._view.close()
        try:
            self.map.close()
        except BufferError:
            pass
        import os

        os.close(self.fd)


class ChainError(AssertionError):
    """The host could not plan, grant or arm the chain as asked."""


@dataclass
class Step:
    """One position in the chain, and what planning and arming made of it."""

    host: Host
    stage_id: str
    position: int
    #: Plain values for the declared keys; missing ones take the declared
    #: default, since the host sends every declared key (§2).
    params: dict
    #: `pb2.Value`s sent AS GIVEN, over ``params`` -- for a scenario that
    #: sends a value of the wrong kind or out of range.  A key mapped to
    #: ``None`` is left out.
    raw_params: dict = field(default_factory=dict)
    input: "pb2.StageShape | None" = None
    planned: "pb2.StagePlanned | None" = None
    output: "pb2.StageShape | None" = None
    new_channels: dict = field(default_factory=dict)
    revision: int = 0

    def declared(self) -> list[dict]:
        for s in self.host.manifest.get("contributions", {}).get("stages", []):
            if s["id"] == self.stage_id:
                return s.get("params", [])
        return []

    def wire_params(self) -> dict:
        """§2: a `Value` for every declared key, its kind fixed by the type."""
        out = {}
        for spec in self.declared():
            key = spec["key"]
            value = self.params.get(key, spec.get("value"))
            kind = spec["type"]
            if kind == "float":
                out[key] = pb2.Value(f=float(value))
            elif kind == "int":
                out[key] = pb2.Value(i=int(value))
            elif kind == "bool":
                out[key] = pb2.Value(b=bool(value))
            else:
                out[key] = pb2.Value(s=str(value))
        for key, value in self.raw_params.items():
            if value is None:
                out.pop(key, None)
            else:
                out[key] = value
        return out

    def needs(self) -> set[int]:
        """Every input channel the accepted plan names."""
        p = self.planned
        found = {m.channel for m in p.modifies} | set(p.reads)
        for m in p.modifies:
            found.update(m.needs)
        for a in p.adds:
            found.update(a.needs)
        return found


@dataclass
class Inspection:
    """What the host saw at a position no stage is armed at (§4)."""

    position: int
    header: SlotHeader
    planes: "np.ndarray | None"


@dataclass
class Pass:
    """One volume's trip through the chain, as the host saw it end.

    ``outcome``: ``"processed"`` (the turn reached the host after the last
    stage), ``"refused"`` or ``"failed"`` (a stage stored ``-k``), or
    ``"deadline"`` (a stage held the turn past the host's deadline; the
    slot is quarantined, §7).  ``planes`` is a copy of the slot's data as
    the header describes it, ``None`` after a failure or a deadline -- the
    data is undefined then (§6.3)."""

    slot: int
    iteration: int
    t: int
    outcome: str
    position: int
    header: "SlotHeader | None"
    planes: "np.ndarray | None"
    #: The turn's positions: each the host stored, and each it saw a stage
    #: store.  A stage that hands on before the host looks is not seen, so
    #: between two consecutive stages the trail may skip a position.
    trail: list[int]
    inspections: list[Inspection]
    elapsed: float
    problems: list[str]
    #: Why the host ended the pass itself, when it did: ``"deadline"``, or
    #: ``"session ended"`` (§7: the holder's session ended mid-turn and the
    #: host took the turn back, stored as ``-k``, and treats it as failed).
    note: str = ""

    def plane(self, channel: int) -> np.ndarray:
        """The plane holding ``channel`` (§5.2: plane p is not channel p)."""
        return self.planes[self.header.channels[: self.header.planes].index(channel)]


@dataclass
class _Ticket:
    slot: int
    iteration: int
    t: int
    deadline: float
    started: float
    position: int
    since: float
    trail: list[int]
    inspections: list[Inspection] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    result: "Pass | None" = None


class Chain:
    """Several plugin sessions, one ring, one pipeline.

    ::

        with Chain(DATASET) as chain:
            a = chain.launch(plugin_dir)
            b = chain.launch(other_plugin_dir)
            chain.add(a, "gain", 1, factor=1.5)
            chain.add(b, "drift", 2)
            chain.plan()
            chain.build_ring(slots=2)
            chain.arm()
            result = chain.run(t=0)
    """

    def __init__(self, dataset: Dataset, *, pipeline: int = 1, deadline: float = DEADLINE_S) -> None:
        if not 1 <= pipeline <= MAX_PIPELINE:
            raise ValueError(f"§4: a pipeline is 1..{MAX_PIPELINE}")
        self.dataset = dataset
        self.pipeline = pipeline
        self.deadline = deadline
        self._tempdir = tempfile.TemporaryDirectory(prefix="vizi-hostsim-chain-")
        self.directory = Path(self._tempdir.name)
        #: Session epochs and the ring's come from one counter, so none alias
        #: (§4, `doorbell_M_memory.md` §3).
        self._epochs = itertools.count(1)
        self.hosts: list[Host] = []
        self.steps: dict[int, Step] = {}
        self.ring: StageRing | None = None
        self.load: tuple[int, ...] = ()
        #: What the chain as planned needs (§3.2) ...
        self.plane_cap = 0
        self.capacity = 0
        #: ... and what the current ring holds, which every header it fills
        #: states (`capacity` is "fixed for the ring", §5.1).  A chain that
        #: outgrows its ring is re-attached by `regrow` (§4).
        self.ring_plane_cap = 0
        self.ring_capacity = 0
        self._revisions: dict[int, int] = {}
        #: Positions the plugin has an arm INSTALLED at (some revision
        #: acknowledged, and not removed): a turn there is the stage's.  Every
        #: other positive position is the host's, including one added to the
        #: chain and not yet armed.  A re-arm leaves its position here -- the
        #: old revision stays installed until the new one is (§4) -- and
        #: `start` separately refuses to hand a slot to a chain any of whose
        #: CURRENT revisions is unacknowledged.
        self.armed: dict[int, Step] = {}
        #: Relay each stage's `D` to the session the slot moved to (§6.2).
        #: ``False`` relays nothing, so a stage must find its turn on its own
        #: schedule; ``relay_delay`` holds each relay back that long.
        self.relay = True
        self.relay_delay = 0.0
        #: ``(from epoch, to epoch, slot, position)`` per relayed hint.
        self.relays: list[tuple[int, int, int, int]] = []
        #: Called ``inspect(position, header, planes)`` at each host position
        #: before the host hands the slot on; may write the slot's planes (the
        #: host holds the turn there).  ``planes`` is a view over the slot.
        self.inspect = None
        self.diagnostics: list[str] = []
        self.quarantined: set[int] = set()
        #: Slots released from quarantine, in order (§7).
        self.unquarantined: list[int] = []
        self._iterations: dict[int, int] = {}
        self._inflight: dict[int, _Ticket] = {}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = False
        self._pump: threading.Thread | None = None
        self._timers: list[threading.Timer] = []

    # -- sessions --

    def launch(self, plugin_dir: Path, *, launch_timeout: "float | None" = None) -> Host:
        """Start one plugin session with its own epoch and wait for `Ready`,
        within the launch budget (§9)."""
        host = Host(plugin_dir, self.dataset, session_epoch=next(self._epochs), launch_timeout=launch_timeout)
        host.on_hint = self._on_hint
        host.__enter__()
        self.hosts.append(host)
        try:
            host.await_ready()
        except RuntimeError as exc:
            # The plugin's failure is named with its directory.
            raise ChainError(f"{plugin_dir}: {exc}") from None
        return host

    def __enter__(self) -> "Chain":
        return self

    def __exit__(self, *_exc) -> None:
        self._stopping = True
        self._wake.set()
        for timer in self._timers:
            timer.cancel()
        if self._pump is not None:
            self._pump.join(5)
        for host in reversed(self.hosts):
            try:
                host.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 -- the others still close
                self.diagnostics.append(f"session {host.session_epoch} did not close cleanly: {exc!r}")
        if self.ring is not None:
            self.ring.dispose()
        self._tempdir.cleanup()

    # -- the chain's shape --

    def add(self, host: Host, stage_id: str, position: int, *, raw_params: dict | None = None, **params) -> Step:
        if not 1 <= position <= MAX_POSITION:
            raise ValueError(f"§4: a position is 1..{MAX_POSITION}")
        step = Step(host, stage_id, position, params, dict(raw_params or {}))
        self.steps[position] = step
        return step

    @property
    def end(self) -> int:
        """The position the host gets the slot back at after the last stage."""
        return max(self.steps) + 1

    @property
    def first(self) -> int:
        """§6.1: the host fills a slot and stores ``position = first``."""
        return min(self.steps)

    def source_shape(self, channels) -> pb2.StageShape:
        _, _, z, y, x = self.dataset.shape_tczyx
        return pb2.StageShape(channels=list(channels), z=z, y=y, x=x, dtype=self.dataset.wire_dtype)

    # -- §3: planning --

    def plan_stage(self, step: Step, inp: pb2.StageShape, timeout: float = 10.0) -> pb2.StagePlanned:
        """One `PlanStage`; an `Error` raises `PluginError`."""
        request = pb2.PlanStage(stage_id=step.stage_id, input=inp, params=step.wire_params())
        return step.host.request(timeout, plan_stage=request).stage_planned

    def _plan_pass(self, load: tuple[int, ...]) -> None:
        sources = self.dataset.shape_tczyx[1]
        next_index = sources  # §4: assigned above the source's channel count
        inp = self.source_shape(load)
        for position in sorted(self.steps):
            step = self.steps[position]
            planned = self.plan_stage(step, inp)
            problems = self._plan_problems(inp, planned)
            if problems:
                raise ChainError(f"stage {step.stage_id!r} at {position}: {'; '.join(problems)}")
            new = {}
            for add in planned.adds:
                new[add.key] = next_index
                next_index += 1
            if next_index - 1 > MAX_PLANES - 1:
                raise ChainError("§4: the chain would need a channel index above 65 535")
            out = pb2.StageShape(
                channels=list(inp.channels) + list(new.values()),
                z=planned.output.z,
                y=planned.output.y,
                x=planned.output.x,
                dtype=planned.output.dtype,
            )
            step.input, step.planned, step.output, step.new_channels = pb2.StageShape(), planned, out, new
            step.input.CopyFrom(inp)
            inp = out

    @staticmethod
    def _plan_problems(inp: pb2.StageShape, planned: pb2.StagePlanned) -> list[str]:
        """§3's rules on a `StagePlanned`, as the host that must size a ring
        from it checks them."""
        problems = []
        out = planned.output
        for axis, have, got in (("z", inp.z, out.z), ("y", inp.y, out.y), ("x", inp.x, out.x)):
            if not 1 <= got <= have:
                problems.append(f"output.{axis} {got} is not in 1..{have}")
        if out.dtype not in _ITEMSIZE:
            problems.append(f"output.dtype {out.dtype} is not a slot dtype")
        present = set(inp.channels)
        named = set(planned.reads) | {m.channel for m in planned.modifies}
        for m in planned.modifies:
            named.update(m.needs)
        for a in planned.adds:
            named.update(a.needs)
        if not named <= present:
            problems.append(f"names channels {sorted(named - present)} the input lacks")
        keys = [a.key for a in planned.adds]
        if any(not k for k in keys) or len(set(keys)) != len(keys):
            problems.append(f"adds keys {keys} are not non-empty and unique")
        return problems

    def plan(self, wants=None) -> None:
        """§3 for the whole chain, then §3.1 and §3.2.

        Planned first with every source channel.  ``wants`` (default: every
        channel the chain ends with) is walked back through the chain (§3.1)
        to the source channels to load; if that is fewer, the chain is
        planned again with the smaller input, which is what it is armed
        with (§4: an arm must stand on an answered plan for its input)."""
        if not self.steps:
            raise ChainError("no stages to plan")
        sources = tuple(range(self.dataset.shape_tczyx[1]))
        self._plan_pass(sources)
        final = self.steps[max(self.steps)].output
        wanted = set(final.channels if wants is None else wants)
        for position in sorted(self.steps, reverse=True):
            step = self.steps[position]
            p = step.planned
            by_mod = {m.channel: set(m.needs) for m in p.modifies}
            by_add = {step.new_channels[a.key]: set(a.needs) for a in p.adds}
            before = set()
            for c in wanted:
                before |= by_add.get(c, by_mod.get(c, {c}))
            # A stage reads whether or not its output is wanted (§3.1).
            wanted = before | set(p.reads)
        load = tuple(c for c in sources if c in wanted)
        if load != sources:
            self._plan_pass(load)
        self.load = load
        # §3.2: the most planes any step holds, the most bytes any step needs.
        _, _, z, y, x = self.dataset.shape_tczyx
        self.plane_cap = max([len(load)] + [len(s.output.channels) for s in self.steps.values()])
        self.capacity = max(
            [len(load) * z * y * x * _ITEMSIZE[self.dataset.wire_dtype]]
            + [len(s.output.channels) * s.output.z * s.output.y * s.output.x * _ITEMSIZE[s.output.dtype] for s in self.steps.values()]
        )
        if self.plane_cap > MAX_PLANES:
            raise ChainError(f"§4: plane_cap {self.plane_cap} > {MAX_PLANES}")

    # -- §4: the ring --

    def slot_extent(self) -> int:
        return _round_up(header_bytes(self.plane_cap) + self.capacity)

    def build_ring(self, slots: int = 2, timeout: float = 10.0) -> StageRing:
        """One `S` segment, granted to every session in the chain under one
        id minted from an epoch no session uses (§4), each grant awaited."""
        if self.ring is not None:
            raise ChainError("the ring is already built; re-attaching is a supersede (§4)")
        ring_id = next(self._epochs) << 32 | 1
        self.ring = StageRing(ring_id, slots, self.slot_extent(), self.directory)
        self.ring_plane_cap, self.ring_capacity = self.plane_cap, self.capacity
        for host in self.sessions():
            host.add(self.ring)
            ack = host.grant(ring_id, timeout)
            if ack.WhichOneof("outcome") != "installed":
                raise ChainError(f"session {host.session_epoch} did not install the ring: {ack}")
        return self.ring

    def sessions(self) -> list[Host]:
        """The sessions with a stage in the chain, each once."""
        seen: list[Host] = []
        for position in sorted(self.steps):
            host = self.steps[position].host
            if host not in seen:
                seen.append(host)
        return seen

    def stage_arm(self, step: Step, **overrides) -> pb2.StageArm:
        """The `StageArm` for ``step``, at its next revision; ``overrides``
        replace fields (a scenario's bad arm)."""
        self._revisions[step.position] = self._revisions.get(step.position, 0) + 1
        step.revision = self._revisions[step.position]
        fields = dict(
            stage_id=step.stage_id,
            revision=step.revision,
            segment_id=self.ring.segment_id,
            pipeline=self.pipeline,
            position=step.position,
            input=step.input,
            output=step.output,
            new_channels=step.new_channels,
            params=step.wire_params(),
        )
        fields.update(overrides)
        return pb2.StageArm(**fields)

    def send_arm(self, step: Step, **overrides) -> pb2.StageArm:
        """Send one `StageArm` and do NOT wait for its acknowledgement."""
        arm = self.stage_arm(step, **overrides)
        step.host.send(pb2.HostMsg(stage_arm=arm))
        return arm

    def await_arm_ack(self, step: Step, revision: int | None = None, timeout: float = 10.0) -> bool:
        """Whether ``step``'s ``revision`` (default its current) is
        acknowledged within ``timeout``."""
        revision = step.revision if revision is None else revision
        key = (self.pipeline, step.position)
        deadline = time.monotonic() + timeout
        while True:
            with step.host._lock:
                got = step.host.stage_arm_acks.get(key)
            if got is not None and got[0] == revision:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def fits(self) -> bool:
        """Whether the chain as planned fits the current ring (§3.2, §4)."""
        return self.plane_cap <= self.ring_plane_cap and self.capacity <= self.ring_capacity

    def arm(self, positions=None, timeout: float = 10.0) -> None:
        """§4: arm each position (all by default) and await each ack --
        the host hands a slot to a position only after it."""
        if not self.fits():
            raise ChainError(
                f"the chain needs {self.plane_cap} planes / {self.capacity} bytes and the ring holds "
                f"{self.ring_plane_cap} / {self.ring_capacity}: regrow() first (§4)"
            )
        for position in sorted(self.steps if positions is None else positions):
            step = self.steps[position]
            self.send_arm(step)
            if not self.await_arm_ack(step, timeout=timeout):
                raise ChainError(f"no StageArmAck for {step.stage_id!r} at {position} revision {step.revision}")
            with self._lock:
                self.armed[position] = step

    def remove(self, position: int) -> Step:
        """Take a position out of the chain.  Its arm is left installed at the
        plugin (inert once nothing hands it a slot); re-plan before re-arming
        the rest."""
        with self._lock:
            self.armed.pop(position, None)
        return self.steps.pop(position)

    @staticmethod
    def session_gone(host: Host) -> bool:
        plugin = host._plugin
        return host.stream_ended.is_set() or (plugin is not None and plugin.poll() is not None)

    def kill_session(self, host: Host, timeout: float = 5.0) -> None:
        """End a session abruptly -- the plugin process killed, as a crash
        ends it.  A turn it holds is taken back by the pump (§7)."""
        host._plugin.kill()
        if not host.stream_ended.wait(timeout):
            raise ChainError(f"session {host.session_epoch}'s stream did not end within {timeout:g}s of the kill")
        self._wake.set()

    def regrow(self, slots: int | None = None, timeout: float = 10.0) -> StageRing:
        """§4, re-attaching a larger buffer: a new `S` ring sized for the
        chain as planned, granted to every session in it SUPERSEDING the old
        (`shared_memory_protocol.md` §9.5, only once every slot's turn is back
        with the host), each awaited with the old ring's `SegmentReleased`;
        a session that held the old ring and has no stage left is sent
        `RetireSegment`; then every position is re-armed on the new id."""
        old = self.ring
        if old is None:
            raise ChainError("no ring to regrow; build_ring first")
        with self._lock:
            if self._inflight or self.quarantined:
                raise ChainError("§9.5: supersede only once every slot's turn is back with the host")
            for index in range(len(old.regions)):
                _, _, position = decode_turn(old.load_turn(index))
                if not self._host_holds(position):
                    raise ChainError(f"slot {index} is at {position}, not the host's")
        # The same ring epoch, the next index: still an epoch no session uses.
        new_id = old.segment_id + 1
        new = StageRing(new_id, len(old.regions) if slots is None else slots, self.slot_extent(), self.directory)
        members = self.sessions()
        for host in self.hosts:
            if old.segment_id not in host.segments or self.session_gone(host):
                continue
            if host in members:
                ack = host.supersede(old.segment_id, new, timeout)
                if ack.WhichOneof("outcome") != "installed":
                    raise ChainError(f"session {host.session_epoch} did not install the larger ring: {ack}")
            else:
                host.retire_segment(old.segment_id, timeout)
        for host in members:
            if new_id not in host.segments:
                host.add(new)
                ack = host.grant(new_id, timeout)
                if ack.WhichOneof("outcome") != "installed":
                    raise ChainError(f"session {host.session_epoch} did not install the larger ring: {ack}")
        with self._lock:
            self.ring = new
            self.ring_plane_cap, self.ring_capacity = self.plane_cap, self.capacity
            self._iterations = {}
        old.dispose()
        self.arm()
        return new

    # -- §5/§6: a volume's trip --

    def _fill(self, index: int, t: int) -> None:
        """Write the source volume for ``t`` into slot ``index`` (§5): the
        loaded channels packed in order, the whole allocation valid, the
        origin at the source's corner, the identity pose."""
        _, _, z, y, x = self.dataset.shape_tczyx
        buf = self.ring.buf(index)
        header = SlotHeader(
            turn=0,
            outcome=OUTCOME_NONE,
            reserved12=0,
            t=t,
            done_through=0,
            reason=0,
            dtype=self.dataset.wire_dtype,
            planes=len(self.load),
            plane_cap=self.ring_plane_cap,
            alloc=(z, y, x),
            reserved52=0,
            capacity=self.ring_capacity,
            extent=(0, z, 0, y, 0, x),
            origin=(0.0, 0.0, 0.0),
            pose_r=_IDENTITY,
            pose_t=(0.0, 0.0, 0.0),
            open_generation=self.dataset.open_generation,
            reserved216=bytes(40),
            channels=self.load,
        )
        write_header(buf, header)
        data = np.stack([self.dataset.volume_at(t, c, 0) for c in self.load]).astype(_NUMPY[self.dataset.wire_dtype])
        start = header.header_bytes
        buf[start : start + data.nbytes] = data.tobytes()
        buf.release()

    def _host_holds(self, position: int) -> bool:
        return position <= 0 or position not in self.armed

    def free_slot(self) -> int | None:
        with self._lock:
            for index in range(len(self.ring.regions)):
                if index in self._inflight or index in self.quarantined:
                    continue
                _, _, position = decode_turn(self.ring.load_turn(index))
                if self._host_holds(position):
                    return index
        return None

    def start(self, t: int, *, slot: int | None = None, deadline: float | None = None) -> _Ticket:
        """Fill a free slot with volume ``t`` and hand it to the first
        position (§6.1).  Returns a ticket for `wait`."""
        if self.ring is None:
            raise ChainError("build the ring first")
        unarmed = sorted(p for p, st in self.steps.items() if not self.await_arm_ack(st, timeout=0))
        if unarmed:
            raise ChainError(f"positions {unarmed} have no acknowledged arm (§4: no slot before StageArmAck)")
        with self._lock:
            index = self.free_slot() if slot is None else slot
            if index is None:
                raise ChainError("no free slot: every slot is in flight or quarantined")
            self._fill(index, t)
            iteration = self._iterations.get(index, 0) + 1 & 0xFFFFFFFF or 1  # §6.1: skip 0
            self._iterations[index] = iteration
            now = time.monotonic()
            ticket = _Ticket(index, iteration, t, self.deadline if deadline is None else deadline, now, self.first, now, [self.first])
            self._inflight[index] = ticket
            self.ring.store_turn(index, encode_turn(iteration, self.pipeline, self.first))
        self._ensure_pump()
        self._hand_to(index, self.first)
        self._wake.set()
        return ticket

    def wait(self, ticket: _Ticket, timeout: float = 30.0) -> Pass:
        if not ticket.done.wait(timeout):
            raise AssertionError(f"slot {ticket.slot} iteration {ticket.iteration} did not finish within {timeout:g}s (trail {ticket.trail})")
        return ticket.result

    def run(self, t: int, **kw) -> Pass:
        timeout = kw.pop("timeout", 30.0)
        return self.wait(self.start(t, **kw), timeout)

    def _hand_to(self, index: int, position: int) -> None:
        """§6.2: host -> plugin `D` to the session armed at ``position``."""
        step = self.armed.get(position)
        if step is not None and not self.session_gone(step.host):
            try:
                step.host.notify(self.ring.segment_id, [index])
            except OSError as exc:
                self.diagnostics.append(f"D to session {step.host.session_epoch} failed: {exc!r}")

    def _on_hint(self, host: Host, hint) -> None:
        """A plugin's `D`: wake the pump, and RELAY it to the session the slot
        is now at (§6.2: plugins never signal each other)."""
        ring = self.ring
        if ring is None or hint.segment != ring.segment_id:
            return
        self._wake.set()
        if not self.relay:
            return
        for index in sorted(hint.regions):
            if index >= len(ring.regions):
                continue
            _, pipeline, position = decode_turn(ring.load_turn(index))
            step = self.armed.get(position) if pipeline == self.pipeline else None
            if step is None:
                continue
            with self._lock:
                self.relays.append((host.session_epoch, step.host.session_epoch, index, position))
            if self.relay_delay:
                timer = threading.Timer(self.relay_delay, self._relay, (step.host, index))
                timer.daemon = True
                with self._lock:
                    self._timers.append(timer)
                timer.start()
            else:
                self._relay(step.host, index)

    def _relay(self, host: Host, index: int) -> None:
        if self._stopping or self.session_gone(host):
            return
        try:
            host.notify(self.ring.segment_id, [index])
        except OSError as exc:  # the session's doorbell is gone
            self.diagnostics.append(f"relay to session {host.session_epoch} failed: {exc!r}")

    def _ensure_pump(self) -> None:
        with self._lock:
            if self._pump is None:
                self._pump = threading.Thread(target=self._pump_loop, daemon=True, name="hostsim-chain")
                self._pump.start()

    def _pump_loop(self) -> None:
        while not self._stopping:
            self._wake.wait(RECHECK_S)
            self._wake.clear()
            try:
                self._pass_once()
            except Exception as exc:  # noqa: BLE001 -- the pump must outlive one bad pass
                self.diagnostics.append(f"chain pump raised {exc!r}")

    def _pass_once(self) -> None:
        with self._lock:
            tickets = list(self._inflight.values())
            quarantined = sorted(self.quarantined)
        for index in quarantined:
            # §7: back in use once the stage hands it on to the host.
            iteration, pipeline, position = decode_turn(self.ring.load_turn(index))
            step = self.armed.get(position)
            if step is not None and self.session_gone(step.host):
                # §7: the holder's session ended; the host takes the turn back.
                self.ring.store_turn(index, encode_turn(iteration, pipeline, -position))
                position = -position
            if self._host_holds(position):
                with self._lock:
                    self.quarantined.discard(index)
                    self.unquarantined.append(index)
        for ticket in tickets:
            self._advance(ticket)

    def _advance(self, ticket: _Ticket) -> None:
        index = ticket.slot
        while True:
            word = self.ring.load_turn(index)
            iteration, pipeline, position = decode_turn(word)
            now = time.monotonic()
            if (iteration, pipeline) != (ticket.iteration, self.pipeline):
                self.diagnostics.append(
                    f"slot {index}: turn word names iteration {iteration} pipeline {pipeline}, "
                    f"not {ticket.iteration} / {self.pipeline} (§6.1: fixed for the whole pass)"
                )
                return self._finish(ticket, "failed", position, header=False)
            if position != ticket.position:
                ticket.position, ticket.since = position, now
                ticket.trail.append(position)
            if position <= 0:
                header = read_header(self.ring.buf(index), word)
                kind = {OUTCOME_REFUSED: "refused", OUTCOME_FAILED: "failed"}.get(header.outcome, "failed")
                if header.outcome not in (OUTCOME_REFUSED, OUTCOME_FAILED):
                    self.diagnostics.append(f"slot {index}: handed back at {position} with outcome {header.outcome}, not refused or failed")
                return self._finish(ticket, kind, position)
            if position in self.armed:
                if self.session_gone(self.armed[position].host):
                    # §7: a stage whose session ends while it holds a turn
                    # loses it: the host takes the turn back, treats it as
                    # failed, and discards the volume (maybe half-written).
                    self.ring.store_turn(index, encode_turn(ticket.iteration, self.pipeline, -position))
                    ticket.trail.append(-position)
                    return self._finish(ticket, "failed", -position, header=False, note="session ended")
                if now - ticket.since > ticket.deadline:
                    # §7: the chain ends for this volume, and the slot is
                    # quarantined -- the stage may still be writing it.
                    with self._lock:
                        self.quarantined.add(index)
                    return self._finish(ticket, "deadline", position, header=False, note="deadline")
                return
            if position >= self.end:
                return self._finish(ticket, "processed", position)
            # A position no stage is armed at is the host's (§4): look, then
            # hand on as a stage would.
            buf = self.ring.buf(index)
            header = read_header(buf, word)
            problems = header_problems(header, len(buf))
            planes = None if problems else self._planes(buf, header, copy=False)
            if self.inspect is not None:
                self.inspect(position, header, planes)
            ticket.inspections.append(Inspection(position, header, None if planes is None else planes.copy()))
            del planes
            buf.release()
            self.ring.store_turn(index, encode_turn(ticket.iteration, self.pipeline, position + 1))
            # Recorded as stored, not as next observed: the stage there may
            # hand on before the host looks again.
            ticket.position, ticket.since = position + 1, time.monotonic()
            ticket.trail.append(position + 1)
            self._hand_to(index, position + 1)

    @staticmethod
    def _planes(buf, header: SlotHeader, copy: bool = True) -> np.ndarray:
        z, y, x = header.alloc
        arr = np.frombuffer(buf, dtype=_NUMPY[header.dtype], count=header.planes * z * y * x, offset=header.header_bytes)
        arr = arr.reshape(header.planes, z, y, x)
        return arr.copy() if copy else arr

    def _finish(self, ticket: _Ticket, outcome: str, position: int, header: bool = True, note: str = "") -> None:
        slot_header, planes, problems = None, None, []
        if header:
            buf = self.ring.buf(ticket.slot)
            slot_header = read_header(buf, self.ring.load_turn(ticket.slot))
            problems = header_problems(slot_header, len(buf))
            if problems:
                self.diagnostics.append(f"slot {ticket.slot} came back with an invalid header: {'; '.join(problems)}")
            elif outcome != "failed":
                planes = self._planes(buf, slot_header)
            buf.release()
        ticket.result = Pass(
            ticket.slot,
            ticket.iteration,
            ticket.t,
            outcome,
            position,
            slot_header,
            planes,
            list(ticket.trail),
            list(ticket.inspections),
            time.monotonic() - ticket.started,
            problems,
            note,
        )
        with self._lock:
            self._inflight.pop(ticket.slot, None)
        ticket.done.set()
