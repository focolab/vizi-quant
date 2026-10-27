"""A host emulator for driving a plugin end to end over the real transport.

It launches a plugin exactly as the Broker would -- the manifest's
``entry.command`` in the plugin's own directory, gRPC on a Unix domain socket,
an inherited ``SOCK_SEQPACKET`` doorbell, segments granted over ``SCM_RIGHTS``
-- and plays host: the ``1.0`` handshake, grants in two halves, arrangements
(``ToolBinding``, ``TimeMapping``, ``EpochChange``), ``E`` records, the
plugin's ``D`` hints and the Pulsed lanes they point at, the loans a push
reserves and their release, and teardown.  Each example's ``hostsim.py``
imports this and supplies the dataset and the scenarios.

It is an emulator, not a host.  It implements what the examples under
``examples/`` need and is deliberately permissive about ordering.  Unix only:
the Windows transport (TCP doorbell, port file, session token) has no emulator
here and is the largest untested surface.

**Its encoders and readers are its own**, written from ``docs/spec/`` and
checked at startup against the cross-check corpus four independent SDKs have
graded (``SELF_CHECK``).  Importing ``vizi_plugin_sdk``'s record encoder or
``PulsedReader`` here would make the thing under test grade itself, which
proves self-consistency and nothing else.  What it does import from the SDK is
the protobuf stubs and the protocol version -- both are the contract, not an
implementation of it.
"""

from __future__ import annotations

import array
import collections
import functools
import hashlib
import json
import math
import mmap
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import grpc
import numpy as np

from vizi_plugin_sdk import pb2, pb2_grpc
from vizi_plugin_sdk.manifest import (
    TABLE_MAX_PATCH_BYTES,
    ManifestError,
    _parse_engine_min,
    engines_message,
    parse_manifest,
)
from vizi_plugin_sdk.server import PROTOCOL_MAJOR, PROTOCOL_MINOR
from vizi_plugin_sdk.stages import param_values
from vizi_plugin_sdk import tables as _tables
from vizi_plugin_sdk import roi_pb2

from . import regions
from vizi_plugin_sdk.control import (
    MAX_INLINE_BYTES,
    MAX_MESSAGE_BYTES,
    MAX_REPEATED,
)
from vizi_plugin_sdk.control import _FIELD_BOUNDS as _SDK_FIELD_BOUNDS

__all__ = [
    "Host",
    "ShapeDrag",
    "Dataset",
    "Segment",
    "HostRing",
    "PluginArena",
    "PulsedLane",
    "Voxel",
    "Column",
    "Ray",
    "RayAndPicked",
    "Hint",
    "LiveValue",
    "Loan",
    "Chunk",
    "LabelTable",
    "PluginError",
    "SeriesChoiceRequired",
    "PendingRequest",
    "CANCEL_ANSWER_WINDOW_S",
    "MAX_OPEN_TASKS",
    "MAX_WORK_TEXT_BYTES",
    "TABLE_HELD_BYTES",
    "series_list_problem",
    "reader_opened_problem",
    "ReadResult",
    "Mask",
    "check",
    "failures",
    "check_encoder",
    "encode_record",
    "decode_record",
    "d_payload",
    "decode_d",
    "PROTOCOL",
    "ENV_PROTOCOL_MINOR",
    "host_protocol_minor",
    "ready_refusal",
    "READY_PREFIX",
    "IDENTITY_MAX_BYTES",
    "SHUTDOWN_WINDOW_S",
    "LAUNCH_BUDGET_S",
    "launch_budget",
    "COMMAND_WINDOW_S",
    "INCLUSION_VOXEL_BUDGET",
    "SCRUBBED_ENV_KEYS",
    "scrubbed_environment",
    "RingFull",
    "plugin_folder_name",
    "DOORBELL_WINDOW_MS",
    "APPEARANCE_PALETTES",
    "SESSION_EPOCH",
    "segment_id",
    "VIEW_2D",
    "VIEW_3D",
    "PHASE_PRESS",
    "PHASE_DRAG",
    "PHASE_RELEASE",
    "PHASE_CANCEL",
    "PHASE_FIRE",
    "MODS_SHIFT",
    "MODS_CTRL",
    "MODS_ALT",
    "MODS_META",
]

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

#: §8.7: the emulator's voxel budget for one `ReadRoiInclusion`
#: -- the spec says "the host's voxel budget" and names no number; this is
#: 256 Mi voxels (a u8 deposit of 256 MiB), ``host.inclusion_voxel_budget``.
INCLUSION_VOXEL_BUDGET = 1 << 28

#: The version this host announces by default.  It is the SDK's own rather than
#: a literal because §3.1 makes a mismatch fatal -- the plugin fails the session
#: with a diagnostic naming both versions -- and a literal would go stale
#: silently.  ``Host(..., protocol_minor=N)`` or ``VIZI_HOSTSIM_PROTOCOL_MINOR``
#: presents another minor of the same major, so a plugin can be run against an
#: older or a newer host.
PROTOCOL = (PROTOCOL_MAJOR, PROTOCOL_MINOR)
ENV_PROTOCOL_MINOR = "VIZI_HOSTSIM_PROTOCOL_MINOR"


def host_protocol_minor(minor: "int | None" = None) -> int:
    """The minor this emulator presents: ``minor``, else
    ``VIZI_HOSTSIM_PROTOCOL_MINOR``, else the SDK's own."""
    if minor is None:
        raw = os.environ.get(ENV_PROTOCOL_MINOR, "").strip()
        if not raw:
            return PROTOCOL[1]
        try:
            minor = int(raw)
        except ValueError:
            raise ValueError(f"{ENV_PROTOCOL_MINOR}={raw!r} is not a protocol minor") from None
    if minor < 0:
        raise ValueError(f"protocol minor {minor} is negative")
    return minor


def ready_refusal(ready: "pb2.Ready", hello: "tuple[int, int]") -> "str | None":
    """§3.1 as the host checks a plugin's `Ready` against its own `Hello`:
    the majors match and the plugin's minor does not exceed the host's.  A
    `Ready` below the host's minor is the plugin clamping, and is accepted;
    the session runs at the `Ready`'s minor."""
    if ready.protocol_major != hello[0] or ready.protocol_minor > hello[1]:
        return (
            f"this plugin speaks protocol {ready.protocol_major}.{ready.protocol_minor}; "
            f"this vizi speaks {hello[0]}.{hello[1]}"
        )
    return None
#: §3.1's identity strings: the host's `Hello.id` and the form the plugin's
#: `Ready.id` must take.  The real host sends `Hello from vizi <version>` and
#: fails the session on a `Ready.id` of any other form, so this emulator does
#: both -- the check here is the plugin's conformance test for the rule.
HELLO_ID = "Hello from vizi_hostsim (examples/) for vizi 1.0"
READY_PREFIX = "Ready; I am "
IDENTITY_MAX_BYTES = 2048
#: What a plugin gets to finish after `Shutdown`: §9's session shutdown grace,
#: which is also the `deadline_ms` the host always sends.  No margin: the `1.0`
#: host counts 5 s from its send, records the plugin as having overrun, and
#: kills it, so a plugin that needs 6 s must fail here and
#: not in vizi.
SHUTDOWN_WINDOW_S = 5.0
#: What the `1.0` host gives a plugin to answer `DoorbellShutdown` with its
#: `DoorbellFinalReport` (§9's doorbell grace row).
DOORBELL_WINDOW_MS = 2000
#: §9's launch deadline: ONE budget from spawn to `Ready`,
#: over both the bind and `Ready`, and host policy -- vizi's is 60 s, or the
#: manifest's advisory `entry.startup_s` held to 10-300 s
#: (`manifest_schema.md` §2.3).
LAUNCH_BUDGET_S = 60.0
STARTUP_S_MIN, STARTUP_S_MAX = 10.0, 300.0
#: vizi's override of the whole budget, read from the environment.
ENV_INIT_TIMEOUT = "VIZI_PLUGIN_INIT_TIMEOUT_SECS"


def launch_budget(startup_s: "float | None", timeout: "float | None" = None) -> float:
    """How long a launch may take, spawn to `Ready`: an explicit ``timeout``; else
    ``$VIZI_PLUGIN_INIT_TIMEOUT_SECS``; else the manifest's ``startup_s`` held
    to [10 s, 300 s]; else 60 s."""
    if timeout is not None:
        return float(timeout)
    if os.environ.get(ENV_INIT_TIMEOUT):
        try:
            return float(os.environ[ENV_INIT_TIMEOUT])
        except ValueError:
            return LAUNCH_BUDGET_S
    if startup_s is not None:
        return min(max(float(startup_s), STARTUP_S_MIN), STARTUP_S_MAX)
    return LAUNCH_BUDGET_S
#: :meth:`Host.invoke`'s default wait for `CommandDone`: §8.3's limit for an
#: UNSUPERVISED command.  A command the host shows
#: the person -- every command in vizi's window -- is supervised and has no
#: deadline (§8.16), so this is the scripted host's limit, not vizi's.  A
#: driver that wants vizi's no-deadline wait uses `invoke_async(...)` and its
#: own ``result(timeout=)``.
COMMAND_WINDOW_S = 10.0
#: §9: how long an UNSUPERVISED host waits on a
#: `shape_answer` before reverting the edit; a supervised one waits with no
#: deadline (§8.16).
SHAPE_COMMIT_WINDOW_S = 10.0
#: §8.16: an unanswered `CancelWork` is offered to the person as
#: *Stop plugin* (`Shutdown{USER_STOPPED}`) after this long.
CANCEL_ANSWER_WINDOW_S = 10.0
#: §9: open background tasks a host shows per session, and the
#: UTF-8 bytes of a `WorkProgress` title or text it keeps.
MAX_OPEN_TASKS = 64
MAX_WORK_TEXT_BYTES = 256
#: §8.4 and §9: what the host's transport holds of one `table` widget's ops
#: between two drains, encoded.  A table's ops are held in order, every one
#: of them -- the selection included -- none superseding or displacing
#: another, exempt from the 256 held-whole ops and bounded by these bytes
#: instead; past them the session faults, naming the widget and the op.
TABLE_HELD_BYTES = 64 << 20
#: The size an `image` widget is drawn at until a driver's `resize` says
#: otherwise: ``(width_px, height_px, dpr)``, device-independent pixels and
#: the screen's pixel ratio (§8.4 `WidgetGeometry`).  The emulator's own
#: figure; vizi reports what its layout gives the widget.
DEFAULT_IMAGE_GEOMETRY = (320, 240, 1.0)
#: `HostAppearance` (§8.4): the colors this emulator tells a plugin for each
#: theme, ``0xRRGGBBAA``.  The reference host takes them from its palette
#: (Window, Base, Text, disabled Text, Highlight, Mid); these are a
#: Fusion-like pair, the emulator's own figures.
APPEARANCE_PALETTES: "dict[str, dict[str, int]]" = {
    "dark": {
        "window_bg": 0x353535FF,
        "panel_bg": 0x2A2A2AFF,
        "text": 0xFFFFFFFF,
        "text_disabled": 0x7F7F7FFF,
        "accent": 0x2A82DAFF,
        "border": 0x4A4A4AFF,
    },
    "light": {
        "window_bg": 0xEFEFEFFF,
        "panel_bg": 0xFFFFFFFF,
        "text": 0x000000FF,
        "text_disabled": 0xBEBEBEFF,
        "accent": 0x308CC6FF,
        "border": 0xB8B8B8FF,
    },
}

#: `InstallSegment.segment_id` is *"never 0, never reused within a session"*.
#: The `1.0` host numbers `(session_epoch << 32) | index`, both halves from 1
#: (`doorbell_M_memory.md` §3), so this emulator
#: does the same: an SDK that quietly kept 32 bits of a segment id would pass
#: against 1 and 2 and fail against the first real grant.
SESSION_EPOCH = 1


def segment_id(index: int) -> int:
    """The `1.0` host's segment numbering; ``index`` from 1."""
    if index < 1:
        raise ValueError("segment indices start at 1")
    return (SESSION_EPOCH << 32) | index


#: vizi's two panes: `view_id` 0 and 1, matching
#: the doorbell's VIEW_2D / VIEW_3D.  Host-assigned, not a protocol constant.
VIEW_2D, VIEW_3D = 0, 1

#: The next epoch per pane, for the whole PROCESS (as the host keeps them: one
#: count per host, not per session, never restarting, never 0).  A process
#: running one emulated session sees 1 first, as before; a second
#: `Host` in the same process continues the count.
_PANE_EPOCHS: dict[int, int] = {}
_PANE_EPOCHS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Records, from `doorbell_protocol.md` §4 and the three op documents
# ---------------------------------------------------------------------------

RECORD_SIZE = 112
PAYLOAD_SIZE = 100
RECORD_VERSION = 2
OP_MEMORY = 0x4D  # 'M'
OP_INPUT = 0x45  # 'E'
OP_NOTIFY = 0x44  # 'D'

#: [op u8][ver u8][flags u8][reserved u8][id u32] — flags and reserved MUST be 0.
_HEADER = struct.Struct("<BBBBI")
#: The `M` payload's head: [segment_id u64][size u64][perms u8][pad u8][name_len u8].
_M_HEAD = struct.Struct("<QQBBB")
#: The mapping preamble (`doorbell_M_memory.md` §5), whose 28 checksummed bytes
#: are [segment_id u64][padding u32][identity proof 16 B].  The padding word is
#: reserved and written zero; every region this host lays out starts at byte 32,
#: so it leaves no host-reserved bytes before the first region.
_PREAMBLE = struct.Struct("<QI16s")
#: The `E` payload's fixed head (`doorbell_E_input.md` §1): eight u32, a u16,
#: eight u8, two padding bytes -- 44 bytes, then the 44-byte position union.
_E_HEAD = struct.Struct("<8I H 8B H")  # the trailing u16 is `action`, padding before
#: The `D` payload (`doorbell_D_notification.md` §1): segment u64, a 512-bit
#: region mask, epoch u32, view_id u8, zero padding.
_D_HEAD = struct.Struct("<Q64sIB")
#: §2's pane header: `u32 view_id` then `u32 epoch`, little-endian.
_PANE = struct.Struct("<II")

PERM_PLUGIN_READ = 0x01
PERM_PLUGIN_WRITE = 0x02
PERM_HOST_READ = 0x04
PERM_HOST_WRITE = 0x08

PHASE_PRESS, PHASE_DRAG, PHASE_RELEASE, PHASE_CANCEL, PHASE_FIRE = 0, 1, 2, 3, 4
KIND_NONE, KIND_VOXEL, KIND_PICKED = 0, 1, 2
KIND_COLUMN_YX, KIND_COLUMN_ZX, KIND_COLUMN_ZY = 3, 4, 5
KIND_RAY, KIND_RAY_AND_PICKED = 6, 7
FLAG_MISS, FLAG_ROLL_REF_Y = 0x01, 0x02
MODS_SHIFT, MODS_CTRL, MODS_ALT, MODS_META = 0x01, 0x02, 0x04, 0x08
TARGET_NONE = 0


def encode_record(op: int, record_id: int, payload: bytes) -> bytes:
    """One 112-byte record: 8-byte header, 100-byte payload, CRC over 0..107.

    The CRC is CRC-32/ISO-HDLC over every preceding byte, all 32 bits, LE —
    which is `zlib.crc32` unmodified.
    """
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError(f"a payload is exactly {PAYLOAD_SIZE} bytes, got {len(payload)}")
    body = _HEADER.pack(op, RECORD_VERSION, 0, 0, record_id) + payload
    return body + struct.pack("<I", zlib.crc32(body))


def decode_record(raw: bytes) -> tuple[int, int, bytes]:
    """``(op, id, payload)`` of one record the plugin sent, or a ValueError.

    The host's own framing, from §4: a record is exactly 112 bytes, its CRC
    covers bytes 0..107, `ver` is 2, and `flags`/`reserved` are zero.
    """
    if len(raw) != RECORD_SIZE:
        raise ValueError(f"a record is {RECORD_SIZE} bytes, got {len(raw)}")
    (crc,) = struct.unpack_from("<I", raw, RECORD_SIZE - 4)
    if crc != zlib.crc32(raw[: RECORD_SIZE - 4]):
        raise ValueError("bad CRC")
    op, ver, flags, reserved, record_id = _HEADER.unpack_from(raw, 0)
    if ver != RECORD_VERSION or flags != 0 or reserved != 0:
        raise ValueError(f"header ver={ver} flags={flags} reserved={reserved}")
    return op, record_id, raw[8 : 8 + PAYLOAD_SIZE]


def _header_fault(raw: bytes) -> tuple[str, int] | None:
    """The first header rule a CRC-good record breaks, with its
    `DoorbellFailureClass`, or ``None``: `ver`, then `flags`, then
    `reserved` (§4, and the host's order)."""
    _op, ver, flags, reserved, _id = _HEADER.unpack_from(raw, 0)
    if ver != RECORD_VERSION:
        return f"bad_ver: {ver}", pb2.DOORBELL_FAILURE_BAD_VER
    if flags != 0:
        return f"bad_flags: {flags:#x}", pb2.DOORBELL_FAILURE_BAD_FLAGS
    if reserved != 0:
        return f"bad_reserved: {reserved:#x}", pb2.DOORBELL_FAILURE_BAD_RESERVED
    return None


def m_payload(segment: int, size: int, perms: int, name: bytes = b"") -> bytes:
    """`M`'s payload: the capability and nothing about what it means (§4).

    The name is `name_len` UTF-8 bytes then one NUL, and is empty on unix,
    where the descriptor is the capability.  Everything after the terminator is
    zero padding, which the receiver checks.
    """
    body = _M_HEAD.pack(segment, size, perms, 0, len(name)) + name + b"\0"
    if len(body) > PAYLOAD_SIZE:
        raise ValueError("the name does not fit the payload")
    return body + bytes(PAYLOAD_SIZE - len(body))


def preamble(segment: int, proof: bytes) -> bytes:
    """The first 32 bytes of every granted mapping (`M` §5).

    The proof is host-generated and random, it never appears in the record, and
    the plugin reads it back OUT OF THE MAPPING into its `GrantAck` — which is
    the check that catches a Windows create-or-open landing on a fresh private
    zero-filled section, where every other check passes.
    """
    head = _PREAMBLE.pack(segment, 0, proof)
    return head + struct.pack("<I", zlib.crc32(head))


# --- positions, `doorbell_E_input.md` §3 -----------------------------------


@dataclass(frozen=True)
class Voxel:
    z: float
    y: float
    x: float

    kind = KIND_VOXEL

    def pack(self) -> bytes:
        return struct.pack("<3f", self.z, self.y, self.x)


@dataclass(frozen=True)
class Column:
    """A projection column: ``kind`` names the collapsed axis, ``a, b`` the
    two in-plane coordinates in the order §3's table gives, ``lo, hi`` the
    half-open clip range along the collapsed axis."""

    kind: int
    a: float
    b: float
    lo: float
    hi: float

    def pack(self) -> bytes:
        return struct.pack("<4f", self.a, self.b, self.lo, self.hi)


@dataclass(frozen=True)
class Ray:
    camera: Voxel
    far: Voxel
    entry_frac: float
    roll: float = 0.0

    kind = KIND_RAY

    def pack(self) -> bytes:
        return self.camera.pack() + self.far.pack() + struct.pack("<2f", self.entry_frac, self.roll)


@dataclass(frozen=True)
class RayAndPicked:
    """`E` §3: the ray, the front of the structure the host
    picked, and ``back_frac`` -- how far that structure runs behind its
    front, as a fraction of the ray left there: the back is ``front +
    back_frac * (far - front)``, ``front`` being ``picked`` projected onto the
    ray (not ``camera + back_frac * (far - camera)``).
    0.0 is a pick of no depth; NaN says nothing stood out, which vizi sends
    as a bare `RAY` unless the person opts in."""

    ray: Ray
    picked: Voxel
    back_frac: float = math.nan

    kind = KIND_RAY_AND_PICKED

    def pack(self) -> bytes:
        return self.ray.pack() + self.picked.pack() + struct.pack("<f", self.back_frac)


def e_payload(
    *,
    gen: int,
    t: int,
    interaction: int,
    seq: int,
    epoch: int,
    tool: int,
    channels: int,
    phase: int,
    view_id: int,
    level: int = 0,
    mods: int = 0,
    click_count: int = 1,
    target_kind: int = TARGET_NONE,
    target_id: int = 0,
    target_part: int = 0,
    position=None,
    miss: bool = False,
    roll_ref_y: bool = False,
    action: int = 0,
) -> bytes:
    """`E`'s payload, field by field from `doorbell_E_input.md` §1.

    ``position`` is one of the dataclasses above or ``None`` for `NONE`; the
    union (``[44..92)``, 48 bytes) has its unused tail zero, and
    so are the trailing 8 bytes of fixed padding, which the receiver checks.  ``action`` (``[42..44)``) is 0 on every phase but a
    `FIRE` that names one of the tool's declared actions, 1-based.
    """
    flags = (FLAG_MISS if miss else 0) | (FLAG_ROLL_REF_Y if roll_ref_y else 0)
    kind = KIND_NONE if position is None else position.kind
    head = _E_HEAD.pack(
        gen & 0xFFFFFFFF,
        t,
        interaction,
        seq,
        epoch,
        target_id,
        target_part,
        tool,
        channels,
        phase,
        view_id,
        level,
        mods,
        click_count,
        target_kind,
        kind,
        flags,
        action,
    )
    union = b"" if position is None else position.pack()
    if len(union) > 48:
        raise ValueError("a position is at most 48 bytes")
    return head + union + bytes(48 - len(union)) + bytes(8)


@dataclass(frozen=True)
class Hint:
    """One `D` the plugin sent: which segment, which regions, which pane."""

    record_id: int
    segment: int
    regions: frozenset[int]
    epoch: int
    view_id: int

    @property
    def pane(self) -> tuple[int, int] | None:
        return (self.view_id, self.epoch) if self.epoch else None


def d_payload(segment: int, regions) -> bytes:
    """`D`'s payload the host sends (`doorbell_D_notification.md` §1): the
    segment, the region mask (bit *i* is region *i*, little-endian), and no
    pane pair (epoch 0), zero-padded."""
    bits = 0
    for index in regions:
        if not 0 <= index < 512:
            raise ValueError(f"a D mask names regions 0..511, not {index}")
        bits |= 1 << index
    head = _D_HEAD.pack(segment, bits.to_bytes(64, "little"), 0, 0)
    return head + bytes(PAYLOAD_SIZE - len(head))


def decode_d(record_id: int, payload: bytes) -> Hint:
    """`D`'s payload (`doorbell_D_notification.md` §1).  The pair is present
    iff ``epoch != 0``; a non-zero ``view_id`` beside ``epoch == 0`` is the
    field ignored, never the record."""
    segment, mask, epoch, view_id = _D_HEAD.unpack_from(payload, 0)
    if any(payload[77:]):
        raise ValueError("non-zero padding after the pane pair")
    bits = int.from_bytes(mask, "little")
    regions = frozenset(i for i in range(512) if bits >> i & 1)
    if epoch == 0:
        view_id = 0
    return Hint(record_id, segment, regions, epoch, view_id)


#: Fixture bytes from `scripts/gen_v2_crosscheck.py`'s `memory_grants`
#: scenario, which the Python, Go, Rust and Java SDKs have all graded and
#: agreed on (`polyglot/CROSSCHECK.md`).  They are inputs, not one
#: implementation's output, which is what makes them usable as a check on a
#: fifth, independent encoder — this file's.
SELF_CHECK = (
    # id 1: segment 7, 4096 bytes, PLUGIN_READ, no name.
    (
        1,
        (7, 4096, PERM_PLUGIN_READ, b""),
        "4d020000010000000700000000000000001000000000000001000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000"
        "000000000000ad4a8ee4",
    ),
    # id 6: the name field a stream transport uses, which unix ignores.
    (
        6,
        (7, 4096, PERM_PLUGIN_READ, b"ignored-on-unix"),
        "4d020000060000000700000000000000001000000000000001000f69676e6f726564"
        "2d6f6e2d756e69780000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000"
        "000000000000ff748bd5",
    ),
    # id 7: a host-written grant -- host writes, plugin reads.
    (
        7,
        (7, 1 << 20, PERM_PLUGIN_READ | PERM_HOST_WRITE, b""),
        "4d020000070000000700000000000000000010000000000009000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000"
        "000000000000eb00f02a",
    ),
)


def check_encoder() -> str:
    """Fail the run before it starts if the encoder disagrees with the corpus."""
    for record_id, args, expected in SELF_CHECK:
        got = encode_record(OP_MEMORY, record_id, m_payload(*args)).hex()
        if got != expected:
            raise AssertionError(
                f"record {record_id} encodes as\n  {got}\nand the corpus says\n  {expected}"
            )
    # `doorbell_protocol.md` §4's own self-test for the primitive, before framing.
    if zlib.crc32(b"123456789") != 0xCBF43926:
        raise AssertionError("CRC-32/ISO-HDLC check value is wrong")
    # The `E` and `D` layouts are sized by the document: 44 + 44 + 12 and
    # 8 + 64 + 4 + 1 + 23.
    if _E_HEAD.size != 44 or _D_HEAD.size != 77:
        raise AssertionError("the E or D layout does not add up")
    # Decoding is the inverse of encoding, header for header.
    raw = encode_record(OP_MEMORY, 9, m_payload(3, 64, PERM_PLUGIN_READ))
    assert decode_record(raw)[:2] == (OP_MEMORY, 9)
    return f"{len(SELF_CHECK)} corpus records + the CRC check value"


# ---------------------------------------------------------------------------
# The scenario harness
# ---------------------------------------------------------------------------

failures: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001
        failures.append(label)
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
    else:
        print(f"ok    {label}" + (f": {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Segments: one memfd, a preamble, and the regions the install declares

# `doorbell_M_memory.md` §1: on Linux the host's backing object is a memfd
# sealed shrink+grow -- containment against the plugin resizing the segment
# under the host.  uv's portable CPython builds lack `os.memfd_create` (built
# against a glibc older than 2.27); libc through ctypes is the same call, so
# the emulator grants a sealed memfd on those too.
_MFD_CLOEXEC, _MFD_ALLOW_SEALING = 0x1, 0x2
_F_ADD_SEALS, _F_SEAL_SHRINK, _F_SEAL_GROW = 1033, 2, 4


def _backing_object(name: str, fallback: Path) -> int:
    if sys.platform.startswith("linux"):
        memfd_create = getattr(os, "memfd_create", None)
        if memfd_create is not None:
            return memfd_create(name, _MFD_CLOEXEC | _MFD_ALLOW_SEALING)
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        libc.memfd_create.restype = ctypes.c_int
        fd = libc.memfd_create(name.encode(), _MFD_CLOEXEC | _MFD_ALLOW_SEALING)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "memfd_create")
        return fd
    # macOS: no memfd and no sealing; an unlinked ordinary file maps the same.
    return os.open(str(fallback), os.O_RDWR | os.O_CREAT, 0o600)


def _seal(fd: int) -> None:
    """Shrink+grow seals after sizing, before the descriptor crosses (§1)."""
    if sys.platform.startswith("linux"):
        import fcntl

        fcntl.fcntl(fd, _F_ADD_SEALS, _F_SEAL_SHRINK | _F_SEAL_GROW)
# ---------------------------------------------------------------------------


class RingFull(RuntimeError):
    """A host ring with no room left while every deposit in it is still lent."""


class HostRing:
    """A host-produced LOANED region: the host writes deposits, the plugin
    reads them and sends `ReleaseDeposit`.  `V` and `X` (§3).

    The cursor advances per deposit, as a real allocator would: a plugin that
    assumed offset 0 would pass a single-read test and fail here.  Reclamation
    is FIFO, so the cursor rewinds only when nothing is outstanding — which is
    also what the SDK's own arena does.
    """

    ALIGN = 64  # §7's allocation alignment, carried in `slot_bytes`

    def __init__(self, segment: "Segment", index: int, offset: int, extent: int) -> None:
        self.segment = segment
        self.index = index
        self.offset = offset
        self.extent = extent
        self.cursor = offset
        self.outstanding: set[tuple[int, int]] = set()

    def region(self) -> pb2.Region:
        return pb2.Region(
            index=self.index,
            offset=self.offset,
            extent=self.extent,
            discipline=pb2.RegionDiscipline.REGION_DISCIPLINE_LOANED,
            producer=pb2.RegionProducer.REGION_PRODUCER_HOST,
            # §7: `slot_bytes` is RESERVED on a LOANED region at 1.0 -- the
            # alignment is fixed at 64 by the rule, not carried by the field --
            # so a sender SHOULD send 64 or 0 and no receiver may reject
            # either.  The `1.0` host sends 64.
            slot_bytes=self.ALIGN,
        )

    def fits(self, length: int) -> bool:
        """Whether a deposit of ``length`` would be accepted now, without
        taking it -- the question the host's growth rule asks."""
        padded = (length + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        if self.cursor + padded <= self.offset + self.extent:
            return True
        return not self.outstanding and padded <= self.extent

    def deposit(self, payload: bytes) -> int:
        """Write one deposit and return its MAPPING-relative offset."""
        padded = (len(payload) + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        if self.cursor + padded > self.offset + self.extent:
            if self.outstanding:
                raise RingFull(f"segment {self.segment.segment_id} is full and still lent")
            self.cursor = self.offset
        offset = self.cursor
        self.cursor += padded
        os.pwrite(self.segment.fd, payload, offset)
        self.outstanding.add((offset, len(payload)))
        return offset

    def release(self, offset: int, length: int) -> None:
        """A `ReleaseDeposit` came back.  The cursor is NOT rewound here: it
        rewinds only when the next deposit does not fit and nothing is lent."""
        self.outstanding.discard((offset, length))


class PluginError(Exception):
    """The plugin answered a host request with `Error`."""

    def __init__(self, kind: int, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def series_list_problem(listed: "pb2.SeriesList") -> str | None:
    """What makes a received `SeriesList` malformed, or ``None`` (§8.2):
    no entries; a `shape_tczyx` present but not five
    positive sizes; one id listed twice (the host answers with the id alone).
    A dtype this host does not know is shown as unknown, never refused."""
    if not listed.entries:
        return "the SeriesList has no entries"
    seen: set[str] = set()
    for entry in listed.entries:
        shape = list(entry.shape_tczyx)
        if shape and (len(shape) != 5 or any(n <= 0 for n in shape)):
            return f"series {entry.id!r} has shape_tczyx {shape}, not five positive sizes"
        if entry.id in seen:
            return f"series id {entry.id!r} is listed twice"
        seen.add(entry.id)
    return None


def reader_opened_problem(opened: "pb2.ReaderOpened") -> str | None:
    """What makes a received `ReaderOpened` fail the open,
    or ``None``: `channel_names` longer than the shape's C (it may
    stop short, and ``""`` leaves a channel unnamed)."""
    shape = list(opened.shape_tczyx)
    if len(shape) == 5 and len(opened.channel_names) > shape[1]:
        return f"channel_names names {len(opened.channel_names)} channels of {shape[1]}"
    return None


def _clip_work_text(text: str) -> str:
    """§9: a `WorkProgress` title or text past 256 UTF-8 bytes, cut at a
    character boundary -- as the receiver does again, whatever the sender did."""
    raw = text.encode("utf-8")
    return text if len(raw) <= MAX_WORK_TEXT_BYTES else raw[:MAX_WORK_TEXT_BYTES].decode("utf-8", errors="ignore")


class ShapeDrag:
    """One asked edit (§8.8 *Region types a plugin shapes*): a
    person's drag of one handle of a region whose type the plugin shapes,
    pointer down to release.  Made by :meth:`Host.begin_shape_drag`.

    - :meth:`move` sends `shape_drag` (the pointer projected onto the
      handle's DoF, the host's half; at most one per 40 ms tick);
    - :meth:`release` sends `shape_commit` and waits for `shape_answer`,
      which is the commit -- supervised, with no deadline, unless
      ``supervised=False`` (§9's 10 s, then `REVERTED`);
    - :meth:`cancel` is the person canceling, during the drag or while the
      commit waits (`CANCELLED`).

    ``on_screen`` is the newest valid proposal's positions (or the rest
    form's), ``ended`` the `ShapeEditEnded` the host sent, if any."""

    def __init__(self, host: "Host", drag_id: int, record: "pb2.RoiRecord", handle: int) -> None:
        self.host = host
        self.drag_id = drag_id
        self.region_id = record.region_id
        self.handle = handle
        # Full state in: the rest form as it stood when the drag began.
        self.roi = roi_pb2.Roi()
        self.roi.CopyFrom(record.roi)
        self.contrib_id = record.contrib_id if record.origin == f"plugin:{host.plugin_id}" else ""
        spec = self.roi.vertices[handle].handle
        self.mode = int(spec.mode)
        self.p0 = regions._vec(self.roi.vertices[handle].position)
        self.axis = regions.handle_axis(self.roi, handle)
        self.seq = 0
        self.sent: dict[int, np.ndarray] = {}
        self.on_screen_seq = 0
        self.on_screen = regions.positions(self.roi)
        self.committing = False
        self.answers = 0
        self.ended: "pb2.ShapeEditEnded | None" = None
        self._answered = threading.Event()
        self._answer: "pb2.ShapeResult | None" = None
        self._last_send = 0.0

    def _edit(self, seq: int, p1) -> pb2.ShapeEdit:
        move = roi_pb2.ConstrainMoveRequest(handle=self.handle, mode=self.mode)
        move.roi.CopyFrom(self.roi)
        for name, v in (("p0", self.p0), ("axis", self.axis), ("p1", p1)):
            field = getattr(move, name)
            field.z, field.y, field.x = (float(c) for c in v)
        return pb2.ShapeEdit(region_id=self.region_id, contrib_id=self.contrib_id, type_name=self.roi.type_name, drag_id=self.drag_id, seq=seq, move=move)

    def move(self, p1) -> np.ndarray:
        """The pointer moves to ``p1`` (ZYX, the region's rest coordinates):
        project it onto the handle's DoF and send `shape_drag`.  Returns
        the projected point."""
        if self.committing or self.ended is not None:
            raise RuntimeError(f"drag {self.drag_id} is no longer being dragged")
        p = regions.project_onto_dof(self.mode, self.p0, self.axis, p1)
        wait = self._last_send + 0.04 - time.monotonic()
        if wait > 0:
            time.sleep(wait)  # the host's pump tick: at most one per 40 ms
        self._last_send = time.monotonic()
        with self.host._lock:
            self.seq += 1
            self.sent[self.seq] = p
            seq = self.seq
        self.host.send(pb2.HostMsg(shape_drag=self._edit(seq, p)))
        return p

    def await_proposal(self, seq: "int | None" = None, timeout: float = 10.0) -> np.ndarray:
        """Wait until the shape on screen answers ``seq`` (default: the
        latest sent) or a newer one; return it."""
        want = self.seq if seq is None else seq
        deadline = time.monotonic() + timeout
        while self.on_screen_seq < want:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no shape_proposal for drag {self.drag_id} seq >= {want} within {timeout} s (on screen: {self.on_screen_seq})")
            time.sleep(0.01)
        return self.on_screen

    def release(self, p1=None, *, supervised: bool = True, timeout: "float | None" = None) -> pb2.ShapeEditEnded:
        """Release (after one last :meth:`move` to ``p1`` if given): send
        `shape_commit` and wait for the plugin's `shape_answer`.  Supervised
        (vizi's activity-list row), there is no deadline: ``timeout`` is
        only the driver's own bound.  Unsupervised, §9's 10 s, then
        `REVERTED`.  Returns the `ShapeEditEnded` sent."""
        if p1 is not None and not self.committing:
            self.move(p1)
        return self._commit(self.sent.get(self.seq, self.p0) if self.seq else self.p0, supervised, timeout)

    def _commit(self, p1, supervised: bool, timeout: "float | None") -> pb2.ShapeEditEnded:
        with self.host._lock:
            self.committing = True
        self.host.send(pb2.HostMsg(shape_commit=self._edit(0, p1)))
        limit = (timeout if timeout is not None else 120.0) if supervised else (timeout if timeout is not None else SHAPE_COMMIT_WINDOW_S)
        if not self._answered.wait(limit):
            if supervised:
                raise TimeoutError(f"no shape_answer for drag {self.drag_id} within the driver's {limit} s (supervised: the host would still be waiting)")
            return self.host._end_drag(self, pb2.ShapeEditEnded.REVERTED, f"no shape_answer within {limit} s (unsupervised)")
        return self.ended

    def send_late_drag(self, p1=None) -> None:
        """A `shape_drag` that arrives after this drag's commit or end --
        what the wire's per-queue order allows (§5.3), and which a plugin
        drops silently (§8.8).  Any answer to it is dropped here too."""
        p = regions.project_onto_dof(self.mode, self.p0, self.axis, self.p0 if p1 is None else p1)
        with self.host._lock:
            self.seq += 1
            self.sent[self.seq] = p
            seq = self.seq
        self.host.send(pb2.HostMsg(shape_drag=self._edit(seq, p)))

    def cancel(self) -> pb2.ShapeEditEnded:
        """The person cancels, during the drag or while the commit waits."""
        return self.host._end_drag(self, pb2.ShapeEditEnded.CANCELLED, "canceled by the person")


class PendingRequest:
    """A host request in flight (:meth:`Host.request_async`): its ``id`` --
    the subject of its `WorkProgress` and of a `CancelWork` for it -- and
    :meth:`result`, which waits for the reply and raises `PluginError` for an
    `Error` (``CANCELLED`` included, which is no fault)."""

    def __init__(self, request_id: str, waiter: "queue.SimpleQueue") -> None:
        self.id = request_id
        self._waiter = waiter

    def result(self, timeout: float = 20.0) -> "pb2.PluginMsg":
        reply = self._waiter.get(timeout=timeout)
        if reply.WhichOneof("msg") == "error":
            raise PluginError(reply.error.kind, reply.error.message)
        return reply


class PendingChunk:
    """An `ExportChunk` in flight (:meth:`Host.export_chunk_async`): ``id``
    for a `CancelWork`, and :meth:`result`, which returns `ChunkWritten` or,
    after the host's `AbortExport`, raises `PluginError`."""

    def __init__(self, host: "Host", handle: str, pending: PendingRequest, release) -> None:
        self.id = pending.id
        self._host = host
        self._handle = handle
        self._pending = pending
        self._release = release

    def result(self, timeout: float = 120.0) -> pb2.ChunkWritten:
        try:
            written = self._pending.result(timeout).chunk_written
        except PluginError as exc:
            self._host._abort_ended_export(self._handle, exc)
            raise
        if self._release is not None:
            ring, offset, length = self._release
            ring.release(offset, length)
        return written


class SeriesChoiceRequired(Exception):
    """The reader answered `OpenReader` with `SeriesList`
    (`reader_writer_contract.md` §1.2a): the file holds several images and
    `reader_writer_contract.md` §1.2a): the file holds several images and
    nothing was opened.  vizi raises its own `SeriesChoiceRequired` out of the
    open and asks the person; a driver is the person here, and answers with
    :meth:`choose`, which repeats the open naming the id -- on the same
    session, so a plugin holding its listing instance opens it there."""

    def __init__(self, host: "Host", uri: str, reader_id: str, listed: "pb2.SeriesList") -> None:
        super().__init__(f"{uri} holds {len(listed.entries)} images{' (truncated)' if listed.truncated else ''}; choose one")
        self.host = host
        self.uri = uri
        self.reader_id = reader_id
        self.listed = listed

    @property
    def entries(self):
        return list(self.listed.entries)

    @property
    def truncated(self) -> bool:
        return bool(self.listed.truncated)

    def choose(self, series: str, timeout: float = 30.0) -> "pb2.ReaderOpened":
        return self.host.open_reader(self.uri, self.reader_id, timeout, series=str(series))


@dataclass(frozen=True)
class ReadResult:
    """A reader's reply: the descriptor, and the bytes read out of the `G`
    arena before they were released."""

    ref: pb2.SegmentRef
    payload: bytes

    @property
    def array(self) -> np.ndarray:
        dtype = {pb2.DTYPE_U8: np.uint8, pb2.DTYPE_U16: np.uint16, pb2.DTYPE_F32: np.float32}[self.ref.dtype]
        return np.frombuffer(self.payload, dtype=dtype).reshape(tuple(self.ref.shape_zyx))


@dataclass
class Mask:
    """One native mask annotation the host compiled from a label volume."""

    region_id: str
    label: int
    name: str
    color: tuple[int, int, int, int]
    origin_zyx: tuple[int, int, int]
    membership: np.ndarray  # bbox-local, bool


@dataclass(frozen=True)
class Loan:
    """One reservation a plugin push named, as the host disposed of it.
    ``uses`` is the sender's declaration (§5.6): ``0`` and ``1`` both mean
    one; the host releases the reservation once, when that many messages have
    been disposed of."""

    segment: int
    region: int
    offset: int
    length: int
    payload: bytes
    uses: int = 1


class PluginArena:
    """A plugin-produced LOANED region: the plugin reserves, a push names the
    bytes, and the host sends `ReleaseLoan` once it has disposed of every
    declared use.  `G`, `P`, and an `O` segment's region 1 (§3, §5.6)."""

    ALIGN = 64

    def __init__(self, segment: "Segment", index: int, offset: int, extent: int) -> None:
        self.segment = segment
        self.index = index
        self.offset = offset
        self.extent = extent

    def region(self) -> pb2.Region:
        return pb2.Region(
            index=self.index,
            offset=self.offset,
            extent=self.extent,
            discipline=pb2.RegionDiscipline.REGION_DISCIPLINE_LOANED,
            producer=pb2.RegionProducer.REGION_PRODUCER_PLUGIN,
            slot_bytes=self.ALIGN,
        )

    def read(self, offset: int, length: int) -> bytes:
        """The bytes a descriptor names, bounds-checked against the region
        before a single one is reinterpreted (§11)."""
        if length <= 0 or offset < self.offset or offset + length > self.offset + self.extent:
            raise ValueError(
                f"[{offset}, {offset + length}) escapes region {self.index} "
                f"[{self.offset}, {self.offset + self.extent})"
            )
        return os.pread(self.segment.fd, length, offset)


class PulsedLane:
    """A plugin-published PULSED region, read per `shared_memory_protocol.md`
    §6.5 through process-shared atomics on the host's own mapping.

    `atomicshm` is the same library the SDK uses for its atomics, and that is
    not the SDK grading itself: the library supplies `load`/`CAS`/`store` on a
    mapping and nothing about this protocol.  The algorithm here is the
    document's.
    """

    STATE_WRITE = 1 << 62
    STATE_READ = 1 << 63
    REVISION_MASK = (1 << 62) - 1

    def __init__(
        self,
        segment: "Segment",
        index: int,
        offset: int,
        buffers: int,
        slot_bytes: int,
        pane_header: bool,
    ) -> None:
        if offset % 8:
            raise ValueError("a PULSED region's offset is u64-aligned")
        self.segment = segment
        self.index = index
        self.offset = offset
        self.buffers = buffers
        self.slot_bytes = slot_bytes
        self.epoch_offset = slot_bytes - 8 if pane_header else None
        self.extent = buffers * 16 + buffers * slot_bytes
        self.last_seen = 0
        self.quarantined: str | None = None
        self.stale_pairs = 0
        self._atomics = None
        self._mapping = None

    def region(self) -> pb2.Region:
        region = pb2.Region(
            index=self.index,
            offset=self.offset,
            extent=self.extent,
            discipline=pb2.RegionDiscipline.REGION_DISCIPLINE_PULSED,
            producer=pb2.RegionProducer.REGION_PRODUCER_PLUGIN,
            buffers=self.buffers,
            slot_bytes=self.slot_bytes,
        )
        if self.epoch_offset is not None:
            region.epoch_offset = self.epoch_offset
        return region

    @property
    def capacity(self) -> int:
        """§3.2: the payload's capacity, less the pane header where declared."""
        return self.slot_bytes - (8 if self.epoch_offset is not None else 0)

    def _open(self):
        if self._atomics is None:
            import atomicshm  # the plugin environment's, as the SDK's is

            self._mapping = mmap.mmap(self.segment.fd, self.segment.size)
            self._view = atomicshm.AtomicView(self._mapping)
            self._atomics = atomicshm
        return self._view

    def close(self) -> None:
        if self._atomics is not None:
            self._view.close()
            self._mapping.close()
            self._atomics = None

    def _state_at(self, i: int) -> int:
        return self.offset + i * 16

    def _payload_at(self, i: int) -> int:
        return self.offset + self.buffers * 16 + i * self.slot_bytes

    def read_newest(self) -> tuple[int, bytes, tuple[int, int] | None] | None:
        """§6.5: ``(revision, payload, pane)`` of the newest READY buffer past
        ``last_seen``, consumed; ``None`` when there is nothing new.

        A quarantined region reads as nothing new, forever, and says why in
        ``quarantined`` -- §6's answer to an impossible state word, an invalid
        published length, or a failed release CAS, all of which are proof of
        a broken peer rather than of a race.
        """
        if self.quarantined is not None:
            return None
        view = self._open()
        acquire, acq_rel = self._atomics.ACQUIRE, self._atomics.ACQ_REL
        for _attempt in range(8):
            best_i, best_rev = -1, self.last_seen
            for i in range(self.buffers):
                word = int(view.load_u64(self._state_at(i), acquire))
                revision = word & self.REVISION_MASK
                flags = word & ~self.REVISION_MASK
                if word == 0:
                    continue
                if revision == 0 or flags == (self.STATE_WRITE | self.STATE_READ):
                    self.quarantined = f"impossible state word {word:#x} in buffer {i}"
                    return None
                if flags == 0 and revision > best_rev:
                    best_i, best_rev = i, revision
            if best_i < 0:
                return None
            claimed = best_rev | self.STATE_READ
            observed = int(
                view.cas_u64(self._state_at(best_i), best_rev, claimed, acq_rel, acquire)
            )
            if observed != best_rev:
                continue  # a writer moved; rescan
            (length,) = struct.unpack_from("<Q", self._mapping, self._state_at(best_i) + 8)
            base = self._payload_at(best_i)
            if not 0 < length <= self.capacity or base + length > self.segment.size:
                view.cas_u64(self._state_at(best_i), claimed, best_rev, acq_rel, acquire)
                self.quarantined = f"published length {length} outside (0, {self.capacity}]"
                return None
            payload = bytes(self._mapping[base : base + length])
            pane = None
            if self.epoch_offset is not None:
                view_id, epoch = _PANE.unpack_from(self._mapping, base + self.epoch_offset)
                # §2: the pair is rejected -- and only the pair -- when
                # `view_id` does not fit the u8 the doorbell carries, or the
                # epoch is 0, which is the cleared state and no pane's.
                if view_id <= 0xFF and epoch != 0:
                    pane = (view_id, epoch)
            released = int(view.cas_u64(self._state_at(best_i), claimed, 0, acq_rel, acquire))
            if released != claimed:
                self.quarantined = f"release CAS on buffer {best_i} found {released:#x}"
                return None
            self.last_seen = best_rev
            return best_rev, payload, pane
        return None


class Segment:
    """The host's side of one granted segment: the object, its preamble, its
    region table, and the `M`/`InstallSegment` halves that publish it."""

    def __init__(
        self,
        segment: int,
        purpose: str,
        size: int,
        directory: Path,
        *,
        subject: str = "",
        perms: int,
    ) -> None:
        self.segment_id = segment
        self.purpose = purpose
        self.subject = subject
        self.size = size
        self.perms = perms
        self.proof = os.urandom(16)
        self.fd = _backing_object(f"vizi-hostsim-{segment}", directory / f"seg{segment}.bin")
        os.ftruncate(self.fd, size)
        os.pwrite(self.fd, preamble(segment, self.proof), 0)
        _seal(self.fd)
        self.regions: list = []
        #: `InstallSegment.supersedes` (`shared_memory_protocol.md` §9.5): the
        #: id this grant retires WHEN IT IS INSTALLED, 0 for none.  Set by
        #: `Host.supersede`, which is how a host-sized arena follows the
        #: dataset (`control_plane_protocol.md` §5.7).
        self.supersedes = 0
        #: The exact 112 bytes of this grant's `M`, id and CRC included, once
        #: sent -- what `Host.resend_capability` sends again, because a
        #: re-send carries the SAME doorbell id (`doorbell_M_memory.md` §7).
        self.m_record: bytes | None = None

    # -- region tables per purpose, as the `1.0` host lays them out --

    @classmethod
    def host_ring(cls, segment: int, purpose: str, size: int, directory: Path) -> "Segment":
        """`V` / `X`: one host-produced LOANED ring from byte 32."""
        self = cls(
            segment,
            purpose,
            size,
            directory,
            perms=PERM_PLUGIN_READ | PERM_HOST_READ | PERM_HOST_WRITE,
        )
        self.regions.append(HostRing(self, 0, 32, size - 32))
        return self

    @staticmethod
    def arena_segment_size(payload_bytes: int) -> int:
        """`reader_writer_contract.md` §0: the 32-byte preamble plus one
        payload padded to the 64-byte slot -- the size the reference host
        grants a `G` or `X` arena at (`grants.arena_segment_size`)."""
        return 32 + max(64, -(-payload_bytes // 64) * 64)

    @classmethod
    def reader_arena(cls, segment: int, opened: "pb2.ReaderOpened", directory: Path) -> "Segment":
        """The `G` grant `reader_writer_contract.md` §1.1 sends AFTER
        `ReaderOpened`: one level-0 volume of the declared shape and dtype,
        which is why it cannot precede the open."""
        _, _, z, y, x = opened.shape_tczyx
        width = {pb2.DTYPE_U8: 1, pb2.DTYPE_U16: 2, pb2.DTYPE_F32: 4}[opened.dtype]
        return cls.plugin_arena(segment, "G", cls.arena_segment_size(z * y * x * width), directory)

    @classmethod
    def plugin_arena(
        cls, segment: int, purpose: str, size: int, directory: Path, *, subject: str = ""
    ) -> "Segment":
        """`G` / `P`: one plugin-produced LOANED arena from byte 32, and for
        `P` the panel id as `subject`."""
        self = cls(
            segment,
            purpose,
            size,
            directory,
            subject=subject,
            perms=PERM_PLUGIN_READ | PERM_PLUGIN_WRITE | PERM_HOST_READ,
        )
        self.regions.append(PluginArena(self, 0, 32, size - 32))
        return self

    @classmethod
    def overlay(
        cls,
        segment: int,
        subject: str,
        size: int,
        directory: Path,
        *,
        slot_bytes: int = 64 << 10,
        buffers: int = 3,
    ) -> "Segment":
        """`O`: region 0 a plugin-published PULSED lane with a pane header,
        region 1 a plugin-produced LOANED arena; `subject` = the overlay id."""
        self = cls(
            segment,
            "O",
            size,
            directory,
            subject=subject,
            perms=PERM_PLUGIN_READ | PERM_PLUGIN_WRITE | PERM_HOST_READ,
        )
        lane = PulsedLane(self, 0, 32, buffers, slot_bytes, pane_header=True)
        arena_offset = (32 + lane.extent + 63) // 64 * 64
        if arena_offset + 64 > size:
            raise ValueError("the overlay segment is too small for a lane and an arena")
        self.regions.append(lane)
        self.regions.append(PluginArena(self, 1, arena_offset, size - arena_offset))
        return self

    @staticmethod
    def chunk_arena_bytes(dataset: "Dataset") -> int:
        """`control_plane_protocol.md` §5.7 rule 4 / §8.12: a CHUNKED
        overlay's arena is sized by the host from the open dataset -- one
        level-0 volume of u16 plus one at the first coarser level (§8.11's
        ``max(1, floor(n / 2))`` per axis, whether or not the dataset
        declares that level) -- each padded to the 64-byte reservation
        alignment.  Not ``segment_mib``, which the manifest may not carry on
        a chunked overlay, and never measured against §9's growth ceiling."""
        _, _, z, y, x = dataset.shape_tczyx
        level0 = 2 * z * y * x
        level1 = 2 * max(1, z // 2) * max(1, y // 2) * max(1, x // 2)
        return sum(-(-n // 64) * 64 for n in (level0, level1))

    @classmethod
    def chunk_arena(
        cls,
        segment: int,
        subject: str,
        dataset: "Dataset",
        directory: Path,
        *,
        arena_bytes: int | None = None,
        slot_bytes: int = 64 << 10,
        buffers: int = 3,
    ) -> "Segment":
        """The `O` grant of a chunked overlay (§5.7): the ordinary overlay
        table -- a lane, which goes unused, and the arena -- with the arena
        HOST-SIZED from ``dataset`` (`chunk_arena_bytes`), or ``arena_bytes``
        when a scenario wants a particular size.  Granted when a dataset is
        open, not at session start with the manifest's other overlays, and
        re-granted through `Host.supersede` when the dataset changes."""
        arena = cls.chunk_arena_bytes(dataset) if arena_bytes is None else int(arena_bytes)
        lane_extent = buffers * 16 + buffers * slot_bytes
        arena_offset = (32 + lane_extent + 63) // 64 * 64
        self = cls.overlay(segment, subject, arena_offset + arena, directory, slot_bytes=slot_bytes, buffers=buffers)
        self.host_sized = True
        return self

    def close(self) -> None:
        for region in self.regions:
            if isinstance(region, PulsedLane):
                region.close()
        os.close(self.fd)

    def install(self) -> pb2.InstallSegment:
        return pb2.InstallSegment(
            segment_id=self.segment_id,
            purpose=self.purpose,
            subject=self.subject,
            supersedes=self.supersedes,
            regions=[region.region() for region in self.regions],
        )

    def region_at(self, offset: int):
        """The region a mapping-relative offset falls in, or ``None``."""
        for region in self.regions:
            if region.offset <= offset < region.offset + region.extent:
                return region
        return None

    @property
    def ring(self) -> HostRing:
        return next(r for r in self.regions if isinstance(r, HostRing))

    @property
    def arena(self) -> PluginArena:
        return next(r for r in self.regions if isinstance(r, PluginArena))

    @property
    def lane(self) -> PulsedLane:
        return next(r for r in self.regions if isinstance(r, PulsedLane))


# ---------------------------------------------------------------------------
# The dataset the emulated host serves
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """What the host answers `GetViewState` and `ReadDataVolume` from, and
    what its arrangements describe.  ``volume(t, c)`` returns a contiguous
    ``(Z, Y, X)`` array of ``np_dtype``."""

    shape_tczyx: tuple[int, int, int, int, int]
    np_dtype: np.dtype
    wire_dtype: int
    open_generation: int
    volume: object  # callable (t, c) -> np.ndarray
    t: int = 0
    z: int = 0
    c: int = 0
    level: int = 0
    crop_min_zyx: tuple[int, int, int] | None = None
    crop_max_zyx: tuple[int, int, int] | None = None
    mip: bool = False
    dataset_uri: str = "hostsim://dataset"
    dataset_name: str = "hostsim"
    #: Seconds per frame.  ``0`` means the dataset has no time increment,
    #: and then -- as in vizi -- there is NO `TimeMapping`: `arrange()` skips
    #: it and every epoch names revision 0.
    time_increment_s: float = 1.0
    #: The names the FILE gives its channels, by index: ``""``
    #: for one it does not name, and it may stop short of C.  A channel with
    #: no name is shown as vizi shows it, ``"Channel N"`` (1-based), with
    #: `CHANNEL_NAME_SOURCE_PLACEHOLDER`; see :meth:`channel_infos`.
    channel_names: tuple[str, ...] = ()
    #: `ViewState.visible_channels` (§8.7): the set enabled for display,
    #: ascending; ``None`` means every channel.  ``c`` is deprecated and
    #: always 0 on the host; this emulator sends whatever a driver
    #: sets, and the reference drivers leave it at the host's value.
    visible_channels: tuple[int, ...] | None = None
    #: `ViewState.selected_region_id`: the region the person
    #: has selected, ``""`` for none.  A driver sets it; `ViewChanged` follows.
    selected_region_id: str = ""
    #: The applied filter, or ``None``: served in
    #: `DataVolume.filter`, `ViewState.filter` and `ViewChanged.filter`.
    applied_filter: "pb2.AppliedFilter | None" = None
    #: What the shared filter does to a volume, ``(volume, t, c) -> volume``,
    #: for a read WITHOUT `raw` while ``applied_filter`` is set ("Plugins See
    #: Filtered Data"); ``None`` serves the source under the
    #: filter's name.  A `raw` read is always the source.
    filter_fn: "Callable[[np.ndarray, int, int], np.ndarray] | None" = None

    def pitch(self) -> str:
        """Which of the three cases the pitch is: ``"all"`` axes recorded,
        ``"none"``, or ``"some"``."""
        axes = self.spacing_zyx or (None, None, None)
        known = sum(v is not None for v in axes)
        return "all" if known == 3 else "none" if known == 0 else "some"

    #: `ViewState.dataset_series`: which image of ``dataset_uri`` is
    #: open, as its reader's `SeriesEntry.id`; ``None`` (the field ABSENT)
    #: for a file of one -- ``""`` is an id a reader may use.
    dataset_series: str | None = None
    #: The pyramid: how many levels the dataset has.  Level ``n`` is level 0
    #: block-downsampled by ``2**n`` per axis the way the host does it -- the
    #: mean of each block, an odd trailing sample discarded --
    #: so the SAMPLING FACTOR of level ``n`` is exactly ``2**n`` while the
    #: shape ratio of an odd axis is not (`control_plane_protocol.md` §8.11:
    #: `Geometry.levels` carries the factor).  One level means level 0 only.
    levels: int = 1
    #: Physical voxel size ZYX and its unit, for `Geometry.spacing`; ``None``
    #: means the host does not know (`SPACING_PROVENANCE_DISPLAY_FALLBACK`,
    #: unit empty), which is the `1.0` host's answer
    #: with no calibration.  An axis may be ``None`` alone: a
    #: PARTLY recorded file (for example XY 0.3208 µm, Z
    #: unrecorded), which `ViewState` carries per axis with NaN for the
    #: unrecorded one and `FILE`, and `Geometry` as the display fallback.
    spacing_zyx: "tuple[float | None, float | None, float | None] | None" = None
    spacing_unit: str = "um"

    @property
    def gen(self) -> int:
        return self.open_generation

    def level_scale(self, level: int) -> int:
        """The nominal factor of ``level``, ``2**level`` -- what every axis
        that is still folding has.  The factor the host DECLARES per axis is
        `axis_scale`, which stops when the axis does."""
        if not 0 <= level < self.levels:
            raise IndexError(f"level {level} is not one of {self.levels}")
        return 2**level

    def axis_scale(self, level: int) -> tuple[int, int, int]:
        """`Geometry.levels[level]` per axis: ``2**min(level, floor(log2 n0))``
        -- the pyramid floors an axis at one sample and copies it from there,
        so the lone sample covers ``2**f`` level-0 samples for the ``f`` folds
        that actually happened, and that is what the factor says
        (`control_plane_protocol.md` §8.11)."""
        self.level_scale(level)  # the range check
        return tuple(2 ** min(level, n.bit_length() - 1) for n in self.shape_tczyx[2:])

    def level_shape(self, level: int) -> tuple[int, int, int]:
        """§8.11: ``max(1, floor(shape_0 / scale))`` per axis."""
        s = self.level_scale(level)
        return tuple(max(1, n // s) for n in self.shape_tczyx[2:])

    def volume_at(self, t: int, c: int, level: int) -> np.ndarray:
        """The ``(Z, Y, X)`` volume at ``level``: level 0 as ``volume(t, c)``
        gives it, deeper levels block-mean downsampled by 2 per step with
        an odd trailing sample discarded (vizi's pyramid), in the dataset's
        dtype -- a float mean rounded back for an integer dtype."""
        v = np.ascontiguousarray(self.volume(t, c))
        for _ in range(self.level_scale(level).bit_length() - 1):
            z, y, x = (max(1, n // 2) for n in v.shape)
            if v.shape[0] >= 2 and v.shape[1] >= 2 and v.shape[2] >= 2:
                block = v[: 2 * z, : 2 * y, : 2 * x].reshape(z, 2, y, 2, x, 2)
                mean = block.astype(np.float64).mean(axis=(1, 3, 5))
            else:
                # An axis already down to one sample stays one sample (§8.11);
                # only the axes with two or more samples fold.
                mean = v.astype(np.float64)
                for axis, n in enumerate(v.shape):
                    if n >= 2:
                        idx = [slice(None)] * 3
                        idx[axis] = slice(0, 2 * (n // 2))
                        mean = mean[tuple(idx)]
                        shape = list(mean.shape)
                        shape[axis : axis + 1] = [n // 2, 2]
                        mean = mean.reshape(shape).mean(axis=axis + 1)
            v = np.rint(mean).astype(self.np_dtype) if np.issubdtype(self.np_dtype, np.integer) else mean.astype(self.np_dtype)
        return np.ascontiguousarray(v)

    def view_state(self) -> pb2.ViewState:
        state = pb2.ViewState(
            t=self.t,
            z=self.z,
            c=self.c,
            level=self.level,
            shape_tczyx=list(self.shape_tczyx),
            dtype=self.wire_dtype,
            has_crop=self.crop_min_zyx is not None,
            mip=self.mip,
            levels=[pb2.LevelInfo(level=n, shape_zyx=list(self.level_shape(n))) for n in range(self.levels)],
            dataset_uri=self.dataset_uri,
            dataset_name=self.dataset_name,
            open_generation=self.open_generation,
        )
        if self.crop_min_zyx is not None:
            state.crop_min_zyx.extend(self.crop_min_zyx)
            state.crop_max_zyx.extend(self.crop_max_zyx)
        visible = self.visible_channels if self.visible_channels is not None else range(self.shape_tczyx[1])
        state.visible_channels.extend(sorted(visible))
        # The pitch, host-wide, with `Geometry`'s semantics --
        # §8.7 sends a tool-less plugin here for what a tool reads from the
        # epoch.  The empty unit
        # with DISPLAY_FALLBACK is what "we do not know how big a voxel is"
        # looks like; a plugin that reads the numbers without that test has
        # invented a calibration.
        # Per axis -- a partly recorded file carries its
        # recorded axes with the unit and NaN for each unrecorded one, `FILE`.  NaN appears only then.
        pitch = self.pitch()
        if pitch == "some":
            state.spacing_zyx.extend(float("nan") if v is None else float(v) for v in self.spacing_zyx)
        else:
            state.spacing_zyx.extend(self.spacing_zyx if pitch == "all" else (1.0, 1.0, 1.0))
        known = pitch != "none"
        state.spacing_unit = self.spacing_unit if known else ""
        state.spacing_provenance = (
            pb2.SPACING_PROVENANCE_FILE if known else pb2.SPACING_PROVENANCE_DISPLAY_FALLBACK
        )
        # The frame interval the host KNOWS, `ReaderOpened`'s
        # rule (0 / "" = unknown), and the selected region.
        if self.time_increment_s:
            state.time_increment = float(self.time_increment_s)
            state.time_unit = "s"
        state.selected_region_id = self.selected_region_id
        if self.applied_filter is not None:
            state.filter.CopyFrom(self.applied_filter)
        if self.dataset_series is not None:
            state.dataset_series = self.dataset_series
        state.channels.extend(self.channel_infos())
        return state

    def channel_infos(self) -> list[pb2.ChannelInfo]:
        """Every channel with its name and where the name came from,
        as vizi's host builds them for `Geometry.channels`,
        `ViewState.channels` and `OpenExport.channels`: the file's name with
        `FILE`, else ``"Channel N"`` (1-based, the label the person sees) with
        `PLACEHOLDER`.  Never an empty name."""
        infos = []
        for i in range(self.shape_tczyx[1]):
            name = self.channel_names[i] if i < len(self.channel_names) else ""
            if name:
                infos.append(pb2.ChannelInfo(index=i, name=name, source=pb2.CHANNEL_NAME_SOURCE_FILE))
            else:
                infos.append(pb2.ChannelInfo(index=i, name=f"Channel {i + 1}", source=pb2.CHANNEL_NAME_SOURCE_PLACEHOLDER))
        return infos

    def geometry(self) -> pb2.Geometry:
        """§8.11's whole-dataset geometry, as the `1.0` host describes it:
        the extent, the crop in EDGE coordinates (index - 0.5), one level, and
        a spacing it does not know -- which is
        1.0 per axis with `DISPLAY_FALLBACK`, the value the host displays with
        and the one `ViewState` carries; the proto's "a plugin MUST NOT read
        DISPLAY_FALLBACK's 1.0 as one" names it."""
        _, _, depth, height, width = self.shape_tczyx
        lo = self.crop_min_zyx or (0, 0, 0)
        hi = self.crop_max_zyx or (depth, height, width)
        # `Geometry` stays finite -- a tool rebuilds physical
        # space from it -- so a partly recorded file is the display fallback.
        known = self.pitch() == "all"
        sz, sy, sx = self.spacing_zyx if known else (1.0, 1.0, 1.0)
        return pb2.Geometry(
            gen=self.open_generation,
            channels=self.channel_infos(),
            extent=pb2.Extent(z0=0, z_n=depth, y0=0, y_n=height, x0=0, x_n=width),
            crop=pb2.EdgeBox(
                z0=lo[0] - 0.5, z1=hi[0] - 0.5, y0=lo[1] - 0.5, y1=hi[1] - 0.5, x0=lo[2] - 0.5, x1=hi[2] - 0.5
            ),
            spacing_z=float(sz),
            spacing_y=float(sy),
            spacing_x=float(sx),
            spacing_unit=self.spacing_unit if known else "",
            spacing_provenance=(
                pb2.SPACING_PROVENANCE_FILE if known else pb2.SPACING_PROVENANCE_DISPLAY_FALLBACK
            ),
            # The SAMPLING FACTOR per level and axis, never a shape ratio, and
            # frozen once the axis is one sample (§8.11).
            levels=[pb2.LevelScale(z=float(sz), y=float(sy), x=float(sx)) for sz, sy, sx in (self.axis_scale(n) for n in range(self.levels))],
        )

    def time_mapping(self, revision: int = 1) -> pb2.TimeMapping:
        frames = self.shape_tczyx[0]
        return pb2.TimeMapping(
            gen=self.open_generation,
            revision=revision,
            t=[i * self.time_increment_s for i in range(frames)],
        )


# ---------------------------------------------------------------------------
# Live values and the plugin's pushes, as the host disposed of them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadRequest:
    """One `ReadDataVolume` as the plugin sent it: the level and the ROI in
    that level's coordinates (``None`` for the whole level)."""

    t: int
    c: int
    level: int
    roi: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    #: `ReadDataVolume.raw`: the source, whatever is shared.
    raw: bool = False


@dataclass(frozen=True)
class LiveValue:
    """One consumed Pulsed value: the lane it came from, its slot revision, the
    decoded `OverlayUpdate` (or the reason it was not applied), and the pane
    pair against the pane's current epoch."""

    segment: int
    region: int
    revision: int
    update: pb2.OverlayUpdate | None
    pane: tuple[int, int] | None
    applied: bool
    reason: str = ""


@dataclass
class Push:
    """A control-plane `OverlayUpdate` / `OverlayRemove`, with every loan it
    named read out of the arena and released."""

    kind: str
    overlay_id: str
    revision: int
    update: pb2.OverlayUpdate | None
    loans: list[Loan] = field(default_factory=list)


@dataclass
class Chunk:
    """One `OverlayChunk` the host APPLIED (`control_plane_protocol.md`
    §5.7): the key, the labels copied out of the arena (``None`` for an
    empty chunk, which is a state), their origin in level-``level``
    indices, and the loan the descriptor named -- released in the same tick
    unless the scenario holds releases."""

    overlay_id: str
    level: int
    t: int | None
    labels: np.ndarray | None
    position_zyx: tuple[int, int, int]
    loan: Loan | None = None
    inline: bool = False

    @property
    def key(self) -> tuple[str, int, int | None]:
        return (self.overlay_id, self.level, self.t)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (0, 0, 0) if self.labels is None else tuple(self.labels.shape)


@dataclass
class LabelTable:
    """One `OverlayLabelTable` the host applied: the LUT, the names and the
    visibility keyed by label id, and the overlay's style."""

    overlay_id: str
    colormap: np.ndarray  # (256, 4) u8
    label_names: dict[int, str]
    label_visible: dict[int, bool]
    style: pb2.OverlayStyle


#: What the reference host removes from a plugin's environment before
#: launching it: an injected library, a redirected Python, another
#: environment's activation and inherited installer settings must not ride from
#: vizi's process into a plugin.  A plugin that works only because it inherited
#: one of these works in a shell and not in vizi, so the emulator removes them
#: too (`docs/debugging.md`, "Trust, environment, and devices").
SCRUBBED_ENV_KEYS = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_FRAMEWORK_PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
)

#: Every inherited variable with one of these prefixes is removed as well,
#: except the names in `SCRUBBED_ENV_KEEP`.
SCRUBBED_ENV_PREFIXES = ("UV_", "PIP_")

#: The one inherited `UV_*` variable the host passes through: a mirror for
#: Python downloads, which a network that blocks the default host needs.
SCRUBBED_ENV_KEEP = ("UV_PYTHON_INSTALL_MIRROR",)


def scrubbed_environment(environ: "dict[str, str]") -> "dict[str, str]":
    """`environ` without what the host removes before launching a plugin.
    `SSL_CERT_FILE` goes too when it names no file, since a client that
    trusts a missing bundle trusts nothing."""
    out = {}
    for key, value in environ.items():
        if key in SCRUBBED_ENV_KEYS:
            continue
        if key.startswith(SCRUBBED_ENV_PREFIXES) and key not in SCRUBBED_ENV_KEEP:
            continue
        if key == "SSL_CERT_FILE" and not os.path.isfile(value):
            continue
        out[key] = value
    return out


#: `control_plane_protocol.md` §2.3: the id is the folder name when it is safe on EVERY OS -- a letter
#: or digit, then letters, digits, `.`, `_`, `-`; not ending in `.`; no
#: upper-case letter; not a Windows device name alone or before a `.`.
_SAFE_FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_DEVICE_NAME = re.compile(r"(con|prn|aux|nul|com[0-9]|lpt[0-9])(\.|\Z)", re.IGNORECASE)


#: §3.3.1: a float slider with no declared step has this many positions and
#: reads out to this many decimals; a step finer than `_MAX_STEP_DECIMALS`
#: decimals reads out at that many.
_FLOAT_SLIDER_POSITIONS = 1000
_FLOAT_SLIDER_DECIMALS = 3
_MAX_STEP_DECIMALS = 9


def _has_step(spec: dict) -> bool:
    """`PanelWidget.has_step`: the manifest entry declares ``step``, so a
    declared ``1.0`` is not the default (§3.3.1)."""
    return "step" in spec


def _int_slider_step(spec: dict) -> int:
    """The step an integer slider moves by: its declared step, a whole
    number.  The manifest parser refuses an integer slider whose step is not
    (``slider '…' step 2.5 is not a whole number``), so a spec that reaches
    here with one was never parsed, and is refused the same way."""
    step = float(spec.get("step", 1))
    if step != int(step):
        raise ValueError(f"slider {spec.get('id', '')!r} step {step:g} is not a whole number")
    return int(step)


def _nearest_stop(lo: float, hi: float, step: float, v: float) -> float:
    """The stop nearest ``v`` among ``lo + k * step`` for every ``k`` that
    stays below ``hi``, and ``hi`` itself: a slider moves by its step and the
    last move clamps at max, so when the range is not a multiple of the step
    every stop below max is still reached."""
    k = math.floor((v - lo) / step)
    below = min(lo + k * step, hi)
    above = min(lo + (k + 1) * step, hi)
    return above if (above - v) <= (v - below) else below


def slider_value(spec: dict, value: float) -> "int | float":
    """Where a slider can stop nearest ``value`` (manifest_schema.md §3.3.1,
    defaults min 0, max 1, step 1).  An integer slider, and a float slider
    with a declared step, stop at ``min + k * step`` -- an integer slider's
    step is whole (`_int_slider_step`) -- and the last position is clamped
    to ``max``, so every move is one step and the last may be shorter.  A
    float slider with no step has 1000 positions over its range.  The value is rounded to the readout's
    precision, so the event carries the value shown, never a position."""
    lo, hi = float(spec.get("min", 0)), float(spec.get("max", 1))
    v = min(max(float(value), lo), hi)
    if not spec.get("float", False):
        return int(_nearest_stop(lo, hi, _int_slider_step(spec), v))
    if _has_step(spec):
        stopped = _nearest_stop(lo, hi, float(spec["step"]), v)
    else:
        span = hi - lo
        position = round((0.0 if span <= 0 else (v - lo) / span) * _FLOAT_SLIDER_POSITIONS)
        stopped = lo + (position / _FLOAT_SLIDER_POSITIONS) * span
    if stopped >= hi:
        return hi
    return min(hi, round(stopped, _step_decimals(spec)))


def slider_readout(spec: dict, value: float) -> str:
    """The readout beside a slider, in the plugin's units, never its
    position: an integer as is; a float to its declared step's precision, or
    to three decimals with no step."""
    if not spec.get("float", False):
        return str(int(value))
    return f"{float(value):.{_step_decimals(spec)}f}"


def _step_decimals(spec: dict) -> int:
    """A float slider's readout precision: the fewest decimals that state its
    declared step (0.05 -> 2, 0.5 -> 1, 10 -> 0), at most 9; three with no
    step."""
    if not _has_step(spec):
        return _FLOAT_SLIDER_DECIMALS
    step = float(spec["step"])
    for decimals in range(_MAX_STEP_DECIMALS + 1):
        if math.isclose(step, round(step, decimals), rel_tol=1e-9, abs_tol=0.0):
            return decimals
    return _MAX_STEP_DECIMALS


def plugin_folder_name(plugin_id: str) -> str:
    """`<id>` in `plugins/<id>/` (`control_plane_protocol.md` §2.3): the id
    itself when it is safe on every OS; otherwise §2.3's six exact steps
    -- runs of other characters
    to one `_`, `.`/`_` trimmed from both ends, the first 64 kept, `plugin`
    if nothing is left, a leading `_` if the part before the first `.` is a
    device name, then `-` and 8 lowercase hex digits of the SHA-256 of the
    UTF-8 id.  The leading `_` is what keeps `aux.plugin` from becoming
    `aux.plugin-6e31b3b9`, still a Windows device name.

    **An id with an upper-case letter is DERIVED**, since
    Windows and macOS compare names without case and `Org.X` and `org.x`
    would otherwise share a folder: `Org.Example` -> `Org.Example-353f0972`.
    The derivation keeps the case -- the hash of the id as written is what
    keeps the two apart."""
    safe = (
        _SAFE_FOLDER.match(plugin_id)
        and not plugin_id.endswith(".")
        and not any("A" <= ch <= "Z" for ch in plugin_id)
        and not _DEVICE_NAME.match(plugin_id)
    )
    if safe:
        return plugin_id
    derived = re.sub(r"[^A-Za-z0-9._-]+", "_", plugin_id).strip("._")[:64] or "plugin"
    if _DEVICE_NAME.match(derived.split(".", 1)[0]):
        derived = "_" + derived
    return f"{derived}-{hashlib.sha256(plugin_id.encode('utf-8')).hexdigest()[:8]}"


def _enum_name(enum, value: int) -> str:
    """``enum.Name(value)``, or the number as text for a value this build's
    stubs do not know -- a plugin built against a later IDL may send one, and
    ``Name`` raises on it; a host must not fault on it."""
    try:
        return enum.Name(value)
    except ValueError:
        return str(value)


def _resolve_command(command: list, plugin_dir: Path) -> list[str]:
    """Where a manifest names a plugin-local ``.venv`` interpreter that does
    not exist -- a fresh checkout, CI -- substitute ``VIZI_HOSTSIM_PYTHON``,
    else this process's own interpreter.

    **This is an emulator convenience, NOT the host's rule.**  The host runs
    ``entry.command`` as written on Unix (Windows only translates
    ``bin`` -> ``Scripts``, `manifest_schema.md` §2.2), so a missing ``.venv``
    fails to launch in vizi; the substitution exists so CI can run every
    ``hostsim.py`` from one synced environment.  Only a
    ``.venv`` interpreter is substituted; any other command runs as written."""
    command = [str(c) for c in command]
    if not command:
        return command
    first = Path(command[0])
    is_venv = not first.is_absolute() and first.parts and first.parts[0] == ".venv"
    if is_venv and not (plugin_dir / first).exists():
        command[0] = os.environ.get("VIZI_HOSTSIM_PYTHON") or sys.executable
    return command


def _message_children(descriptor, value) -> list:
    """The message values under one field: the one, the repeated ones, or a
    ``map<k, Message>``'s values.  A map field's ``ListFields`` value iterates
    its KEYS, so a walker that treats it as a repeated message field is handed
    scalars -- ``OverlayLabelVolume.label_names`` (``map<uint32, string>``)
    is such a field."""
    if descriptor.message_type is not None and descriptor.message_type.GetOptions().map_entry:
        entry = descriptor.message_type.fields_by_name["value"]
        return list(value.values()) if entry.type == entry.TYPE_MESSAGE else []
    return list(value) if _is_repeated(descriptor) else [value]


def _is_repeated(descriptor) -> bool:
    """`is_repeated` where the protobuf build has it (``label`` is deprecated
    there), ``label`` where it does not."""
    repeated = getattr(descriptor, "is_repeated", None)
    return descriptor.label == descriptor.LABEL_REPEATED if repeated is None else bool(repeated)


def _geom_buffers(message):
    """Every populated `GeomBuffer` below a message, in field order."""
    for descriptor, value in message.ListFields():
        if descriptor.type != descriptor.TYPE_MESSAGE:
            continue
        for sub in _message_children(descriptor, value):
            if sub.DESCRIPTOR.full_name.endswith("GeomBuffer"):
                yield sub
            else:
                yield from _geom_buffers(sub)


def _panel_widgets(root: dict) -> dict[str, tuple[str, dict]]:
    """Every widget in a declared panel's tree, containers included -- the
    host's ``build_panel`` registers each by id, so a `set_visible` on a
    ``row`` is as legal as one on a ``label``."""
    found: dict[str, tuple[str, dict]] = {}
    stack = [root] if root else []
    while stack:
        node = stack.pop()
        if "id" in node:
            found[str(node["id"])] = (str(node.get("kind", "")), node)
        stack.extend(node.get("children", []))
        # A `list`'s `row` template widgets are CELLS, in the
        # panel's id space, each tagged with its list.
        if node.get("kind") == "list":
            for cell in node.get("row", []):
                if "id" in cell:
                    found[str(cell["id"])] = (str(cell.get("kind", "")), {**cell, "_list": str(node["id"])})
    return found


#: §9's per-field bounds as the `1.0` host applies them to EVERY plugin
#: message (the SDK's `vizi_plugin_sdk/control.py` ``_FIELD_BOUNDS``,
#: a repeated field or map past its count, a bytes or string field past
#: `MAX_INLINE_BYTES`.  Keyed by full name; anything absent takes
#: `MAX_REPEATED`.  **Taken from the synced SDK, not copied here**: the table
#: is §9's (the contract, like the `MAX_*` values beside it), and a copy
#: would go stale.
_FIELD_BOUNDS: dict[str, int] = dict(_SDK_FIELD_BOUNDS)


def _over_bound(message) -> tuple[str, str] | None:
    """``(rule, detail)`` for the first field of ``message`` over its §9
    bound, depth first -- ``"repeated"`` or ``"inline"`` -- or ``None``, as
    the host walks a message."""
    for descriptor, value in message.ListFields():
        repeated = _is_repeated(descriptor)
        mapping = repeated and descriptor.message_type is not None and descriptor.message_type.GetOptions().map_entry
        if repeated:
            limit = _FIELD_BOUNDS.get(descriptor.full_name, MAX_REPEATED)
            if len(value) > limit:
                return "repeated", f"{descriptor.full_name} carries {len(value)}, over {limit}"
        if descriptor.type in (descriptor.TYPE_BYTES, descriptor.TYPE_STRING):
            for item in value if repeated else (value,):
                if len(item) > MAX_INLINE_BYTES:
                    return "inline", f"{descriptor.full_name} carries {len(item)} bytes inline, over {MAX_INLINE_BYTES}"
            continue
        if descriptor.type != descriptor.TYPE_MESSAGE:
            continue
        if mapping:
            entry = descriptor.message_type.fields_by_name["value"]
            children = list(value.values()) if entry.type == entry.TYPE_MESSAGE else []
        else:
            children = list(value) if repeated else [value]
        for child in children:
            found = _over_bound(child)
            if found is not None:
                return found
    return None


def lut_row(label: int) -> int:
    """§8.8's LUT rule: row 0 is label 0's alone; a foreground id draws with
    ``1 + (id - 1) mod 255``, so ids above 255 wrap over the foreground rows
    and never onto the background row."""
    return 0 if label == 0 else 1 + (label - 1) % 255


# ---------------------------------------------------------------------------
# The emulated host
# ---------------------------------------------------------------------------


class Host:
    def __init__(
        self,
        plugin_dir: Path,
        dataset: Dataset,
        *,
        session_epoch: int = SESSION_EPOCH,
        launch_timeout: "float | None" = None,
        protocol_minor: "int | None" = None,
        theme: "str | None" = "dark",
    ) -> None:
        #: §3.1: the version this host's `Hello` announces -- the SDK's own
        #: unless ``protocol_minor`` or ``VIZI_HOSTSIM_PROTOCOL_MINOR`` says
        #: otherwise, so a plugin's older-host and newer-host branches can
        #: both be run.  The session runs at the plugin's `Ready` minor.
        self.protocol: tuple[int, int] = (PROTOCOL_MAJOR, host_protocol_minor(protocol_minor))
        #: The handshake's refusal of the plugin's `Ready`, when there was one.
        self.handshake_refusal: str | None = None
        #: The readout the last slider `invoke_widget` left beside its handle.
        self.last_readout: str | None = None
        #: The high half of every segment id this session mints
        #: (`doorbell_M_memory.md` §3).  One `Host` alone is epoch 1, as
        #: before; `vizi_hostsim.chain.Chain` runs several sessions at once and
        #: gives each its own, since session epochs never alias -- which is
        #: what lets it mint a processing-stage ring's id from an epoch no
        #: session uses (`processing_stages.md` §4).
        self.session_epoch = session_epoch
        # What the Broker reads before launching: the manifest, for the command
        # to run and the id `Ready.plugin_id` echoes.  The version is kept for
        # a scenario's log line; nothing here compares it to anything (§3.1).
        self.plugin_dir = plugin_dir.resolve()
        self.manifest = json.loads((self.plugin_dir / "vizi-plugin.json").read_text(encoding="utf-8"))
        # The host scans a manifest before it will launch anything, and a
        # manifest it refuses never runs.  Refuse what the reference parser
        # refuses -- `engines` first, against the minor this host presents:
        # engines is the floor, the oldest host the author tested.
        try:
            parsed = parse_manifest(self._manifest_at_this_minor(), self.plugin_dir)
        except ManifestError as exc:
            raise ManifestError(f"vizi refuses {self.plugin_dir / 'vizi-plugin.json'}: {exc}") from None
        self.command: list[str] = _resolve_command(self.manifest["entry"]["command"], self.plugin_dir)
        self.plugin_id: str = self.manifest["id"]
        self.plugin_version: str = self.manifest["version"]
        #: §8.8: the `Roi.type_name`s this plugin shapes
        #: (manifest §3.8).  A region of one is ASKED: its handles' drags go
        #: to the plugin (`ShapeDrag`).
        self.region_types: frozenset[str] = frozenset(t.id for t in parsed.region_types)
        #: §9: this launch's one budget, spawn to `Ready` --
        #: ``launch_timeout`` is vizi's explicit override, for a scenario.
        self.launch_budget: float = launch_budget(parsed.startup_s, launch_timeout)
        self.launch_deadline: float | None = None
        self.dataset = dataset
        self._tempdir = tempfile.TemporaryDirectory(prefix="vizi-hostsim-")
        self.directory = Path(self._tempdir.name)
        self._uds = str(self.directory / "plugin.sock")
        #: `control_plane_protocol.md` §2.3: the
        #: plugin's settings folder, passed as ``VIZI_PLUGIN_SETTINGS_DIR``.
        #: **NAMED, NOT CREATED**: beside the manifest the host
        #: only checks that a file can be made there, and the SDK creates the
        #: folder on first use, so a plugin that keeps no settings leaves no
        #: empty folder behind, and an accessor that refuses a missing
        #: folder fails here as it fails on the host.  The
        #: host puts it beside the manifest; the emulator puts it in its temp
        #: dir so a run leaves nothing in the checkout.  Its parent exists.
        #: ``None`` sends the variable ABSENT -- the host's answer when
        #: neither this folder nor its fallback is usable.
        self.settings_dir: Path | None = self.directory / "settings"
        #: The open dataset's per-file state folder, sent on
        #: `ViewState` and as `OpenExport`'s source: `plugins/<id>/` in the
        #: file's companion directory.  NAMED, NEVER CREATED -- the plugin
        #: creates it on its first write.  ``""`` says the dataset has no
        #: local path, which a plugin must read as "none", never as a path.
        self.file_state_dir: str = self.file_state_for("/data/dataset.tif")
        self.permissions: set[str] = set(self.manifest.get("permissions", []))
        #: `manifest_schema.md` §5.1: the host pushes this
        #: session asked to be arranged.  An unknown name is warned and
        #: ignored, `permissions`'
        #: posture, so a manifest from a later SDK still launches here.
        declared_subs = self.manifest.get("subscriptions", [])
        #: ``"view"`` and ``"regions"`` each need
        #: `data_access`, and declared without it the host warns and does not
        #: arrange it, as for an unknown name (`manifest_schema.md` §5.1).
        known_subs = {"epochs", "view", "regions"}
        needs_data = {"view", "regions"}
        granted = "data_access" in set(self.manifest.get("permissions", []))
        self.subscriptions: set[str] = {s for s in declared_subs if s in known_subs and (s not in needs_data or granted)}
        # warned where `permissions`' unknown names are, once `diagnostics` exists
        self._unknown_subscriptions = [s for s in declared_subs if s not in known_subs]
        self._ungranted_subscriptions = [s for s in declared_subs if s in needs_data and not granted]
        #: True when this session is on the `EpochChange` chain: it declares a
        #: tool (§8.7's rule, unchanged) or it asked for ``"epochs"``.  A
        #: session that asked is held to §8.10's acknowledgement contract just
        #: like a tool session, which is why asking is opt-in.
        self.epochs_arranged: bool = bool(self.manifest.get("contributions", {}).get("tools")) or "epochs" in self.subscriptions
        #: Seconds each `ReadDataVolume` answer waits before depositing: a
        #: slow host.  **It sleeps on the gRPC reader thread**, so everything
        #: the plugin sends meanwhile waits behind it -- fine for a slow host,
        #: useless for proving a busy guard, which needs the plugin's refusal
        #: to arrive WHILE the read is held.  For that use `hold_reads()`.
        #: 0 by default.
        self.read_delay: float = 0.0
        #: `hold_reads` / `await_reads` / `release_reads`: while held, a
        #: `ReadDataVolume` is recorded and PARKED -- the reader thread goes
        #: straight back to work, so the plugin's `CommandDone`, its refusals
        #: and its panel patches keep arriving -- and is answered only by
        #: `release_reads`.  Parking is what makes "prove the busy guard engages"
        #: deterministic rather than a race against `read_delay`.
        self._reads_held = False
        self._held_reads: list[tuple[str, pb2.ReadDataVolume]] = []
        self._read_lock = threading.RLock()
        contributions = self.manifest.get("contributions", {})
        #: The declared tool ids, which `ActivateTool` is checked against.
        self.tool_ids: set[str] = {t["id"] for t in contributions.get("tools", [])}
        self.overlay_ids: set[str] = {o["id"] for o in contributions.get("overlays", [])}
        #: `manifest_schema.md` §3.5: the CHUNKED overlays and each one's
        #: ``time`` -- ``"per_t"`` (the default) or ``"static"``.  What the
        #: host reads before it grants anything: a chunked overlay's arena is
        #: host-sized (`Segment.chunk_arena`), its state is `OverlayChunk` and
        #: `OverlayLabelTable`, and an `OverlayUpdate` to it is refused.
        self.chunked: dict[str, str] = {
            o["id"]: o.get("time", "per_t") for o in contributions.get("overlays", []) if o.get("chunked")
        }
        # The rule that a tool's `view_targets` need a panel's is the
        # reference parser's, applied above with every other manifest rule.
        for o in contributions.get("overlays", []):
            if o.get("chunked") and "segment_mib" in o:
                raise ValueError(f"overlay {o['id']!r}: segment_mib is refused on a chunked overlay (manifest_schema.md §3.5)")
            if "time" in o and not o.get("chunked"):
                raise ValueError(f"overlay {o['id']!r}: time applies only to a chunked overlay")
        #: §5.7: overlay id -> the want set this host last sent, in its
        #: order, as ``(level, t | None)`` pairs; the chunks it holds, keyed
        #: ``(overlay, level, t)``; every applied chunk in arrival order;
        #: the tables; what was dropped (unwanted, counted, loan released),
        #: evicted (its want left the set), or refused (site 30, told to the
        #: plugin through `ContractViolation`).
        self.wants: dict[str, list[tuple[int, int | None]]] = {}
        self.chunks: dict[tuple[str, int, int | None], Chunk] = {}
        self.chunk_log: list[Chunk] = []
        self.label_tables: dict[str, LabelTable] = {}
        self.table_log: list[LabelTable] = []
        self.dropped_chunks: list[tuple[str, int, int | None]] = []
        self.evicted: list[tuple[str, int, int | None]] = []
        self.refusals: list[tuple[str, str]] = []
        #: The host copies a chunk out and releases its loan in the tick that
        #: applies it (§5.7).  A scenario that wants to SEE the plugin defer
        #: -- a full arena, no shortfall, the retry after the release -- sets
        #: this and releases by hand (`release_chunk`, `release_held_chunks`).
        self.hold_chunk_releases = False
        #: The host's transport between the plugin and the panels (§8.4): it
        #: applies each table op as it arrives unless a scenario sets
        #: ``hold_table_drain``, which models a drain that has not run yet.
        #: Then every table op -- `set_table`, `patch_table` and a table's
        #: `set_value` -- waits in ``table_transport`` in arrival order, as
        #: ``(panel, op)``, each widget's bounded by `TABLE_HELD_BYTES`;
        #: ``drain_tables()`` applies them in that order.
        self.hold_table_drain = False
        self.table_transport: list[tuple[str, pb2.PatchOp]] = []
        self._table_transport_bytes: dict[tuple[str, str], int] = {}
        self.held_chunks: dict[tuple[str, int, int | None], Loan] = {}
        #: The host's wants follow the panes once per PUMP TICK (40 ms,
        #: `doorbell_protocol.md` §11), so two `ChunksWanted` for one overlay
        #: are never closer than that; `want_chunks` keeps the cadence.
        self.want_tick = 0.04
        self._last_want: dict[str, float] = {}
        self._outbound: queue.Queue = queue.Queue()
        self._waiters: dict[str, queue.SimpleQueue] = {}
        #: The kind of each request a waiter holds, by id: its reply must be
        #: of the kind that request takes (`_HOST_REQUEST_REPLIES`), or `Error`.
        self._asked: dict[str, str] = {}
        #: The command each pending `InvokeCommand` invoked, by id: its
        #: `CommandDone` must echo it (§8.3).
        self._invoked: dict[str, str] = {}
        self._lock = threading.Lock()
        self.ready: pb2.Ready | None = None
        #: §3.1: the plugin's identity string, off the `Ready`'s envelope.
        self.ready_id: str | None = None
        self.patches: list[tuple[str, str, object]] = []
        self.images: list[tuple[str, str, Loan]] = []
        self.contributions: queue.SimpleQueue = queue.SimpleQueue()
        self.released: list[tuple[int, int, int, int]] = []
        self.loans_released: list[tuple[int, int, int, int]] = []
        #: §5.6's ledger: reservation ``(segment, offset, len)`` -> (declared
        #: uses, uses still to dispose).  A reservation is released ONCE, at
        #: zero; ``uses_disagreement`` (a later message declaring a different
        #: count) applies the first count and is recorded in ``diagnostics``.
        self.loan_uses: dict[tuple[int, int, int], list[int]] = {}
        self.diagnostics: list[str] = []
        #: Names in the manifest this host read and ignored -- `permissions`'
        #: posture, and `subscriptions`' (§5.1).  Not `diagnostics`: an
        #: ignored name is a manifest from a later SDK, not a protocol fault.
        self.manifest_warnings: list[str] = [
            f"subscription '{s}' is not one this host knows; ignored" for s in self._unknown_subscriptions
        ] + [
            f"subscription '{s}' needs the data_access permission; not arranged" for s in self._ungranted_subscriptions
        ]
        #: The subscribed pushes (§8.7), as the host sends them: each state restated every pump tick and sent only when
        #: it changed, so a scrub costs one message per tick.  Every one sent,
        #: in order, for a driver to read.
        self.view_pushes: list[pb2.ViewChanged] = []
        self.region_pushes: list[pb2.RegionsChanged] = []
        self._view_sent = None
        self._regions_sent = None
        #: `RoiRecord.revision`: region id -> (the record object, revision).
        #: A different object under an id, or an in-place edit (`_touch`), is
        #: a new revision from one counter, never reused.
        self._region_revisions: dict[str, tuple[object, int]] = {}
        self._region_counter = 0
        #: `GetColormap` (§8.4): name -> 1024 bytes of RGBA.  The
        #: names are vizi's curated set; only ``gray`` is vizi's exact table,
        #: the rest are the emulator's own ramps, so a driver compares shape,
        #: not color.  A driver may add a user colormap here.
        self.colormaps: dict[str, bytes] = _default_colormaps()
        #: `RequestSavePath` (§8.9), scripted: the path the "person"
        #: picks, or ``None`` to decline; and every request, in order.
        self.save_path_answer: str | None = None
        self.save_requests: list[pb2.RequestSavePath] = []
        #: The newest `set_progress` per (panel, widget) -- a fraction in
        #: [0, 1], or ``None`` for indeterminate.  A progress
        #: widget binds no command, so nothing is ever echoed back.
        self.progress: dict[tuple[str, str], float | None] = {}
        #: Every `set_progress` in order, for a driver that wants to see the
        #: work advance rather than only where it ended.
        self.progress_log: list[tuple[str, str, float | None]] = []
        #: The decoded-message ceiling this host enforces on a `GetRois`
        #: reply (§9).  A driver lowers it to exercise the
        #: refusal without pushing 32 MiB of poses through the test, the way
        #: it sizes an arena down to exercise a shortfall.
        self.max_message_bytes: int = MAX_MESSAGE_BYTES
        #: `(region_id, t)` per `ReadRoiInclusion` answered.
        self.inclusion_requests: list[tuple[str, int]] = []
        #: The host's voxel budget for one `ReadRoiInclusion`
        #: (§8.7's table): a region whose CLIPPED box holds more voxels is
        #: refused `OUT_OF_BOUNDS`, which is why the clip box exists.  The
        #: spec names no value; a driver lowers this to drive the refusal.
        self.inclusion_voxel_budget: int = INCLUSION_VOXEL_BUDGET
        #: `RequestStatusQuery` subjects the plugin asked about.
        self.status_queries: list[str] = []
        # the picker's dialog, when a driver is holding it up
        self._picker_held: threading.Event | None = None
        self._picker_pending: dict[str, str] = {}
        self._picker_abandoned = False
        #: Request ids of `RequestAsset`s refused because a picker was
        #: already up (§8.9: one per session at a time).
        self.picker_refusals: list[str] = []
        #: Every `ExtendRoiPoses` batch accepted, by `contrib_id`.
        self.pose_extensions: list[pb2.RoiPoseExtension] = []
        #: The dense pose domain [t_start, t_end] each region carries, by
        #: region id -- what a window is a slice of, and what an extension
        #: grows.  A region contributed without pose rows has none.
        self.pose_domains: dict[str, tuple[int, int]] = {}
        self.acks: dict[int, pb2.GrantAck] = {}
        #: `SegmentReleased` per retired segment (`shared_memory_protocol.md` §9).
        self.retired: dict[int, pb2.SegmentReleased] = {}
        #: §9.1: segments this host has retired
        #: (`RetireSegment`, or superseded) whose `SegmentReleased` has not
        #: arrived.  A push naming one is dropped QUIETLY, its loans returned
        #: -- never refused, never reported.
        self.retiring_segments: set[int] = set()
        #: Every push dropped that way: (what, segment id).
        self.dropped_retiring: list[tuple[str, int]] = []
        #: For each released segment, how many quiet drops had happened when
        #: its `SegmentReleased` arrived (a scenario checks the order).
        self.drops_at_release: dict[int, int] = {}
        self.tool_acks: dict[int, int] = {}
        self.epoch_acks: dict[int, int] = {}
        self.mapping_acks: dict[int, int] = {}
        #: The acknowledged `TimeMapping` revision per gen, which each
        #: `EpochChange` names.
        self.time_revisions: dict[int, int] = {}
        self.violations: list[pb2.ContractViolation] = []
        self.shortfalls: list[pb2.SegmentShortfall] = []
        self.reads: list[tuple[int, int, int]] = []
        #: Every `ReadDataVolume` with its ROI, for a driver that checks WHAT
        #: was read and not only how often (`ray_probe`'s level scenarios).
        self.read_requests: list[ReadRequest] = []
        #: The self-serve guard (`reader_writer_contract.md` §1.9): when
        #: the active dataset is served by THIS plugin's reader, the host
        #: answers `ReadDataVolume` with `UNSUPPORTED` at once rather than
        #: call back into a reader it is blocked on.  `GetViewState` is not
        #: guarded.  The emulator's dataset is never plugin-served, so a
        #: scenario that wants the guard's reply sets this.
        self.self_serve_guard = False
        self.segments: dict[int, Segment] = {}
        self.pushes: list[Push] = []
        self.hints: list[Hint] = []
        self.live: list[LiveValue] = []
        self.unhandled: list[str] = []
        #: `GrantState` answers by the query id they echo, and what
        #: `recover_grant` did about each.
        self.grant_states: dict[str, pb2.GrantState] = {}
        self.recoveries: list[tuple[int, str, str]] = []
        self._auto_index = 999
        #: `V` rings the emulator granted ITSELF, the host's way: lazily at
        #: the first read, sized to that deposit, and grown by `supersedes`
        #: (see `_v_for`).  In grant order.
        self.v_rings: list[int] = []
        #: Replies sent BARE into a `V` ring whose installation had already
        #: ridden an earlier reply and was not yet acknowledged -- the case the
        #: spec says the plugin holds, and which the host sends too.
        self.bare_replies = 0
        #: Outgrown `V` rings awaiting their last deposit's release before
        #: `RetireSegment` (see `_v_for`), and the knob that supersedes them
        #: undrained instead, which the host does not do.
        self._v_draining: set[int] = set()
        self.v_supersede_undrained = os.environ.get("VIZI_HOSTSIM_SUPERSEDE_UNDRAINED") == "1"
        #: Outgrown `V` rings retired after draining rather than superseded.
        self.v_retired_on_drain: list[int] = []
        #: Per-export `X` arenas (`grant_export_arena`), reused at each
        #: `ChunkWritten` as the host reuses them.
        self._per_export_x: set[int] = set()
        #: §8.6: each export a chunk `Error` ended, which the host
        #: then aborted -- ``(handle, error kind, ExportDone or abort error kind)``.
        self.export_aborts: list[tuple[str, int, "pb2.ExportDone | int"]] = []
        #: Terminal doorbell framing faults the plugin caused (at most one).
        self.doorbell_faults: list[str] = []
        #: Ops the plugin sent that this version does not define (§5: dropped
        #: and counted).
        self.skipped_ops: list[int] = []
        #: `D` hints naming a segment not granted -- 0, or just retired.
        self.stray_hints: list[Hint] = []
        #: Called as ``on_hint(host, hint)`` on the doorbell thread for every
        #: `D` naming a granted segment, after it is recorded: how a
        #: `vizi_hostsim.chain.Chain` hears a stage hand a slot on and relays
        #: the hint to the next stage's session (`processing_stages.md` §6.2).
        self.on_hint = None
        #: `StageArmAck`s, ``(pipeline, position)`` -> ``(revision, stage_id)``
        #: (`processing_stages.md` §4: an arm is identified by the pair).
        self.stage_arm_acks: dict[tuple[int, int], tuple[int, str]] = {}
        #: Host -> plugin `D`s this session sent, as ``(segment, regions)``.
        self.notifies: list[tuple[int, frozenset[int]]] = []
        #: Doorbell sends come from the driver's thread and, in a chain, from
        #: another session's doorbell thread relaying a hint; the record id
        #: must still rise by exactly one per record (`doorbell_protocol.md` §4).
        self._db_lock = threading.RLock()
        #: Whether a `D` brings the lane scan forward at once, as the host's
        #: does.  Off, a scenario can look at a value AFTER the pane has moved.
        self.scan_on_hint = True
        #: `doorbell_protocol.md` §4: one counter per direction per session,
        #: starting at 1 and incremented by exactly one per record.
        self._record_id = 1
        self._seq = 0
        #: Which segment the next deposit goes into, and whether its
        #: `InstallSegment` still has to ride the reply that names it.
        self.volume_segment: int | None = None
        self.ride_along: set[int] = set()
        #: Which `X` ring an export chunk is deposited into.
        self.export_segment: int | None = None
        #: The host's annotation store: native ROIs by region id, and the
        #: masks `CompileLabelVolume` made, which are ROIs too.
        self.rois: dict[str, pb2.RoiRecord] = {}
        self.masks: dict[str, Mask] = {}
        #: Every `ActivateTool` the plugin sent (§8.10), in order.
        self.activations: list[pb2.ActivateTool] = []
        #: The ENGAGED tool: one per session,
        #: whose binding is live.  Engaging another displaces it -- its binding
        #: is retired first -- and each engagement is a NEW handle from the
        #: session's counter.  ``None`` when nothing is.
        self.engaged: str | None = None
        #: `ActivateTool` refusals this host sent, as ``(tool_id, kind, message)``.
        self.activation_refusals: list[tuple[str, int, str]] = []
        #: The panels the host has OPENED this session (`open_panel`), which
        #: are the only ones a `PanelPatch` may name (a panel not live in the
        #: session is refused `panel_patch_refused`).  Once opened a panel stays live -- the host
        #: hides its dock, it does not tear it down.
        self.opened_panels: set[str] = set()
        #: A declared panel that is not open KEEPS the
        #: latest op per (widget, property, row) and applies them when it
        #: opens.
        self._held_patches: dict[str, dict[tuple, pb2.PatchOp]] = {}
        #: Each list's current ordered row keys, per
        #: (panel, list); every applied row-addressed op as
        #: ``(panel, widget, row, value)``; and a selectable list's selection.
        self.list_rows: dict[tuple[str, str], list[str]] = {}
        self.row_patches: list[tuple[str, str, str, object]] = []
        #: The latest image each list row's image cell shows, by
        #: ``(panel, widget, row)``, as a `Loan` (its ``payload`` is the PNG).
        self.row_images: dict[tuple[str, str, str], Loan] = {}
        self.list_selection: dict[tuple[str, str], str] = {}
        #: Each `table` widget's rows and selection (§8.4), per (panel,
        #: table), as the SDK's `tables.TableState` -- the reference rules
        #: (`tables.apply_op`, `tables.select`) are applied to it, never a
        #: copy of them.  ``_table_cells`` is each row's cells as the host
        #: shows them, padded to the declared columns with ``None`` for a
        #: blank cell -- a missing one, or one of a kind the host does not
        #: show; ``table_blank_kinds`` the cell kinds shown blank (vizi's
        #: local diagnostic, said once each).  `table_rows` and
        #: ``table_selection`` read them.  The rows the person can see are
        #: not modeled.
        self.tables: dict[tuple[str, str], _tables.TableState] = {}
        self._table_cells: dict[tuple[str, str], dict[str, tuple]] = {}
        self.table_blank_kinds: set[str] = set()
        #: `WidgetGeometry` (§8.4): the size
        #: each `image` widget is drawn at, ``(width_px, height_px, dpr)`` by
        #: (panel, widget) -- `DEFAULT_IMAGE_GEOMETRY` until `resize` says
        #: otherwise -- and every one sent, in order, as the message.
        self.image_geometry: dict[tuple[str, str], tuple[int, int, float]] = {}
        self.geometry_sent: list[pb2.WidgetGeometry] = []
        self._geometry_last: dict[tuple[str, str], tuple[int, int, float]] = {}
        #: `HostAppearance` (§8.4): the person's theme, sent once right after
        #: `Ready` and again by `set_theme` only when the message differs from
        #: the last one sent; ``theme=None`` is a host that sends none.  Every
        #: one sent is kept, in order.
        if theme is not None and theme not in APPEARANCE_PALETTES:
            raise ValueError(f"theme is one of {sorted(APPEARANCE_PALETTES)}, not {theme!r}")
        self.theme: "str | None" = theme
        self.appearance_sent: list[pb2.HostAppearance] = []
        self._appearance_last: "bytes | None" = None
        #: Set-view (§8.15), as the host keeps it: the
        #: causes minted for `moves_view` widgets (`invoke_widget`), the
        #: person's touches since, and each request's recorded answer.
        self._causes: dict[int, dict] = {}
        self._next_cause = 1
        self._touch_serial = 0
        self._touched: dict[str, int] = {}
        #: The person's "Let <plugin> move the view" switch, per panel -- the
        #: first row of a panel that declares `view_targets` or
        #: `follow_targets`.  As in vizi it is OFF at every open, so a
        #: driver turns it on (`set_view_toggle`, or `view_toggle = True` for
        #: every panel) before a cause can move anything.
        self.view_toggles: dict[str, bool] = {}
        #: Panels the person closed: in vizi a closed dock is HIDDEN, not
        #: unopened -- its patches and images keep applying, but
        #: its switch counts for nothing while it is hidden.
        self.hidden_panels: set[str] = set()
        #: Whether a dataset is open.  With none, a `moves_view`
        #: interaction carries cause 0, a tool's FIRE mints no cause, and
        #: `ViewChanged` says generation 0 and nothing else (§8.7).
        self.dataset_open = True
        #: A movie export owning the view (`EXPORT_ACTIVE`), for a driver.
        self.export_active = False
        #: A scripted outcome, ``(outcome, reason, detail)`` by name
        #: (``("REFUSED", "SUPERSEDED", "...")``), answered after the cause
        #: checks in place of the rules -- so a driver reaches any result.
        self.view_script: tuple | None = None
        #: Every ``(ViewRequest, ViewRequestResult)``, and every move applied.
        self.view_requests: list[tuple[pb2.ViewRequest, pb2.ViewRequestResult]] = []
        self.view_moves: list[dict] = []
        # -- Follow (§8.15) --
        #: panel id -> the offer's label (an unopened panel's offer is kept).
        self.follow_offers: dict[str, str] = {}
        #: The one running Follow, or None: follow_id, panel_id, targets,
        #: touches, last_seq, pending, last_applied_at, generation.
        self.follow: "dict | None" = None
        self._next_follow = 1
        #: The panel whose Follow the person may Resume (after USER_TOOK_CONTROL).
        self.resumable: "str | None" = None
        #: Every `FollowStarted` and `FollowEnded` sent, in order.
        self.follows_started: list[pb2.FollowStarted] = []
        self.follows_ended: list[pb2.FollowEnded] = []
        #: Every state applied: (follow_id, seq, targets).
        self.follow_applied: list[tuple[int, int, dict]] = []
        #: Every state dropped: (follow_id, seq, why).
        self.follow_dropped: list[tuple[int, int, str]] = []
        self._follow_lock = threading.RLock()
        self._follow_pump_on = False
        #: A tool's FIRE is a cause, by its `(tool, interaction)`.
        self._tool_causes: dict[tuple[int, int], dict] = {}
        #: The camera state `look_at` / `orbit_point` last set.
        self.camera: dict[str, object] = {}
        #: Per declared panel, widget id -> (kind, manifest spec); and each
        #: dropdown's CURRENT options, which `set_options` replaces and a
        #: string `set_value` must name.
        self._panel_widgets: dict[str, dict[str, tuple[str, dict]]] = {
            panel["id"]: _panel_widgets(panel.get("root", {})) for panel in contributions.get("panels", [])
        }
        self._dropdown_options: dict[tuple[str, str], list[str]] = {
            (panel_id, wid): [str(o) for o in spec.get("options", [])]
            for panel_id, widgets in self._panel_widgets.items()
            for wid, (kind, spec) in widgets.items()
            if kind == "dropdown"
        }
        #: The host's request ledger: every
        #: plugin REQUEST taken in and not yet answered, id -> awaiting_user.
        #: Opened as a request is dispatched, cleared as ANY reply echoing its
        #: id leaves (`send`), so `RequestStatusQuery` answers PENDING for a
        #: read parked by `hold_reads` as much as for a held picker.
        self._open_requests: dict[str, bool] = {}
        #: Data-access requests that arrived after `Shutdown` and were dropped
        #: unanswered, as vizi's data-access service drops them.
        self.dropped_in_shutdown: list[tuple[str, str]] = []
        #: `ReleaseRequest` hints by segment (a HINT: recorded, never acted
        #: on); a segment not held is
        #: dropped as the host drops it.
        self.release_requests: dict[int, str] = {}
        #: Per reader handle, its `ReaderOpened` and the `G` arena granted for
        #: it -- what a reply is checked against.
        self.readers: dict[str, pb2.ReaderOpened] = {}
        self.reader_arenas: dict[str, int] = {}
        self._last_reader: str | None = None
        #: Per reader handle, the URI it opened: a `SeriesList` for the URI
        #: the open reader holds means the plugin closed it first (§1.2a).
        self.reader_uris: dict[str, str] = {}
        #: Every `SeriesList` a reader answered, ``(uri, SeriesList)``.
        self.series_lists: list[tuple[str, pb2.SeriesList]] = []
        #: Plugin messages refused over §9's per-field bounds, ``(kind, detail)``.
        self.over_bound: list[tuple[str, str]] = []
        self.compiles: list[pb2.CompileLabelVolume] = []
        self._region_seq = 0
        #: `RequestAsset` slots: path (the user picked it) or None (declined).
        self.assets: dict[str, Path | None] = {}
        self.asset_requests: list[pb2.RequestAsset] = []
        #: The arrangements this host has made: pane -> epoch, tool_id -> handle.
        self.epochs: dict[int, int] = {}
        self.tools: dict[str, int] = {}
        #: Every handle ever bound, to its tool id (a handle is never reused).
        self._tool_of_handle: dict[int, str] = {}
        self._next_tool = 1
        self._next_interaction = 1
        self._open_interactions: dict[int, "_Gesture"] = {}
        self.shutdown_report: pb2.ShutdownReport | None = None
        self.final_report: pb2.DoorbellFinalReport | None = None
        #: Set once the doorbell's end has begun (by a notice
        #: or by `end_session_for_doorbell`), so no second cause restarts it;
        #: ``doorbell_end`` names the cause that won, ``doorbell_end_reason``
        #: its `DoorbellShutdownReason` -- ``PEER_CLOSED`` for a hang-up
        #: until a `DoorbellFault` inside the window upgrades it to
        #: ``PEER_FAULT`` (`doorbell_protocol.md` §10).
        #: ``doorbell_shutdowns`` is every `DoorbellShutdown` sent, in order.
        self._doorbell_ending = False
        self.doorbell_end: str | None = None
        self.doorbell_end_reason: int | None = None
        self.doorbell_shutdowns: list[pb2.DoorbellShutdown] = []
        #: The plugin's own `DoorbellFault` reports.
        self.plugin_doorbell_faults: list[pb2.DoorbellFault] = []
        #: Messages of a kind this emulator does not know, by id
        #: ("" for a push).
        self.unknown_kinds: list[str] = []
        #: Ids are never reused: the plugin's requests and its replies, each
        #: by id, and the re-sends dropped (id, kind).
        self._ids = _IdLedger()
        self.deduplicated: list[tuple[str, str]] = []
        #: Why this emulator faulted the session on a protocol error (a
        #: first message that is not `Ready`, an id reused with different
        #: content); empty while the session is sound.
        self.session_faults: list[str] = []
        #: §8.16: every `WorkProgress` as the host kept it (title
        #: and text cut to 256 bytes), in arrival order; the latest per
        #: subject, keyed ``("request" | "task", id)``; the reports dropped
        #: because the host was not waiting on the request they name; and
        #: the tasks past §9's 64 that the host does not show.
        self.work_reports: list[pb2.WorkProgress] = []
        #: §8.8: the asked edits this session began, by
        #: `drag_id` (`ShapeDrag`, live or ended), and every `shape_proposal`
        #: and `shape_answer` as received.
        self.shape_drags: dict[int, "ShapeDrag"] = {}
        self.shape_proposals: list[pb2.ShapeResult] = []
        self.shape_answers: list[pb2.ShapeResult] = []
        self._drag_seq = 0
        self.work: dict[tuple[str, str], pb2.WorkProgress] = {}
        self.work_dropped: list[str] = []
        self.tasks_unshown: list[str] = []
        self._work_changed = threading.Condition()
        self.stream_ended = threading.Event()
        self.stream_error: Exception | None = None
        self.doorbell_closed = threading.Event()
        self._shutdown_sent = False
        self._plugin: subprocess.Popen | None = None

    # -- lifecycle --

    def add(self, segment: Segment) -> Segment:
        with self._lock:
            self.segments[segment.segment_id] = segment
            if segment.purpose == "G" and self._last_reader is not None:
                # `reader_writer_contract.md` §1.1: the `G` granted after an
                # open is THAT reader's arena, and its replies must name it.
                self.reader_arenas[self._last_reader] = segment.segment_id
        return segment

    def _manifest_at_this_minor(self) -> dict:
        """The manifest as the reference parser should read it from a host at
        :attr:`protocol`.  A manifest whose `engines.vizi` minor is above this
        host's is refused, before launch, with the host's sentence.  The
        parser itself speaks the SDK's minor, so an `engines` this host
        accepts but the parser would not (only when this host presents a
        newer minor) is read as the SDK's own for the rest of the checks."""
        engines = self.manifest.get("engines") if isinstance(self.manifest, dict) else None
        spec = engines.get("vizi") if isinstance(engines, dict) else None
        if not isinstance(spec, str):
            return self.manifest
        try:
            declared = _parse_engine_min(spec, "manifest")
        except ManifestError:
            return self.manifest  # the parser says why
        if declared[0] != self.protocol[0]:
            return self.manifest
        if declared[1] > self.protocol[1]:
            name = str(self.manifest.get("name") or self.manifest.get("id", ""))
            raise ManifestError(engines_message(name, str(self.manifest.get("version", "")), spec, self.protocol))
        if declared[1] > PROTOCOL_MINOR:
            return {**self.manifest, "engines": {**engines, "vizi": f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"}}
        return self.manifest

    def __enter__(self) -> "Host":
        # The one socket type each platform has (`doorbell_protocol.md` §3):
        # SEQPACKET on Linux, DGRAM on macOS -- which has no AF_UNIX
        # SEQPACKET at all -- with the plugin end's SO_RCVBUF raised before
        # inheritance, as the host does (§11: the stock 32-record depth is
        # below the 40 ms pump interval).  The SDK checks SO_TYPE against its
        # platform, so a Linux shape here would be refused on a Mac.
        sock_type = socket.SOCK_DGRAM if sys.platform == "darwin" else socket.SOCK_SEQPACKET
        self._host_db, child_db = socket.socketpair(socket.AF_UNIX, sock_type)
        if sys.platform == "darwin":
            child_db.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
        env = scrubbed_environment(dict(os.environ))
        env.pop("VIZI_PLUGIN_SETTINGS_DIR", None)
        if self.settings_dir is not None:
            # named, never created (§2.3): the SDK makes it on first use
            env["VIZI_PLUGIN_SETTINGS_DIR"] = str(self.settings_dir)
        env["VIZI_PLUGIN_UDS"] = self._uds
        env["VIZI_PLUGIN_DOORBELL_FD"] = str(child_db.fileno())
        # `entry.command` runs with the plugin directory as its working
        # directory (`manifest_schema.md` §2.2), so a relative interpreter
        # path resolves there -- Popen looks a relative executable up against
        # `cwd` on POSIX, which is the rule the host applies.
        self._plugin = subprocess.Popen(
            self.command,
            cwd=str(self.plugin_dir),
            env=env,
            pass_fds=(child_db.fileno(),),
        )
        child_db.close()

        self.launch_deadline = time.monotonic() + self.launch_budget
        while not Path(self._uds).exists():
            code = self._plugin.poll()
            if code is not None:
                raise RuntimeError(f"the plugin exited (code {code}) before it was ready (connecting)")
            if time.monotonic() > self.launch_deadline:
                self._kill_launch()
                raise RuntimeError(
                    f"did not bind {Path(self._uds).name} within {self.launch_budget:g}s "
                    "(the host's launch budget, spawn to Ready, §9)"
                )
            time.sleep(0.02)

        # `:authority`: the host sends `localhost` on both transports
        # (`control_plane_protocol.md` §2.1), and so
        # does this emulator.  Left to itself grpcio would derive one from the
        # unix:// target -- the socket path, percent-encoded -- which grpc-go,
        # grpc-java and grpcio's own server accept and the Rust `http` crate
        # refuses before the request reaches the SDK.
        # `VIZI_HOSTSIM_AUTHORITY` overrides the
        # value; set it empty to get grpcio's derivation back.
        # §9's 32 MiB message ceiling, as the host decodes it; grpcio's
        # default is 4 MiB.
        options = [("grpc.max_receive_message_length", MAX_MESSAGE_BYTES)]
        authority = os.environ.get("VIZI_HOSTSIM_AUTHORITY", "localhost")
        if authority:
            options.append(("grpc.default_authority", authority))
        self._channel = grpc.insecure_channel(f"unix://{self._uds}", options=options)
        stub = pb2_grpc.PluginStub(self._channel)
        self._responses = stub.Session(iter(self._outbound.get, None))
        self._pump = threading.Thread(target=self._read_loop, daemon=True, name="hostsim-grpc")
        self._pump.start()
        self._bell = threading.Thread(target=self._doorbell_loop, daemon=True, name="hostsim-bell")
        self._bell.start()
        major, minor = self.protocol
        # `session_token` is empty on unix: a UDS path already carries
        # filesystem permissions, and the token exists for the transports whose
        # endpoint any local process can reach.
        self.send(pb2.HostMsg(id=HELLO_ID, hello=pb2.Hello(protocol_major=major, protocol_minor=minor)))
        return self

    def shutdown(self, reason=None, deadline_ms: int = 5000, detail: str = "hostsim is finished") -> bool:
        """§3.3, as the host: send ``Shutdown`` and wait for the plugin to close.

        Closing the stream is what "done" means, so that is what is waited
        for -- not the process.  Returns whether the stream ended inside the
        window.
        """
        if not self._shutdown_sent:
            self._shutdown_sent = True
            self.send(
                pb2.HostMsg(
                    shutdown=pb2.Shutdown(
                        reason=pb2.SHUTDOWN_REASON_HOST_EXIT if reason is None else reason,
                        deadline_ms=deadline_ms,
                        detail=detail,
                    )
                )
            )
        return self.stream_ended.wait(deadline_ms / 1000.0)

    def __exit__(self, *_exc) -> None:
        if self.refusals:
            # Every host->plugin refusal this session sent, so a green run
            # that provoked a `panel_patch_refused` (a patch before its panel
            # was opened, say) still says so where a reader will see it.
            sites: dict[str, int] = {}
            for site, _detail in self.refusals:
                sites[site] = sites.get(site, 0) + 1
            print(
                "hostsim: refusals sent to the plugin: " + ", ".join(f"{n} x {s}" for s, n in sorted(sites.items())),
                file=sys.stderr,
            )
        if self.v_rings:
            # What the host-shaped `V` put this plugin through, so a green run
            # says whether growth and a bare reply actually happened.
            print(
                f"hostsim: V granted the host's way: {len(self.v_rings)} ring(s), "
                f"{self.bare_replies} bare reply(ies) into a pending ring, "
                f"{len(self.v_retired_on_drain)} outgrown ring(s) retired once drained",
                file=sys.stderr,
            )
        try:
            if not self.shutdown() and not self.stream_ended.is_set():
                # The host records a
                # plugin that does not close inside the window as having
                # OVERRUN, logs it, and kills it.
                self.diagnostics.append(
                    f"overran: the plugin did not close the stream within the {SHUTDOWN_WINDOW_S:g} s shutdown window"
                )
                print(f"hostsim: the plugin overran its {SHUTDOWN_WINDOW_S:g} s shutdown window", file=sys.stderr)
            self._outbound.put(None)
            # The plugin does not exit on its own: `serve()` runs until the
            # host reaps it, which is what §3.3 says happens next.
            self._plugin.terminate()
            try:
                self._plugin.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._plugin.kill()
        finally:
            self._channel.close()
            try:
                self._host_db.close()
            except OSError:
                pass
            for segment in self.segments.values():
                # A processing-stage ring is one mapping granted to every
                # session in its chain; the chain closes it, once.
                if not getattr(segment, "shared", False):
                    segment.close()
            self._tempdir.cleanup()

    # -- the two channels --

    def send(self, message: pb2.HostMsg) -> None:
        if message.id:
            # The host clears the request ledger as every reply leaves; a host REQUEST's id is simply not in it.
            with self._lock:
                self._open_requests.pop(message.id, None)
        self._outbound.put(message)

    def doorbell(self, op: int, payload: bytes, fd: int | None = None) -> int:
        """Send one record; returns its id."""
        return self._encode_and_send(op, payload, fd)[0]

    def _encode_and_send(self, op: int, payload: bytes, fd: int | None) -> tuple[int, bytes]:
        with self._db_lock:
            record_id = self._record_id
            record = encode_record(op, record_id, payload)
            self._record_id = (self._record_id + 1) & 0xFFFFFFFF
            self._send_record(record, fd)
        return record_id, record

    def notify(self, segment: int, regions) -> int:
        """A host -> plugin `D` naming ``regions`` of ``segment``, with no
        pane pair (`doorbell_D_notification.md` §1, §5): "slot *s* is at a
        position you are armed at" (`processing_stages.md` §6.2)."""
        regions = frozenset(regions)
        with self._lock:
            self.notifies.append((segment, regions))
        return self.doorbell(OP_NOTIFY, d_payload(segment, regions))

    def _send_record(self, record: bytes, fd: int | None) -> None:
        if fd is None:
            self._host_db.sendall(record)
        else:
            socket.send_fds(self._host_db, [record], [fd])

    def close_doorbell(self) -> None:
        """The host's end of the socket, once the window has closed."""
        try:
            self._host_db.close()
        except OSError:
            pass

    # -- grants --

    def grant_capability(self, segment: int, *, lost: bool = False) -> None:
        """The doorbell half: `M`, carrying one descriptor by `SCM_RIGHTS`.

        The perms follow the producer, as the `1.0` host's do
        -- and on Linux the host seals a host-produced mapping
        against plugin writes, which this emulator cannot check.

        ``lost=True`` numbers and records the `M` without sending it: a grant
        lost with the send reporting success, which `control_plane_protocol.md`
        §8.12 names as the case the grant backstop exists for.
        """
        seg = self.segments[segment]
        payload = m_payload(segment, seg.size, seg.perms)
        if lost:
            with self._db_lock:
                record_id = self._record_id
                self._record_id = (self._record_id + 1) & 0xFFFFFFFF
            seg.m_record = encode_record(OP_MEMORY, record_id, payload)
            return
        _, seg.m_record = self._encode_and_send(OP_MEMORY, payload, seg.fd)

    def resend_capability(self, segment: int) -> None:
        """The `not_present` row of `doorbell_M_memory.md` §7's table: the SAME
        `M` again -- same segment id, same payload, same doorbell `id` and so
        the same CRC -- byte for byte.  A fresh id would make the recovery
        register as a conflicting grant if the original is still undrained
        (`doorbell_protocol.md` §4).  The host re-sends the original
        record byte for byte, as this does."""
        seg = self.segments[segment]
        assert seg.m_record is not None, f"segment {segment} was never sent an M"
        self._send_record(seg.m_record, seg.fd)

    def new_segment_id(self) -> int:
        """An id for a segment the EMULATOR decides to grant -- a `V` ring
        it grows, a per-export `X` -- numbered from index 1000 so it never
        meets the fixed ids a driver chooses."""
        self._auto_index += 1
        return (self.session_epoch << 32) | self._auto_index

    # -- the grant path's backstop (`control_plane_protocol.md` §8.12) --

    def query_grant(self, segment: int, timeout: float = 10.0) -> pb2.GrantState:
        """`GrantStateQuery`, a one-way whose answer is a PUSH echoing its id."""
        with self._lock:
            self._seq += 1
            query_id = f"r{self._seq}"
        self.send(pb2.HostMsg(id=query_id, grant_state_query=pb2.GrantStateQuery(segment_id=segment)))
        return self._await(lambda: self.grant_states.get(query_id), timeout, f"no GrantState for segment {segment}")

    def recover_grant(self, segment: int, timeout: float = 10.0) -> pb2.GrantState:
        """What the host does on a missing acknowledgement: query, then do
        what `doorbell_M_memory.md` §7's table obliges for the answer --
        nothing for `installed`, `failed` or `released` (the answer IS the
        ack); for `pending`, send `InstallSegment` if that is the missing
        half and NEVER `M`; for `not_present`, re-send the SAME grant, both
        halves, `M` byte for byte.  Each action is kept in `recoveries`."""
        state = self.query_grant(segment, timeout)
        kind = state.state
        if kind == pb2.GRANT_STATE_NOT_PRESENT:
            self.resend_capability(segment)
            self.grant_install(segment)
            action = "re-sent both halves, M byte for byte"
        elif kind == pb2.GRANT_STATE_PENDING and state.have_capability and not state.have_install:
            self.grant_install(segment)
            action = "sent the missing InstallSegment"
        else:
            action = "nothing"
        with self._lock:
            self.recoveries.append((segment, _enum_name(pb2.GrantStateKind, kind), action))
        return state

    def grant_install(self, segment: int) -> None:
        """The control-plane half: the region table, as its own arrangement."""
        self.send(pb2.HostMsg(install_segment=self.segments[segment].install()))

    def grant(self, segment: int, timeout: float = 10.0) -> pb2.GrantAck:
        """Both halves, `M` first, then the acknowledgement."""
        self.grant_capability(segment)
        self.grant_install(segment)
        return self.await_ack(segment, timeout)

    def await_ack(self, segment: int, timeout: float = 10.0) -> pb2.GrantAck:
        return self._await(lambda: self.acks.get(segment), timeout, f"no GrantAck for segment {segment}")

    def retire_segment(self, segment: int, timeout: float = 10.0) -> pb2.SegmentReleased:
        """`RetireSegment`, then the plugin's `SegmentReleased`
        (`shared_memory_protocol.md` §9) -- what a reader sees at the end of
        its life instead of `CloseReader` (`reader_writer_contract.md` §1.8).
        Every loan inside must already be released (§9.3); the segment leaves
        the table and its object is closed here, as the host releases its
        lease."""
        seg = self.segments[segment]
        outstanding = [k for k in self.loan_uses if k[0] == segment]
        assert not outstanding, f"retiring segment {segment} with loans still lent: {outstanding}"
        with self._lock:
            self.retiring_segments.add(segment)
        self.send(pb2.HostMsg(retire_segment=pb2.RetireSegment(segment_id=segment)))
        released = self._await(lambda: self.retired.get(segment), timeout, f"no SegmentReleased for segment {segment}")
        with self._lock:  # `scan` snapshots the table under this lock
            self.segments.pop(segment, None)
            self.acks.pop(segment, None)
        seg.close()
        return released

    def supersede(self, old: int, replacement: Segment, timeout: float = 10.0, release_timeout: float = 2.0) -> pb2.GrantAck:
        """`shared_memory_protocol.md` §9.5's idiom: grant ``replacement``
        naming ``old`` in `InstallSegment.supersedes`, so the old segment
        retires WHEN the new one is installed -- never at receipt, never on
        a failed grant -- and wait for both the ack and the old segment's
        `SegmentReleased`.  This is how a chunked overlay's host-sized arena
        follows the dataset (`control_plane_protocol.md` §5.7, §8.12): the
        host empties the want set, re-grants at the new size, and names the
        new volumes.  Every loan in the old arena must already be released
        (§9.3), which a host that holds exactly what it wants has done by
        emptying the set."""
        outstanding = [k for k in self.loan_uses if k[0] == old]
        assert not outstanding, f"superseding segment {old} with loans still lent: {outstanding}"
        seg = self.segments[old]
        replacement.supersedes = old
        self.add(replacement)
        with self._lock:
            # Retiring from the moment the replacement can be installed.
            self.retiring_segments.add(old)
        ack = self.grant(replacement.segment_id, timeout)
        if ack.WhichOneof("outcome") != "installed":
            # A failed grant leaves the superseded segment live and unchanged.
            replacement.supersedes = 0
            with self._lock:
                self.retiring_segments.discard(old)
            return ack
        # §9: `supersedes` "behaves identically in every respect" to
        # `RetireSegment`, so the old segment's `SegmentReleased` is owed.  A
        # plugin that retires it silently is told so with the explicit
        # message -- which a host may send at any time -- and the lapse is
        # recorded in `diagnostics` for the scenario to judge.
        try:
            self._await(lambda: self.retired.get(old), release_timeout, f"no SegmentReleased for superseded segment {old}")
        except AssertionError:
            self.diagnostics.append(
                f"supersede: segment {old} sent no SegmentReleased within {release_timeout:g}s of "
                f"{replacement.segment_id} being installed; RetireSegment sent instead"
            )
            self.send(pb2.HostMsg(retire_segment=pb2.RetireSegment(segment_id=old)))
            self._await(lambda: self.retired.get(old), timeout, f"no SegmentReleased for segment {old} after RetireSegment")
        with self._lock:
            self.segments.pop(old, None)
            self.acks.pop(old, None)
        seg.close()
        return ack

    def _await(self, get, timeout: float, what: str):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                value = get()
            if value is not None:
                return value
            time.sleep(0.01)
        raise AssertionError(f"{what} within {timeout:g}s")

    # -- arrangements (§5.5, §8.10, §8.11) --

    def _engage(self, tool_id: str, *, triggers=(), actions=(), mode=pb2.TOOL_MODE_GESTURE, cancel_on_camera_move: bool = True) -> int:
        """Engage a tool as the host does, without waiting: RETIRE the
        engaged tool's binding if there is one (the displacement), then bind
        ``tool_id`` under a FRESH handle from the session's counter -- a
        handle is never reused (`doorbell_E_input.md` §5), so re-engaging the
        same tool is a new handle too.  Returns the handle; nothing is
        awaited, which is what lets `_on_activate_tool` bind BEFORE it replies
        from the reader thread, as the host does."""
        if self.engaged is not None:
            self._send_retirement(self.engaged)
        handle = self._next_tool
        self._next_tool += 1
        self.tools[tool_id] = handle
        self._tool_of_handle[handle] = tool_id
        self.engaged = tool_id
        wire_triggers = []
        for trig in triggers:
            if isinstance(trig, pb2.Trigger):
                wire_triggers.append(trig)
            elif isinstance(trig, str):
                wire_triggers.append(pb2.Trigger(key=trig, hold=pb2.TRIGGER_HOLD_TOGGLE))
            else:
                button, mods = trig
                wire_triggers.append(pb2.Trigger(button=button, mods=mods, hold=pb2.TRIGGER_HOLD_MOMENTARY))
        wire_actions = []
        for index, act in enumerate(actions, start=1):
            if isinstance(act, pb2.ToolAction):
                wire_actions.append(act)
            elif isinstance(act, str):
                wire_actions.append(pb2.ToolAction(action=index, action_id=act))
            else:
                action_id, key = act
                wire_actions.append(pb2.ToolAction(action=index, action_id=action_id, key=key))
        self.send(
            pb2.HostMsg(
                tool_binding=pb2.ToolBinding(
                    tool=handle,
                    tool_id=tool_id,
                    triggers=wire_triggers,
                    actions=wire_actions,
                    mode=mode,
                    cancel_on_camera_move=cancel_on_camera_move,
                    revision=1,
                )
            )
        )
        return handle

    def _send_retirement(self, tool_id: str) -> int:
        """Disengage: `retired` on the same message under the
        same handle, a new revision; the tool is no longer engaged.  Returns
        the handle retired."""
        handle = self.tools[tool_id]
        self.send(pb2.HostMsg(tool_binding=pb2.ToolBinding(tool=handle, tool_id=tool_id, retired=True, revision=2)))
        if self.engaged == tool_id:
            self.engaged = None
        return handle

    def bind_tool(
        self,
        tool_id: str,
        *,
        triggers=(),
        actions=(),
        mode=pb2.TOOL_MODE_GESTURE,
        cancel_on_camera_move: bool = True,
        timeout: float = 10.0,
    ) -> int:
        """Engage a tool (§8.10, modal) as the user does from
        the Tools menu: retire whatever is engaged, allocate a fresh handle,
        send `ToolBinding` revision 1, and wait for the `ToolBindingAck` (and
        the displaced tool's retirement ack).  Returns the handle.

        ``triggers`` are the ASSIGNED subset of what the manifest asked --
        `pb2.Trigger`s, key names (a `TOGGLE` key), or ``(button, mods)``
        tuples (a `MOMENTARY` chord); empty when the tool is engaged from the
        menu, which is the ordinary case.  No tool ever names the primary
        button.  ``actions`` are every declared action in manifest order,
        1-based: action ids, or ``(id, key)`` pairs for the key each kept.
        A `FIRE` with ``action=n`` (`host.fire(..., action=n)`) then names
        ``actions[n - 1]``.

        **Engaging displaces**:
        one tool is engaged at a time, so binding a second one --
        or the same one again -- retires the binding that was live.
        """
        displaced = self.tools.get(self.engaged) if self.engaged is not None else None
        handle = self._engage(
            tool_id, triggers=triggers, actions=actions, mode=mode, cancel_on_camera_move=cancel_on_camera_move
        )
        self._await(lambda: self.tool_acks.get(handle), timeout, f"no ToolBindingAck for {tool_id}")
        if displaced is not None:
            self._await(lambda: 2 if self.tool_acks.get(displaced) == 2 else None, timeout, f"no retirement ack for the displaced handle {displaced}")
        return handle

    def retire_tool(self, tool_id: str, timeout: float = 10.0) -> None:
        """Withdraw the handle (§8.10): the same message with `retired` set,
        revision advanced, never advancing the high-water mark.  The tool is
        no longer engaged."""
        handle = self._send_retirement(tool_id)
        self._await(lambda: 2 if self.tool_acks.get(handle) == 2 else None, timeout, f"no retirement ack for {tool_id}")

    def arrange_time(self, revision: int = 1, timeout: float = 10.0) -> None:
        """`TimeMapping`, acknowledged before any epoch names it (§8.11).

        Waits for the ack of THIS revision, and later `EpochChange`s name it.

        A dataset with no time increment has NO mapping: the `1.0` host
        arranges none and its epochs name revision 0 (absence is
        expressible).
        `arrange()` skips this call for such a dataset; calling it anyway
        is a driver asking for something the host never sends."""
        if not self.dataset.time_increment_s:
            raise AssertionError(
                "this dataset has no time increment, so the host arranges no TimeMapping and its "
                "epochs name revision 0; do not call arrange_time"
            )
        gen = self.dataset.gen
        self.send(pb2.HostMsg(time_mapping=self.dataset.time_mapping(revision)))
        self._await(
            lambda: revision if self.mapping_acks.get(gen) == revision else None,
            timeout,
            f"no TimeMappingAck for revision {revision}",
        )
        self.time_revisions[gen] = revision

    def arrange_pane(self, view_id: int, epoch: int | None = None, timeout: float = 10.0) -> int:
        """One `EpochChange` for one pane, as vizi sends them, naming the
        acknowledged mapping -- or revision 0 when none was arranged, the
        `1.0` host's value for a dataset without one; waits for the
        `EpochAck`.  Returns the epoch.

        The epoch comes from a PER-VIEW counter that is the process's, not the
        session's (`_PANE_EPOCHS`), as the host keeps one per host:
        it starts at 1, never restarts, never repeats -- so a second `Host`
        in one process continues the count, as a plugin launched into a host
        that has been running a while joins a chain already in progress.

        Refused for a session that is neither declaring a tool nor subscribed
        to ``"epochs"`` (§8.7, `manifest_schema.md` §5.1): the host does not arrange
        the chain for a session that never asked, and a port that expects an
        epoch without asking should find that out here rather than in vizi.
        """
        if not self.epochs_arranged:
            raise AssertionError(
                "this session declares no tool and no \"epochs\" subscription, so the host "
                "arranges no EpochChange for it (§8.7; subscriptions, §5.1)"
            )
        with _PANE_EPOCHS_LOCK:
            if epoch is None:
                epoch = _PANE_EPOCHS.get(view_id, 1)
            # a driver's explicit number (a stale or repeated one, on purpose)
            # never makes the counter go back
            _PANE_EPOCHS[view_id] = max(_PANE_EPOCHS.get(view_id, 1), epoch + 1)
        self.send(
            pb2.HostMsg(
                epoch_change=pb2.EpochChange(
                    datasets=[
                        pb2.DatasetState(
                            gen=self.dataset.gen,
                            geometry=self.dataset.geometry(),
                            time_mapping_revision=self.time_revisions.get(self.dataset.gen, 0),
                        )
                    ],
                    panes=[pb2.PaneEpoch(view_id=view_id, epoch=epoch, gen=self.dataset.gen)],
                )
            )
        )
        self._await(lambda: epoch if self.epoch_acks.get(view_id) == epoch else None, timeout, f"no EpochAck for pane {view_id}")
        self.epochs[view_id] = epoch
        return epoch

    def arrange(self, tool_id: str, panes=(VIEW_2D, VIEW_3D), **binding) -> int:
        """The sequence the `1.0` host runs before any `E`:
        `ToolBinding`, `TimeMapping` (none for a dataset
        with no time increment -- the host sends none and names revision 0),
        one `EpochChange` per pane.  Returns the tool handle."""
        handle = self.bind_tool(tool_id, **binding)
        if self.dataset.time_increment_s:
            self.arrange_time()
        for view_id in panes:
            self.arrange_pane(view_id)
        return handle

    # -- input (`doorbell_E_input.md`) --

    def _event(self, phase: int, tool: int, view_id: int, gesture: "_Gesture | None", **fields) -> int:
        if gesture is None:
            interaction, seq = self._next_interaction, 0
            self._next_interaction += 1
        else:
            gesture.seq += 1
            interaction, seq = gesture.interaction, gesture.seq
        fields.setdefault("t", self.dataset.t)
        fields.setdefault("channels", 1 << self.dataset.c)
        payload = e_payload(
            gen=self.dataset.gen,
            t=fields.pop("t"),
            interaction=interaction,
            seq=seq,
            epoch=self.epochs[view_id],
            tool=tool,
            channels=fields.pop("channels"),
            phase=phase,
            view_id=view_id,
            **fields,
        )
        self.doorbell(OP_INPUT, payload)
        return interaction

    def press(self, tool: int, view_id: int, position, **fields) -> "_Gesture":
        """Open a gesture; the returned handle carries `drag`/`release`/`cancel`."""
        interaction = self._event(PHASE_PRESS, tool, view_id, None, position=position, **fields)
        gesture = _Gesture(self, tool, view_id, interaction)
        self._open_interactions[interaction] = gesture
        return gesture

    def fire(self, tool: int, view_id: int, position, **fields) -> int:
        """A whole interaction in one record.  A FIRE of a
        tool that declares `view_targets` is a cause (§8.15), recorded by its
        `(tool, interaction)` as the host sends it."""
        tool_id = self._tool_of_handle.get(int(tool))
        spec = next((t for t in self.manifest.get("contributions", {}).get("tools", []) if t["id"] == tool_id), {})
        # The cause is minted BEFORE the record goes: a plugin
        # that answers at once must find it.  A gesture-less FIRE takes the
        # next interaction id, so it is known here.
        interaction = self._next_interaction
        if spec.get("view_targets") and self.dataset_open:
            self._forget_expired_causes()
            self._tool_causes[(int(tool), int(interaction))] = {
                "surface": None, "targets": frozenset(spec["view_targets"]), "stream": ("tool", int(tool)),
                "minted_at": time.monotonic(), "serial": self._touch_serial,
                "generation": int(self.dataset.open_generation), "superseded": False, "answer": None,
            }
        sent = self._event(PHASE_FIRE, tool, view_id, None, position=position, **fields)
        assert sent == interaction, (sent, interaction)
        return interaction

    # -- the doorbell's inbound side --

    def _doorbell_fault(self, what: str, failure_class: int = pb2.DOORBELL_FAILURE_UNSPECIFIED) -> None:
        """A terminal framing fault (`doorbell_protocol.md` §5, §8): the host
        tears the session down, so the emulator stops reading and records it
        in ``doorbell_faults`` and in ``unhandled`` -- a conformant plugin
        never causes one, and every driver's ``not host.unhandled`` says so."""
        with self._lock:
            self.doorbell_faults.append(what)
        self.unhandled.append(f"doorbell fault: {what}")
        self._end_session_soon(what, reason=pb2.DOORBELL_SHUTDOWN_FRAMING_FAULT, failure_class=failure_class)

    def _doorbell_loop(self) -> None:
        # One byte more than a record, and room for two descriptors, so an
        # oversize datagram, an attached descriptor and a truncated control
        # message are all SEEN (§8 step 1, §7.4) rather than cut to fit.
        ancillary = socket.CMSG_SPACE(2 * array.array("i").itemsize)
        try:
            while True:
                raw, ancdata, flags, _ = self._host_db.recvmsg(RECORD_SIZE + 1, ancillary)
                fds: list[int] = []
                for level, kind, data in ancdata:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        fds.extend(array.array("i", data[: len(data) - len(data) % array.array("i").itemsize]))
                for fd in fds:  # every exit path closes what arrived (§7.4)
                    os.close(fd)
                if raw == b"" and not fds and not flags & socket.MSG_CTRUNC and sys.platform != "darwin":
                    # A real hang-up: no bytes AND no ancillary data.
                    # A 0-byte read carrying a descriptor or
                    # MSG_CTRUNC is a FRAME, judged below (ctrunc, then
                    # wrong_fd_count, then bad_length).  A hang-up the host
                    # did not start ends the
                    # session.  Linux only: on a macOS DGRAM doorbell a
                    # peer close is not observable and an EMPTY datagram is a
                    # frame of the wrong length -- `bad_length` on the host, judged below.
                    # The window opens
                    # as `peer_closed` (doorbell §10), not
                    # `local_transport_fault` -- the host's own transport did
                    # not fail -- and a `DoorbellFault` arriving inside it is
                    # answered `peer_fault` and names the ending.
                    if not self._doorbell_ending and not self._shutdown_sent:
                        self._end_session_soon("the plugin hung up its doorbell", reason=pb2.DOORBELL_SHUTDOWN_PEER_CLOSED)
                    break
                if flags & socket.MSG_CTRUNC:
                    self._doorbell_fault("ctrunc", pb2.DOORBELL_FAILURE_CTRUNC)
                    break
                if len(fds) > 1:
                    self._doorbell_fault(f"{len(fds)} descriptors on one record", pb2.DOORBELL_FAILURE_WRONG_FD_COUNT)
                    break
                if flags & socket.MSG_TRUNC or len(raw) != RECORD_SIZE:
                    self._doorbell_fault(f"bad_length: {len(raw)} bytes", pb2.DOORBELL_FAILURE_BAD_LENGTH)
                    break
                # §8's order after the length, with the class the host
                # reports: the CRC, then `ver`, `flags`, `reserved`.
                (crc,) = struct.unpack_from("<I", raw, RECORD_SIZE - 4)
                if crc != zlib.crc32(raw[: RECORD_SIZE - 4]):
                    self._doorbell_fault("bad CRC", pb2.DOORBELL_FAILURE_BAD_CRC)
                    break
                header_fault = _header_fault(raw)
                if header_fault is not None:
                    self._doorbell_fault(*header_fault)
                    break
                op, record_id, payload = decode_record(raw)
                if op == 0xFF:
                    self._doorbell_fault("reserved_op", pb2.DOORBELL_FAILURE_RESERVED_OP)
                    break
                if op in (OP_MEMORY, OP_INPUT):
                    # §2: M and E are host->plugin forever; the other way is a
                    # session fault, not something to skip.
                    self._doorbell_fault(f"op {op:#x} in an illegal direction", pb2.DOORBELL_FAILURE_ILLEGAL_DIRECTION)
                    break
                if op != OP_NOTIFY:
                    # §5: an op this version does not define is dropped and
                    # counted, never a fault -- and so never ``unhandled``.
                    with self._lock:
                        self.skipped_ops.append(op)
                    continue
                if fds:
                    self._doorbell_fault("wrong_fd_count: a D carries no descriptor", pb2.DOORBELL_FAILURE_WRONG_FD_COUNT)
                    break
                try:
                    hint = decode_d(record_id, payload)
                except ValueError as exc:
                    self._doorbell_fault(f"D {record_id}: {exc}", pb2.DOORBELL_FAILURE_BAD_PADDING)
                    break
                if hint.segment == 0:
                    # `D` §4's one transport-checkable content violation: the
                    # record is DROPPED and the host tells the plugin with
                    # `ContractViolation{zero_id}`.
                    self.refuse("zero_id", f"D {record_id} names segment 0")
                    continue
                with self._lock:
                    self.hints.append(hint)
                    granted = hint.segment in self.segments
                    if not granted:
                        # D §4: a hint naming an ungranted segment -- 0, or
                        # one retired a moment ago -- is counted and the
                        # session goes on.
                        self.stray_hints.append(hint)
                if granted and self.scan_on_hint:
                    self.scan(hint.segment, hint.regions)
                if granted and self.on_hint is not None:
                    self.on_hint(self, hint)
        except OSError:
            pass
        finally:
            self.doorbell_closed.set()

    def scan(self, segment: int | None = None, regions=None) -> list[LiveValue]:
        """Read every lane (or the named ones) as the host's display-rate scan
        would, consuming what is new and applying §3.2's pane rule."""
        found: list[LiveValue] = []
        # The table is snapshotted
        # under the lock `retire_segment` deletes under, and a segment retired
        # between the snapshot and its read -- its mapping closed -- is
        # skipped, as a hint naming a just-retired segment is (D §4).
        with self._lock:
            if segment is not None:
                targets = [self.segments[segment]] if segment in self.segments else []
            else:
                targets = list(self.segments.values())
        for seg in targets:
            for region in seg.regions:
                if not isinstance(region, PulsedLane):
                    continue
                if regions is not None and region.index not in regions:
                    continue
                try:
                    got = region.read_newest()
                except (ValueError, OSError):
                    with self._lock:
                        still = self.segments.get(seg.segment_id) is seg
                    if still:
                        raise
                    continue  # retired under us; nothing to read
                if got is None:
                    continue
                revision, payload, pane = got
                found.append(self._apply_live(seg, region, revision, payload, pane))
        with self._lock:
            self.live.extend(found)
        return found

    def _apply_live(self, seg: Segment, lane: PulsedLane, revision: int, payload: bytes, pane) -> LiveValue:
        update = pb2.OverlayUpdate()
        try:
            update.ParseFromString(payload)
        except Exception as exc:  # noqa: BLE001
            return LiveValue(seg.segment_id, lane.index, revision, None, pane, False, f"undecodable: {exc}")
        # §3.3's two rules, as the receiver.
        for buf in _geom_buffers(update):
            if buf.WhichOneof("source") != "inline":
                return LiveValue(seg.segment_id, lane.index, revision, update, pane, False, "live_geometry_not_inline")
        if update.overlay_id != seg.subject:
            return LiveValue(seg.segment_id, lane.index, revision, update, pane, False, "lane_subject_mismatch")
        # A value carrying a pair is applied only
        # while that pair is the pane's current epoch.
        if pane is not None and self.epochs.get(pane[0]) != pane[1]:
            lane.stale_pairs += 1
            return LiveValue(seg.segment_id, lane.index, revision, update, pane, False, "stale_pane")
        return LiveValue(seg.segment_id, lane.index, revision, update, pane, True)

    def doorbell_released_by_peer(self, timeout: float) -> "bool | None":
        """Whether the plugin's end of the doorbell closed: on `SEQPACKET` a
        peer close reads as an empty datagram, which the reader thread sees.
        On a `DGRAM` doorbell (macOS) a peer close is not observable at all
        -- measured on real hardware, and why the `DoorbellShutdown` notice
        is the stop signal there -- so the answer is ``None``: unknowable,
        not "still open"."""
        if sys.platform == "darwin":
            return None
        return self.doorbell_closed.wait(timeout)

    def doorbell_shutdown(
        self,
        reason: int = pb2.DOORBELL_SHUTDOWN_FRAMING_FAULT,
        failure_class: int = pb2.DOORBELL_FAILURE_BAD_CRC,
        detail: str = "hostsim pretends record 9 failed its CRC",
    ) -> None:
        """The NOTICE alone: `DoorbellShutdown` with `deadline_ms = 2000`,
        emission stopped at once.  The host never stops there -- see
        `end_session_for_doorbell`, which
        is what a scenario wants; this is the first step of it."""
        notice = pb2.DoorbellShutdown(reason=reason, failure_class=failure_class, deadline_ms=DOORBELL_WINDOW_MS, detail=detail)
        with self._lock:
            self._doorbell_ending = True
            self.doorbell_shutdowns.append(notice)
        self.send(pb2.HostMsg(doorbell_shutdown=notice))

    def end_session_for_doorbell(
        self,
        detail: str,
        *,
        reason: int = pb2.DOORBELL_SHUTDOWN_FRAMING_FAULT,
        failure_class: int = pb2.DOORBELL_FAILURE_UNSPECIFIED,
    ) -> bool:
        """`control_plane_protocol.md` §8.13, `doorbell_protocol.md`
        §10): **a doorbell that closes ends its session, politely.**  Whatever
        closed it -- the plugin's `DoorbellFault`, a framing fault this host
        saw, a hang-up -- the host sends `DoorbellShutdown`, waits out the 2 s
        window (a `DoorbellFinalReport` closes it early), releases its end,
        and sends `Shutdown{session_fault}` naming the doorbell's failure.  A
        plugin does not end the session itself; it ends on that `Shutdown`.

        **A hang-up opens the window as `peer_closed`** (§10;
        `local_transport_fault` would say the HOST's transport
        failed).  A faulting plugin closes its socket and reports on the
        control plane, and the two race, so the window is what lets the
        report land: a `DoorbellFault` inside a `peer_closed` window is
        answered `peer_fault` -- `doorbell_end_reason` becomes
        ``PEER_FAULT`` and the `Shutdown{session_fault}` detail names the
        plugin's report, not the hang-up.  No second `DoorbellShutdown` is
        sent: the doorbell is already closed and the notice went out once.

        The emulator calls this itself on each of those causes; a scenario
        calls it to drive the path.  Returns whether the stream ended inside
        the shutdown window.  Idempotent: the first cause wins, except that
        upgrade."""
        with self._lock:
            if self._doorbell_ending or self._shutdown_sent:
                if reason == pb2.DOORBELL_SHUTDOWN_PEER_FAULT and self.doorbell_end_reason == pb2.DOORBELL_SHUTDOWN_PEER_CLOSED:
                    self.doorbell_end = detail
                    self.doorbell_end_reason = reason
                return self.stream_ended.is_set()
            self._doorbell_ending = True
            self.doorbell_end = detail
            self.doorbell_end_reason = reason
        notice = pb2.DoorbellShutdown(reason=reason, deadline_ms=DOORBELL_WINDOW_MS, detail=detail)
        if reason in (pb2.DOORBELL_SHUTDOWN_FRAMING_FAULT, pb2.DOORBELL_SHUTDOWN_PEER_FAULT, pb2.DOORBELL_SHUTDOWN_LOCAL_TRANSPORT_FAULT):
            # the IDL: present for these, absent otherwise (peer_closed has none)
            notice.failure_class = failure_class
        with self._lock:
            self.doorbell_shutdowns.append(notice)
        self.send(pb2.HostMsg(doorbell_shutdown=notice))
        try:
            self.await_final_report(DOORBELL_WINDOW_MS / 1000.0)
        except AssertionError:
            pass  # the window closed without a report; the host goes on regardless
        self.close_doorbell()
        # `doorbell_end`, not `detail`: a hang-up upgraded by the plugin's own
        # report during the window is named by the report.
        return self.shutdown(reason=pb2.SHUTDOWN_REASON_SESSION_FAULT, detail=f"doorbell: {self.doorbell_end}")

    def _end_session_soon(self, detail: str, **why) -> None:
        """`end_session_for_doorbell` off the calling thread -- the doorbell
        reader and the gRPC reader must not block for the 2 s window."""
        threading.Thread(target=self.end_session_for_doorbell, args=(detail,), kwargs=why, daemon=True).start()

    def request_unknown_kind(self, timeout: float = 10.0) -> pb2.PluginMsg:
        """A `HostMsg` whose only arm is a
        field number the plugin does not know, with an id.  The plugin must answer
        `Error{UNSUPPORTED}` echoing it; silence costs the sender its timeout."""
        with self._lock:
            self._seq += 1
            request_id = f"u{self._seq}"
            waiter: queue.Queue = queue.Queue()
            self._waiters[request_id] = waiter
        unknown_field = bytes([0xBA, 0x3E, 0x00])  # field 999, length-delimited, empty
        self.send(pb2.HostMsg.FromString(pb2.HostMsg(id=request_id).SerializeToString() + unknown_field))
        try:
            return waiter.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f"no answer to a message kind the plugin predates within {timeout:g}s") from None

    def send_raw_doorbell(self, data: bytes, fds=()) -> None:
        """Bytes on the doorbell exactly as given, with ``fds`` attached --
        for the framing faults a well-behaved host never sends (a 0-byte
        record carrying a descriptor is a frame, not a hang-up)."""
        if fds:
            socket.send_fds(self._host_db, [data], list(fds))
        else:
            self._host_db.send(data)

    def await_final_report(self, timeout: float) -> pb2.DoorbellFinalReport:
        return self._await(lambda: self.final_report, timeout, "no DoorbellFinalReport")

    # -- the control plane's host-initiated messages --

    def open_panel(self, panel_id: str, grant: "int | Segment | None" = None) -> None:
        """§8.4: panel-open is a control-plane message.
        From here on the panel is LIVE for this session and a declared
        panel's held state is applied.

        ``grant`` is the host's order:
        the panel's `P` -- a segment id already added, or a
        `Segment` to add -- is granted, both halves, and `PanelOpened` follows
        in the same call WITHOUT waiting for the plugin's `GrantAck`.  So a
        plugin can see `PanelOpened` before it has installed `P`, and an image
        it flushes only on `PanelOpened` meets no segment.  Without
        ``grant`` the driver grants `P` itself, usually waiting for the ack
        first -- an order vizi never produces."""
        if grant is not None:
            if isinstance(grant, Segment):
                self.add(grant)
                grant = grant.segment_id
            self.grant_capability(grant)
            self.grant_install(grant)
        self._mark_panel_open(panel_id)
        self.hidden_panels.discard(panel_id)
        # vizi: the switch is off at every open.
        if panel_id in self._switch_panels():
            self.set_view_toggle(panel_id, False)
        self.send(pb2.HostMsg(panel_opened=pb2.PanelOpened(panel_id=panel_id)))
        self._send_all_geometry(panel_id)

    def reprovision_panel(self, panel_id: str, grant: "int | Segment", *, visible: bool = True) -> None:
        """vizi's crash relaunch:
        the dead session's panel stays on screen, and the
        relaunched session is granted its `P`.  Run it on a fresh `Host` (the
        relaunched session): the panel is live here from the grant, so its
        patches apply.

        ``visible`` is the dock: vizi posts `PanelOpened` right after the
        grant, without waiting for its `GrantAck`, when the dock is not
        hidden, and
        nothing when it is.  So ``visible=False`` is the case where a plugin
        that waits for `PanelOpened` before sending anything leaves the panel
        as the dead process left it."""
        if isinstance(grant, Segment):
            self.add(grant)
            grant = grant.segment_id
        self.grant_capability(grant)
        self.grant_install(grant)
        self._mark_panel_open(panel_id)
        if visible:
            self.send(pb2.HostMsg(panel_opened=pb2.PanelOpened(panel_id=panel_id)))
            self._send_all_geometry(panel_id)

    # -- WidgetGeometry (§8.4) --

    def _image_widgets(self, panel_id: str) -> list[str]:
        """The panel's `image` widgets, in declaration order -- not a list's
        image cells, which are not reported (§8.4; vizi sizes the widgets its
        builder registers, and a cell is a row's, not the panel's)."""
        found: list[str] = []

        def walk(node: dict) -> None:
            if node.get("kind") == "image" and "id" in node:
                found.append(str(node["id"]))
            for child in node.get("children", []):
                walk(child)

        panel = next((p for p in self.manifest.get("contributions", {}).get("panels", []) if p.get("id") == panel_id), {})
        walk(panel.get("root", {}))
        return found

    def _send_all_geometry(self, panel_id: str) -> None:
        """As the host tells a panel's geometry: right
        after `PanelOpened`, every image widget's size, sent whatever was sent
        before (a new session, or one that missed them while starting)."""
        for widget_id in self._image_widgets(panel_id):
            self._geometry_last.pop((panel_id, widget_id), None)
            self._send_geometry(panel_id, widget_id)

    def _send_geometry(self, panel_id: str, widget_id: str) -> None:
        """As the host sends one widget's geometry: nothing while the dock is hidden, and
        only a change -- the host's half of the coalescing (it also waits for
        a resize to hold still for 100 ms, which a driver's `resize` already
        has)."""
        if panel_id in self.hidden_panels:
            return
        geometry = self.image_geometry.get((panel_id, widget_id), DEFAULT_IMAGE_GEOMETRY)
        if self._geometry_last.get((panel_id, widget_id)) == geometry:
            return
        message = pb2.WidgetGeometry(
            panel_id=panel_id, widget_id=widget_id, width_px=geometry[0], height_px=geometry[1], dpr=geometry[2]
        )
        self._geometry_last[(panel_id, widget_id)] = geometry
        self.geometry_sent.append(message)
        self.send(pb2.HostMsg(widget_geometry=message))

    def resize(self, panel_id: str, widget_id: str, width_px: int, height_px: int, dpr: float = 1.0) -> bool:
        """The person resizes the dock (or moves it to a screen of another
        pixel ratio) and ``widget_id`` -- an `image` -- is drawn ``width_px``
        by ``height_px`` device-independent pixels at ``dpr``.  Once the panel
        is open and showing, the host sends `WidgetGeometry` if that differs
        from what it last sent; returns whether it sent.  Before the panel
        opens the size is only remembered, and goes with the open."""
        if widget_id not in self._image_widgets(panel_id):
            raise ValueError(f"panel {panel_id!r} has no image widget {widget_id!r}")
        if width_px < 0 or height_px < 0 or not dpr > 0:
            raise ValueError("a widget is at least 0 by 0 pixels, at a positive pixel ratio")
        self.image_geometry[(panel_id, widget_id)] = (int(width_px), int(height_px), float(dpr))
        if panel_id not in self.opened_panels:
            return False
        before = len(self.geometry_sent)
        self._send_geometry(panel_id, widget_id)
        return len(self.geometry_sent) > before

    # -- HostAppearance (§8.4) --

    def _send_appearance(self) -> bool:
        """As the host sends its appearance: a coalescing push keyed by the
        session, sent only when its bytes differ from the last one sent."""
        if self.theme is None:
            return False
        message = pb2.HostAppearance(theme=self.theme, **APPEARANCE_PALETTES[self.theme])
        data = message.SerializeToString(deterministic=True)
        if data == self._appearance_last:
            return False
        self._appearance_last = data
        self.appearance_sent.append(message)
        self.send(pb2.HostMsg(appearance=message))
        return True

    def set_theme(self, theme: str) -> bool:
        """The person switches vizi's theme to ``"dark"`` or ``"light"``
        (`APPEARANCE_PALETTES` gives each one's colors).  Once the session is
        ready the host sends `HostAppearance` if it differs from what it last
        sent; returns whether it sent.  Before `Ready` the theme is only
        remembered, and goes with the first push."""
        if theme not in APPEARANCE_PALETTES:
            raise ValueError(f"theme is one of {sorted(APPEARANCE_PALETTES)}, not {theme!r}")
        self.theme = theme
        if self.ready is None:
            return False
        return self._send_appearance()

    def _mark_panel_open(self, panel_id: str) -> None:
        with self._lock:
            self.opened_panels.add(panel_id)
            held = self._held_patches.pop(panel_id, None)
        if held:
            # As the host applies held state: a list's rows first, so its cells
            # exist, then options, so a held value is set against the
            # latest options.  A table's one held set_table goes with the rows,
            # so its selection, held as any value, comes after it; built on
            # this side and never a plugin's message, it is under no message bound.
            order = {"set_rows": 0, "set_table": 0, "patch_table": 0, "set_options": 1}
            ops = sorted(held.values(), key=lambda op: order.get(op.WhichOneof("op"), 2))
            self._apply_panel_ops(panel_id, self._panel_widgets[panel_id], ops)

    def request(self, timeout: float = 20.0, **arm) -> pb2.PluginMsg:
        """One host request (§5.1): send it with a fresh id, return the
        plugin's reply whole.  An `Error` reply raises `PluginError`."""
        return self.request_async(**arm).result(timeout)

    def request_async(self, **arm) -> "PendingRequest":
        """Send a host request and return at once: the
        :class:`PendingRequest` carries its id, so a scenario can watch its
        `WorkProgress` and send `CancelWork` for it while it runs, then take
        the reply with ``.result(timeout)``."""
        self._seq += 1
        request_id = f"r{self._seq}"
        waiter: queue.SimpleQueue = queue.SimpleQueue()
        message = pb2.HostMsg(id=request_id, **arm)
        with self._lock:
            self._waiters[request_id] = waiter
            self._asked[request_id] = message.WhichOneof("msg")
            if message.HasField("invoke_command"):
                self._invoked[request_id] = message.invoke_command.command_id
        self.send(message)
        return PendingRequest(request_id, waiter)

    def invoke_async(self, command_id: str, **args) -> "PendingRequest":
        """`InvokeCommand` without waiting: see :meth:`request_async`."""
        typed = {key: _value(value) for key, value in args.items()}
        return self.request_async(invoke_command=pb2.InvokeCommand(command_id=command_id, args=typed))

    # -- long-running work (§8.16) --

    def cancel_work(self, *, request_id: str | None = None, task_id: str | None = None) -> None:
        """`CancelWork`, a host ONE-WAY (§5.2): it leaves at once, not behind
        the request it names, which keeps its slot until the plugin answers
        it -- `Error{CANCELLED}`, which is no fault, or whatever it had.
        Naming finished or unknown work is a no-op at the plugin."""
        if (request_id is None) == (task_id is None):
            raise ValueError("name exactly one of request_id and task_id")
        work = pb2.CancelWork(request_id=request_id) if request_id is not None else pb2.CancelWork(task_id=task_id)
        self.send(pb2.HostMsg(cancel_work=work))

    def stop_plugin(self, deadline_ms: int = 5000, detail: str = "the person stopped the plugin from work that did not answer a cancel") -> bool:
        """*Stop plugin*, which vizi offers when a cancel has gone unanswered
        for `CANCEL_ANSWER_WINDOW_S`: `Shutdown{USER_STOPPED}` (§3.3, §8.16).
        Returns whether the stream ended within the window plus a second's
        grace: the work that ignored the cancel is still running, and a
        plugin may spend its whole window letting it finish before it closes
        (the reference SDK does) -- vizi kills it at the window's end."""
        self.shutdown(reason=pb2.SHUTDOWN_REASON_USER_STOPPED, deadline_ms=deadline_ms, detail=detail)
        return self.stream_ended.wait(1.0)

    def await_work(self, key: tuple[str, str], predicate=lambda w: True, timeout: float = 10.0) -> pb2.WorkProgress:
        """Wait until the latest report for ``key`` (``("request", id)`` or
        ``("task", id)``) satisfies ``predicate``; returns it.  The latest,
        never a particular report: `WorkProgress` coalesces."""
        deadline = time.monotonic() + timeout
        with self._work_changed:
            while True:
                work = self.work.get(key)
                if work is not None and predicate(work):
                    return work
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no WorkProgress for {key} satisfying the condition within {timeout} s (latest: {work})")
                self._work_changed.wait(left)

    def _on_work_progress(self, work: "pb2.WorkProgress") -> None:
        """§8.16 as the host receives it: title and text cut to 256 bytes at a
        character boundary; a report for a request the host is not waiting on
        dropped (its answer overtook it); a task past §9's 64 not shown."""
        subject = work.WhichOneof("subject")
        if subject is None:
            self.diagnostics.append("a WorkProgress naming no request and no task")
            return
        ident = getattr(work, subject)
        kept = pb2.WorkProgress()
        kept.CopyFrom(work)
        kept.title = _clip_work_text(work.title)
        kept.text = _clip_work_text(work.text)
        with self._work_changed:
            if subject == "request_id":
                with self._lock:
                    waiting = ident in self._waiters
                if not waiting:
                    self.work_dropped.append(ident)
                    return
                key = ("request", ident)
            else:
                key = ("task", ident)
                open_tasks = [k for k, w in self.work.items() if k[0] == "task" and not w.done]
                if key not in self.work and len(open_tasks) >= MAX_OPEN_TASKS:
                    self.tasks_unshown.append(ident)
                    return
            self.work_reports.append(kept)
            self.work[key] = kept
            self._work_changed.notify_all()

    def invoke(self, command_id: str, timeout: float = COMMAND_WINDOW_S, **args) -> pb2.CommandDone:
        """`InvokeCommand`, awaiting `CommandDone` for an unsupervised
        command's 10 s (`COMMAND_WINDOW_S`, §8.3).  vizi supervises its commands and waits on them
        with no deadline (§8.16); `invoke_async` models that."""
        typed = {key: _value(value) for key, value in args.items()}
        return self.request(timeout, invoke_command=pb2.InvokeCommand(command_id=command_id, args=typed)).command_done

    # -- readers (§8.2): the plugin deposits into its `G` arena, the host
    # reads the loan and releases it --

    def file_state_for(self, uri: str) -> str:
        """The per-file state folder the host would name for ``uri``:
        `plugins/<id>/` in its companion directory (here under the temp dir,
        never created), or ``""`` for a URI that is not a local path."""
        if "://" in uri:
            return ""
        return str(self.directory / "companions" / (Path(uri).name + ".vizi") / "plugins" / plugin_folder_name(self.plugin_id))

    def open_reader(self, uri: str, reader_id: str = "", timeout: float = 30.0, file_state_dir: str | None = None, series: str | None = None) -> pb2.ReaderOpened:
        """`reader_writer_contract.md` §1.2's H1: sent while the previous
        reader is still open -- the host constructs the next dataset before
        it closes the last -- and from this instant nothing more is sent for
        the old one.  The plugin's H2 is open-then-close: `Error` means
        nothing changed.  A driver then grants the new `G` through
        `supersede(old_g, Segment.reader_arena(...))` (H3-H5): the old arena
        retires on the install, no `CloseReader`, no `RetireSegment`.

        ``series`` (§1.2a) is sent only when given -- presence is the
        signal, and ``""`` may be an id.  A `SeriesList` answer raises
        :class:`SeriesChoiceRequired`, as vizi's open does; it changed nothing
        (H2), except that a listing of the URI the open reader holds closed
        that reader first, so the host treats it as gone.  A list answered to
        a NAMED series is a reader defect, refused as the host refuses it.  So is a malformed list (no
        entries, a shape present but not five positive sizes, an id listed
        twice) -- the open fails with the reason and no choice is offered --
        and a `ReaderOpened` naming more channels than it has."""
        # The folder for the file being OPENED -- `file_state_for(uri)`
        # unless the driver names one ("" to send it empty).
        state = self.file_state_for(uri) if file_state_dir is None else file_state_dir
        request = pb2.OpenReader(uri=uri, reader_id=reader_id, file_state_dir=state)
        if series is not None:
            request.series = series
        reply = self.request(timeout, open_reader=request)
        kind = reply.WhichOneof("msg")
        if kind == "series_list":
            with self._lock:
                self.series_lists.append((uri, reply.series_list))
                for handle in [h for h, u in self.reader_uris.items() if u == uri]:
                    self.readers.pop(handle, None)
                    self.reader_uris.pop(handle, None)
            if series is not None:
                why = f"open: asked for series {series!r}, the reader answered with a list"
                self.diagnostics.append(f"reader reply refused: {why}")
                raise PluginError(pb2.Error.PROTOCOL, f"hostsim refuses the reply: {why}")
            why = series_list_problem(reply.series_list)
            if why is not None:
                self.diagnostics.append(f"reader reply refused: open: {why}")
                raise PluginError(pb2.Error.PROTOCOL, f"hostsim refuses the reply (§8.2, a malformed SeriesList): {why}")
            raise SeriesChoiceRequired(self, uri, reader_id, reply.series_list)
        if kind != "reader_opened":
            raise PluginError(pb2.Error.PROTOCOL, f"open: expected reader_opened, got {kind}")
        opened = reply.reader_opened
        why = reader_opened_problem(opened)
        if why is not None:
            self.diagnostics.append(f"reader reply refused: open: {why}")
            raise PluginError(pb2.Error.PROTOCOL, f"hostsim refuses the reply (§8.2): {why}")
        with self._lock:
            self.readers[opened.handle] = opened
            self.reader_uris[opened.handle] = uri
            self._last_reader = opened.handle
        return opened

    def _reader_arena(self, handle: str) -> int | None:
        """The `G` this reader's replies must name: the one granted after its
        open, else -- a reader reopened without a new grant -- the newest
        `G` this host holds."""
        with self._lock:
            arena = self.reader_arenas.get(handle)
            if arena is not None and arena in self.segments:
                return arena
            live = [sid for sid, seg in self.segments.items() if seg.purpose == "G"]
        return live[-1] if live else None

    def _read_reply(self, reply: pb2.PluginMsg, handle: str, op: str, shapes) -> ReadResult:
        """A reader's deposit reply, checked as the host checks it before it believes a byte: the segment is THIS reader's
        `G`, the dtype is the one `ReaderOpened` announced, a level-0 shape
        is the one it announced (``shapes``; a plane may be `(1, Y, X)` or
        `(Y, X)`), and the length is shape x dtype.  A reply that fails is
        REFUSED -- `PluginError(PROTOCOL)` to the driver, recorded in
        ``diagnostics`` -- and its loan disposed of either way (§5.6), as the
        host's `finally` does."""
        if reply.WhichOneof("msg") != "segment":
            raise PluginError(pb2.Error.PROTOCOL, f"{op}: expected segment, got {reply.WhichOneof('msg')}")
        ref = reply.segment
        why = None
        opened = self.readers.get(handle)
        arena = self._reader_arena(handle)
        width = {pb2.DTYPE_U8: 1, pb2.DTYPE_U16: 2, pb2.DTYPE_F32: 4}.get(ref.dtype)
        if width is None:
            why = f"segment has unsupported dtype {ref.dtype}"
        elif arena is not None and ref.segment_id != arena:
            why = f"the reply names segment {ref.segment_id}; this reader's arena is {arena}"
        elif opened is not None and ref.dtype != opened.dtype:
            why = f"the reply is {_enum_name(pb2.DType, ref.dtype)}; ReaderOpened announced {_enum_name(pb2.DType, opened.dtype)}"
        elif shapes is not None and tuple(ref.shape_zyx) not in shapes:
            why = f"the reply's shape {tuple(ref.shape_zyx)} is not {shapes[0]} by ReaderOpened"
        elif ref.len != int(np.prod(ref.shape_zyx)) * width:
            why = f"SegmentRef.len {ref.len} != shape x dtype {int(np.prod(ref.shape_zyx)) * width}"
        try:
            loan = self._take_loan(ref)
        except ValueError as exc:
            why = why or str(exc)
            loan = None
        if loan is not None:
            self.release_loan(loan)
        if why is not None:
            self.diagnostics.append(f"reader reply refused: {op}: {why}")
            raise PluginError(pb2.Error.PROTOCOL, f"hostsim refuses the reply: {op}: {why}")
        return ReadResult(ref, loan.payload)

    def _level0_shapes(self, handle: str, plane: bool):
        opened = self.readers.get(handle)
        if opened is None:
            return None
        _, _, z, y, x = opened.shape_tczyx
        return ((1, y, x), (y, x)) if plane else ((z, y, x),)

    def read_volume(self, handle: str, t: int, c: int, level: int = 0, timeout: float = 30.0) -> ReadResult:
        reply = self.request(timeout, read_volume=pb2.ReadVolume(handle=handle, t=t, c=c, level=level))
        return self._read_reply(reply, handle, "read_volume", self._level0_shapes(handle, False) if level == 0 else None)

    def read_plane(self, handle: str, t: int, z: int, c: int, level: int = 0, timeout: float = 30.0) -> ReadResult:
        reply = self.request(timeout, read_plane=pb2.ReadPlane(handle=handle, t=t, z=z, c=c, level=level))
        return self._read_reply(reply, handle, "read_plane", self._level0_shapes(handle, True) if level == 0 else None)

    def metadata(self, handle: str, timeout: float = 30.0) -> pb2.MetadataResult:
        return self.request(timeout, metadata=pb2.Metadata(handle=handle)).metadata

    def close_reader(self, handle: str, timeout: float = 30.0) -> pb2.Closed:
        return self.request(timeout, close_reader=pb2.CloseReader(handle=handle)).closed

    # -- exporters (§8.6): the host deposits into the `X` ring, the plugin
    # reads and releases with `ReleaseDeposit` --

    def grant_export_arena(self, payload_bytes: int) -> int:
        """The `1.0` host's `X`, one per export job: sized to ONE chunk, `M` and the
        standalone `InstallSegment` sent BEFORE `OpenExport`, and no
        acknowledgement waited for (`reader_writer_contract.md` §2.1) -- the
        receiver holds a chunk into a segment it has not installed yet.  Every
        chunk lands at the same offset: the host reuses the arena the moment
        `ChunkWritten` arrives (§2.3), so a plugin's `ReleaseDeposit` for it
        is honored but not waited for.  Sets `export_segment`; returns the
        id.  Pair with `retire_export_arena` when the job is done.

        The driver-granted fixed `X` (`Segment.host_ring` + `export_segment`)
        still works; the host grants a fresh `X` per job."""
        seg = Segment.host_ring(self.new_segment_id(), "X", Segment.arena_segment_size(payload_bytes), self.directory)
        self.add(seg)
        self._per_export_x.add(seg.segment_id)
        self.grant_capability(seg.segment_id)
        self.grant_install(seg.segment_id)
        self.export_segment = seg.segment_id
        return seg.segment_id

    def retire_export_arena(self, timeout: float = 10.0) -> pb2.SegmentReleased:
        """`RetireSegment` for the job's `X`, awaiting `SegmentReleased`.

        The host does the same when the job ends.  Retiring is
        legal at any time (§9), so a plugin must already handle it."""
        segment = self.export_segment
        assert segment in self._per_export_x, "no per-export X to retire"
        self.export_segment = None
        self._per_export_x.discard(segment)
        return self.retire_segment(segment, timeout)

    def open_export(self, exporter_id: str, locator: str, shape_tczyx, dtype: int, *, little_endian: bool = True, source_metadata: tuple[str, str] | None = None, spacing_zyx=None, spacing_unit: str = "", time_increment: float = 0.0, time_unit: str = "", timeout: float = 30.0, file_state_dir: str | None = None, filter: "pb2.AppliedFilter | None" = None, channels=None, provenance: "dict | str | None" = None) -> pb2.ExportOpened:
        """§8.3.  ``spacing_zyx`` / ``time_increment`` are the calibration the
        host KNOWS (fields 7-12, `1.0`); left unset they say *unknown*, which
        an exporter must not write as a measurement.  ``filter`` is
        `OpenExport.filter` (§8.6): the filter the export reads
        through, absent for the file's own voxels.

        ``source_metadata`` is the SOURCE's document
        ``(form, text)`` -- never the artifact's description.
        ``channels`` is the artifact's channels in its C
        order: ``None`` takes the open dataset's first C (:meth:`Dataset.
        channel_infos`, `FILE` or `PLACEHOLDER`), a sequence of source indices
        takes those, or give `ChannelInfo`s outright.  ``provenance`` is the
        host's provenance record: ``None`` builds one shaped as the host's,
        a dict is sent as JSON, a string as
        is (``""`` sends none)."""
        # The SOURCE's folder (the open dataset's), not the locator's.
        state = self.file_state_dir if file_state_dir is None else file_state_dir
        msg = pb2.OpenExport(exporter_id=exporter_id, locator=locator, shape_tczyx=list(shape_tczyx), dtype=dtype, little_endian=little_endian, file_state_dir=state)
        if source_metadata is not None:
            msg.source_metadata.CopyFrom(pb2.MetadataDoc(form=source_metadata[0], text=source_metadata[1]))
        infos = {info.index: info for info in self.dataset.channel_infos()}
        if channels is None:
            channels = range(int(shape_tczyx[1]))
        for ch in channels:
            if isinstance(ch, pb2.ChannelInfo):
                msg.channels.append(ch)
            else:
                i = int(ch)
                msg.channels.append(infos.get(i) or pb2.ChannelInfo(index=i, name=f"Channel {i + 1}", source=pb2.CHANNEL_NAME_SOURCE_PLACEHOLDER))
        if spacing_zyx is not None:
            msg.spacing_z, msg.spacing_y, msg.spacing_x = (float(v or 0.0) for v in spacing_zyx)
            msg.spacing_unit = spacing_unit or "um"
        if time_increment:
            msg.time_increment = float(time_increment)
            msg.time_unit = time_unit or "s"
        if filter is not None:
            msg.filter.CopyFrom(filter)
        if provenance is None:
            provenance = self._provenance_record(msg)
        msg.provenance = provenance if isinstance(provenance, str) else json.dumps(provenance, sort_keys=True)
        return self.request(timeout, open_export=msg).export_opened

    def _provenance_record(self, msg: "pb2.OpenExport") -> dict:
        """A provenance record for ``msg`` in vizi's shape: `data_export.
        build_manifest` as `plugins/exporter._provenance` sends it,
        from the open dataset and the
        message.  A channel the source does not name is ``""`` in
        ``channel_names``, never its ``"Channel N"`` placeholder; with a
        filter, ``filter_failed`` is ``null`` (not known at open) and each
        chunk's ``filtered_channels`` says what it met.  Where the emulator
        differs it is because its dataset is no file: the fingerprint is
        null, ``source.reader.kind`` is ``"emulated"``, and an export is
        always uncropped and unprojected."""
        d = self.dataset
        t, _, z, y, x = (int(n) for n in msg.shape_tczyx)
        psz, psy, psx = d.spacing_zyx if d.spacing_zyx is not None else (None, None, None)  # an unrecorded axis stays None

        def phys(n: int, p):
            return [0.0, n * p] if p is not None else None

        source = {
            "path": d.dataset_uri,
            "name": d.dataset_name,
            "size_bytes": None,
            "sha256": None,
            "shape_tczyx": list(d.shape_tczyx),
            "dtype": str(np.dtype(d.np_dtype)),
            "axis_order": "TCZYX",
            "physical_size_xyz": [psx, psy, psz],
            "time_increment_s": d.time_increment_s or None,
            "reader": {"kind": "emulated"},
        }
        if d.dataset_series is not None:
            source["series"] = d.dataset_series
        processing: dict = {
            "display_processing_applied": False,
            "note": "no display LUT / contrast / colormap applied — raw intensities",
            "dtype_promotion": None,
            "filters": [],
            "projection": None,
        }
        if msg.HasField("filter"):
            f = msg.filter
            processing["filters"] = [{
                "plugin": f.plugin_id,
                "version": f.plugin_version,
                "stage": f.stage_id,
                "params": param_values(f.params),
                "channels": sorted(f.channels),
            }]
            processing["selected_timepoints"] = list(range(t))
            processing["filter_failed"] = None
            processing["filter_failed_note"] = (
                "not known when this record was written (export start); each chunk's "
                "filtered_channels says whether the filter ran on it"
            )
        return {
            "manifest_version": 1,
            "software": {"name": "vizi_hostsim", "version": "1.0", "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())},
            "source": source,
            "selection": {
                "t_indices": list(range(t)),
                "channels": [c.index for c in msg.channels],
                "channel_names": ["" if c.source == pb2.CHANNEL_NAME_SOURCE_PLACEHOLDER else c.name for c in msg.channels],
                "crop_voxel": {"z": [0, z], "y": [0, y], "x": [0, x]},
                "crop_physical_xyz": {"x": phys(x, psx), "y": phys(y, psy), "z": phys(z, psz)},
            },
            "processing": processing,
            "output": {
                "format": f"plugin:{self.plugin_id}:{msg.exporter_id}",
                "shape_tczyx": list(msg.shape_tczyx),
                "dtype": str(np.dtype(d.np_dtype)),
            },
        }

    def export_chunk(self, handle: str, t: int, channels, volumes, *, inline: bool = False, dtype: int, timeout: float = 120.0, filtered_channels=()) -> pb2.ChunkWritten:
        """One chunk: `channels` volumes of one shape concatenated, either as
        a deposit into the `X` ring (the `1.0` host's shape) or `inline`.
        ``filtered_channels`` is `ExportChunk.filtered_channels` (§8.6):
        which of ``channels`` are the export filter's output.  A driver may
        send a contradictory one -- non-empty with no `OpenExport.filter`, or
        naming a channel not in ``channels`` -- which the host never does and
        a plugin refuses `INVALID_ARGUMENT`.

        An `Error` reply ends the export early, so the host sends
        `AbortExport` before `PluginError` is raised (§8.6):
        see :meth:`export_chunk_async`."""
        return self.export_chunk_async(handle, t, channels, volumes, inline=inline, dtype=dtype, filtered_channels=filtered_channels).result(timeout)

    def export_chunk_async(self, handle: str, t: int, channels, volumes, *, inline: bool = False, dtype: int, filtered_channels=()) -> "PendingChunk":
        """:meth:`export_chunk` without waiting: its ``id`` is the subject of
        a `CancelWork` (the activity row's Cancel, §8.16), and ``.result``
        takes the reply.  However the chunk ends in an `Error` --
        `CANCELLED`, refused or failed -- the host then sends `AbortExport`
        for the handle (§8.6: however an export ends early), recorded in
        :attr:`export_aborts` as ``(handle, error kind, ExportDone or the
        abort's own error kind)``.  A plugin that retired the handle itself
        answers that abort `NOT_FOUND`, which is no fault."""
        arrays = [np.ascontiguousarray(v) for v in volumes]
        payload = b"".join(a.tobytes() for a in arrays)
        msg = pb2.ExportChunk(handle=handle, t=t, channels=list(channels), filtered_channels=list(filtered_channels))
        if inline:
            msg.inline = payload
        else:
            seg = self.segments[self.export_segment]
            offset = seg.ring.deposit(payload)
            msg.segment.CopyFrom(pb2.SegmentRef(segment_id=seg.segment_id, offset=offset, len=len(payload), shape_zyx=list(arrays[0].shape), dtype=dtype))
            if seg.segment_id in self.ride_along:
                self.ride_along.discard(seg.segment_id)
                msg.install.CopyFrom(seg.install())
        release = None
        if not inline and seg.segment_id in self._per_export_x:
            release = (seg.ring, offset, len(payload))  # reused at ChunkWritten, §2.3
        return PendingChunk(self, handle, self.request_async(export_chunk=msg), release)

    def _abort_ended_export(self, handle: str, error: "PluginError") -> None:
        """§8.6: an export that ends early is aborted."""
        try:
            outcome = self.abort_export(handle)
        except PluginError as exc:
            outcome = exc.kind
        with self._lock:
            self.export_aborts.append((handle, error.kind, outcome))

    def finish_export(self, handle: str, timeout: float = 60.0) -> pb2.ExportDone:
        return self.request(timeout, finish_export=pb2.FinishExport(handle=handle)).export_done

    def abort_export(self, handle: str, timeout: float = 10.0) -> pb2.ExportDone:
        return self.request(timeout, abort_export=pb2.AbortExport(handle=handle)).export_done

    def release_loan(self, loan: Loan) -> None:
        """One disposition of ``loan`` (§5.6): decrement its declared uses and
        send the one `ReleaseLoan` when the count reaches zero."""
        key = (loan.segment, loan.offset, loan.length)
        declared = max(1, loan.uses)
        entry = self.loan_uses.setdefault(key, [declared, declared])
        if entry[0] != declared:
            # §5.6: every message naming a reservation carries the SAME count;
            # the first one given is applied and the disagreement is reported.
            self.diagnostics.append(f"uses_disagreement: reservation {key} declared {declared} after {entry[0]}")
        entry[1] -= 1
        if entry[1] > 0:
            return
        del self.loan_uses[key]
        self.send(
            pb2.HostMsg(
                release_loan=pb2.ReleaseLoan(segment_id=loan.segment, region=loan.region, offset=loan.offset, len=loan.length)
            )
        )
        self.loans_released.append((loan.segment, loan.region, loan.offset, loan.length))

    # -- chunked overlays (`control_plane_protocol.md` §5.7, §8.5): the host
    # names the (level, t) volumes it wants; the plugin speaks of one at a
    # time; the host holds exactly what it wants --

    MAX_CHUNK_WANTS = 8  # §9's `MAX_CHUNK_WANTS`

    def want_chunks(self, overlay_id: str, wants) -> None:
        """Restate the WHOLE want set of a chunked overlay (`ChunksWanted`),
        in this host's priority order: ``wants`` is a sequence of ``(level,
        t)`` pairs, ``t = None`` for a `static` overlay's one volume per
        level, ``[]`` when nothing is wanted.  Sent once per change, as the
        host does -- no queue at either end.  The host holds exactly what it
        wants: a chunk held under a key that left the set is evicted here
        (`evicted`), its loan released if a scenario was holding it."""
        pairs = [(int(level), None if t is None else int(t)) for level, t in wants]
        if overlay_id not in self.chunked:
            raise ValueError(f"overlay {overlay_id!r} is not declared chunked")
        if len(pairs) > self.MAX_CHUNK_WANTS:
            raise ValueError(f"{len(pairs)} wants; §9 caps a ChunksWanted at {self.MAX_CHUNK_WANTS}")
        if self.chunked[overlay_id] == "static" and any(t is not None for _, t in pairs):
            raise ValueError(f"overlay {overlay_id!r} is static; a want carries no t")
        with self._lock:
            self.wants[overlay_id] = pairs
            keep = set(pairs)
            for key in [k for k in self.chunks if k[0] == overlay_id and (k[1], k[2]) not in keep]:
                del self.chunks[key]
                self.evicted.append(key)
                held = self.held_chunks.pop(key, None)
                if held is not None:
                    self.release_loan(held)
        msg = pb2.ChunksWanted(overlay_id=overlay_id)
        for level, t in pairs:
            want = msg.wants.add(level=level)
            if t is not None:
                want.t = t
        # One send per change, at most one change per pump tick (the host's
        # cadence): a plugin sees every set the host sent, one tick apart.
        wait = self._last_want.get(overlay_id, 0.0) + self.want_tick - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_want[overlay_id] = time.monotonic()
        self.send(pb2.HostMsg(chunks_wanted=msg))

    def refuse(self, site: str, detail: str, count: int = 1) -> None:
        """Tell the plugin the host did not apply something: a host → plugin
        `ContractViolation` at ``site`` (a `ViolationSite` spelling such as
        ``"overlay_chunk_refused"``), report-and-continue (§12)."""
        where = pb2.ViolationSite.Value("VIOLATION_SITE_" + site.upper())
        with self._lock:
            self.refusals.append((site, detail))
        self.send(pb2.HostMsg(contract_violation=pb2.ContractViolation(where=where, detail=detail, count=count)))

    def release_chunk(self, key: tuple[str, int, int | None]) -> None:
        """Release the loan of a chunk held under ``hold_chunk_releases``."""
        with self._lock:
            loan = self.held_chunks.pop(key, None)
        if loan is not None:
            self.release_loan(loan)

    def release_held_chunks(self) -> int:
        """Release every loan held under ``hold_chunk_releases``; returns how many."""
        with self._lock:
            held = list(self.held_chunks.items())
            self.held_chunks.clear()
        for _key, loan in held:
            self.release_loan(loan)
        return len(held)

    def await_chunks(self, overlay_id: str, keys, timeout: float = 5.0) -> dict:
        """The held chunks of ``overlay_id`` once every ``(level, t)`` in
        ``keys`` is held; raises at the deadline naming what is missing."""
        wanted = [(overlay_id, int(level), None if t is None else int(t)) for level, t in keys]

        def ready():
            return dict(self.chunks) if all(k in self.chunks for k in wanted) else None

        try:
            return self._await(ready, timeout, f"chunks {wanted} not all held")
        except AssertionError as exc:
            missing = [k for k in wanted if k not in self.chunks]
            raise AssertionError(f"{exc}; missing {missing}, held {sorted(k for k in self.chunks if k[0] == overlay_id)}") from None

    def await_chunk_log(self, count: int, timeout: float = 5.0) -> list[Chunk]:
        """Every applied chunk in arrival order, once there are ``count``."""
        return self._await(lambda: list(self.chunk_log) if len(self.chunk_log) >= count else None, timeout, f"no {count} applied chunks")

    def _overlay_segment(self, overlay_id: str) -> Segment | None:
        """The `O` segment granted for ``overlay_id`` (the one whose
        ``subject`` names it).  During a supersede there may briefly be two;
        a descriptor names one by id, which `_chunk_loan` checks."""
        for seg in self.segments.values():
            if seg.purpose == "O" and seg.subject == overlay_id:
                return seg
        return None

    def _chunk_loan(self, overlay_id: str, ref: pb2.SegmentRef) -> Loan:
        """A chunk's descriptor: into the overlay's own segment (§5.7: *"a
        ref into the wrong segment"* is refused), bounds-checked like every
        other loan."""
        seg = self.segments.get(ref.segment_id)
        if seg is None or seg.purpose != "O" or seg.subject != overlay_id:
            raise ValueError(f"descriptor names segment {ref.segment_id}, which is not overlay {overlay_id!r}'s")
        return self._take_loan(ref)

    def _on_overlay_chunk(self, chunk: pb2.OverlayChunk) -> None:
        """§5.7's four dispositions of an `OverlayChunk`: refused (site 30)
        when the overlay is not chunked or the chunk cannot be applied;
        dropped and counted when the host did not name its key; applied --
        copied out of the arena, the loan released in the same tick -- when
        it can.  Every disposition releases the loan."""
        overlay_id = chunk.overlay_id
        t = int(chunk.t) if chunk.HasField("t") else None
        key = (overlay_id, int(chunk.level), t)
        labels = chunk.labels if chunk.HasField("labels") else None
        loan: Loan | None = None
        source = labels.WhichOneof("source") if labels is not None else None
        if source == "segment" and self._drop_if_retiring(f"chunk {key}", [labels.segment]):
            return
        if source == "segment":
            try:
                loan = self._chunk_loan(overlay_id, labels.segment)
            except ValueError as exc:
                return self.refuse("overlay_chunk_refused", f"chunk {key}: descriptor: {exc}")

        def dispose():
            if loan is not None:
                self.release_loan(loan)

        if overlay_id not in self.chunked:
            dispose()
            return self.refuse("overlay_chunk_refused", f"chunk {key}: overlay {overlay_id!r} is not declared chunked")
        if (key[1], key[2]) not in self.wants.get(overlay_id, []):
            # Ordinary, not a defect: a want that left the set while the
            # chunk was on the wire.  Dropped, counted, loan released.
            dispose()
            with self._lock:
                self.dropped_chunks.append(key)
            return
        why = self._chunk_defect(overlay_id, chunk, labels, loan)
        if why is not None:
            dispose()
            return self.refuse("overlay_chunk_refused", f"chunk {key}: {why}")
        array = None
        position = (0, 0, 0)
        if labels is not None:
            dtype = np.uint8 if labels.dtype == pb2.DTYPE_U8 else np.uint16
            payload = loan.payload if loan is not None else labels.inline
            array = np.frombuffer(payload, dtype=dtype).reshape(tuple(labels.shape)).copy()
            position = tuple(int(v) for v in chunk.position_zyx) if chunk.position_zyx else (0, 0, 0)
        applied = Chunk(overlay_id, key[1], t, array, position, loan, inline=(source == "inline"))
        with self._lock:
            self.chunks[key] = applied
            self.chunk_log.append(applied)
            superseded = self.held_chunks.pop(key, None)
            hold = self.hold_chunk_releases and loan is not None
            if hold:
                self.held_chunks[key] = loan
        if superseded is not None:
            self.release_loan(superseded)
        if not hold:
            dispose()

    def _chunk_defect(self, overlay_id: str, chunk: pb2.OverlayChunk, labels, loan: Loan | None) -> str | None:
        """Why a WANTED chunk cannot be applied, or ``None``: a level the
        dataset lacks, a static overlay's chunk carrying `t` (or a per-t
        one without), labels that are not a 3-D u8/u16 volume, an inline
        field over §9's bound, a descriptor whose length is not the shape's,
        a box outside the level's shape."""
        level = int(chunk.level)
        if not 0 <= level < self.dataset.levels:
            return f"level {level} is not one of the dataset's {self.dataset.levels}"
        static = self.chunked[overlay_id] == "static"
        if static and chunk.HasField("t"):
            return "a static overlay's chunk carries no t"
        if not static and not chunk.HasField("t"):
            return "a per_t overlay's chunk names its t"
        if labels is None:
            return None
        if labels.dtype not in (pb2.DTYPE_U8, pb2.DTYPE_U16):
            return f"labels are u8 or u16, not {_enum_name(pb2.DType, labels.dtype)}"
        if len(labels.shape) != 3 or any(n < 1 for n in labels.shape):
            return f"labels are a (Z, Y, X) volume, got shape {list(labels.shape)}"
        width = 1 if labels.dtype == pb2.DTYPE_U8 else 2
        expected = int(np.prod(labels.shape)) * width
        if loan is not None and loan.length != expected:
            return f"descriptor len {loan.length} != shape x dtype {expected}"
        if labels.WhichOneof("source") == "inline":
            if len(labels.inline) > 8 << 20:
                return f"inline labels of {len(labels.inline)} bytes exceed §9's 8 MiB"
            if len(labels.inline) != expected:
                return f"inline len {len(labels.inline)} != shape x dtype {expected}"
        position = list(chunk.position_zyx) or [0, 0, 0]
        if len(position) != 3:
            return f"position_zyx has {len(position)} entries"
        shape = self.dataset.level_shape(level)
        if any(p + n > s for p, n, s in zip(position, labels.shape, shape)):
            return f"box at {position} of shape {list(labels.shape)} is outside level {level}'s shape {list(shape)}"
        return None

    def _on_label_table(self, table: pb2.OverlayLabelTable) -> None:
        """`OverlayLabelTable`, coalescing per overlay: the LUT (inline or a
        loan, read and released), names and visibility by label id, the
        style.  A malformed LUT is refused on site 30."""
        overlay_id = table.overlay_id
        loan: Loan | None = None
        if table.colormap.WhichOneof("source") == "segment" and self._drop_if_retiring(f"label table {overlay_id}", [table.colormap.segment]):
            return  # vizi drops it before anything else
        if overlay_id not in self.chunked:
            return self.refuse("overlay_chunk_refused", f"label table: overlay {overlay_id!r} is not declared chunked")
        lut = table.colormap
        source = lut.WhichOneof("source")
        if source == "segment":
            try:
                loan = self._chunk_loan(overlay_id, lut.segment)
            except ValueError as exc:
                return self.refuse("overlay_chunk_refused", f"label table: descriptor: {exc}")
            payload = loan.payload
            self.release_loan(loan)
        elif source == "inline":
            payload = lut.inline
        else:
            return self.refuse("overlay_chunk_refused", "label table: no colormap")
        if list(lut.shape) != [256, 4] or lut.dtype != pb2.DTYPE_U8 or len(payload) != 1024:
            return self.refuse("overlay_chunk_refused", f"label table: colormap is (256, 4) u8, got shape {list(lut.shape)} {_enum_name(pb2.DType, lut.dtype)} of {len(payload)} bytes")
        applied = LabelTable(
            overlay_id,
            np.frombuffer(payload, dtype=np.uint8).reshape(256, 4).copy(),
            {int(k): str(v) for k, v in table.label_names.items()},
            {int(k): bool(v) for k, v in table.label_visible.items()},
            pb2.OverlayStyle(),
        )
        applied.style.CopyFrom(table.style)
        with self._lock:
            self.label_tables[overlay_id] = applied
            self.table_log.append(applied)

    # -- inbound --

    def _read_loop(self) -> None:
        try:
            for message in self._responses:
                try:
                    self._on_message(message)
                except Exception as exc:  # noqa: BLE001 -- see below
                    self._handler_failed(message, exc)
        except grpc.RpcError as exc:
            self.stream_error = exc
        finally:
            self.stream_ended.set()

    def _handler_failed(self, message: pb2.PluginMsg, exc: Exception) -> None:
        """A handler raised on something the plugin sent.  The host never
        lets that kill its reader (a request must never go unanswered, so
        it is answered `INTERNAL`), so neither does this:
        a REQUEST is answered `Error{INTERNAL}` echoing its id, and the
        exception goes into ``unhandled`` -- every driver's ``not
        host.unhandled`` then fails loudly, naming it, since an emulator
        exception is a gap in the emulator or a plugin input nothing here
        anticipated."""
        kind = message.WhichOneof("msg")
        detail = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self.unhandled.append(f"handler for {kind} raised {detail}")
        if message.id and kind in _REQUEST_KINDS:
            self._fail(message.id, pb2.Error.INTERNAL, f"hostsim could not answer {kind}: {detail}")

    def _fault_session(self, detail: str) -> None:
        """A protocol error ends the session: recorded, and `Shutdown{session_fault}`
        sent off the reader thread.  What arrives after it is not dispatched."""
        with self._lock:
            first = not self.session_faults
            self.session_faults.append(detail)
            # every driver's `not host.unhandled` then fails, naming it
            self.unhandled.append(f"session fault: {detail}")
        if first:
            threading.Thread(
                target=self.shutdown, kwargs={"reason": pb2.SHUTDOWN_REASON_SESSION_FAULT, "detail": detail},
                daemon=True, name="hostsim-fault",
            ).start()

    def _on_message(self, message: pb2.PluginMsg) -> None:
        kind = message.WhichOneof("msg")
        if self.session_faults:
            if kind in ("shutdown_report", "doorbell_final_report"):
                pass  # the end of a session this emulator faulted, still recorded
            else:
                return
        elif self.ready is None and kind != "ready":
            # §3.1: `Ready` is the plugin's first message, whatever follows;
            # the unknown-kind rule applies only after the handshake.
            return self._fault_session(f"the plugin's first message was {kind or 'of a kind hostsim does not know'}, not Ready")
        seen = self._ids.observe(message, kind)
        if seen == "resent":
            # A byte-identical re-send (at-least-once delivery): no second dispatch.
            with self._lock:
                self.deduplicated.append((message.id, kind or ""))
            return
        if seen == "reused":
            return self._fault_session(f"the plugin reused id {message.id!r} with different content ({kind or 'an unknown kind'})")
        if kind is None:
            # `control_plane_protocol.md` §4 rule 8 and §11: a kind this
            # emulator does not know is never a reply.  Under the id of one of
            # the host's own requests in flight it breaks rule 8 (a pending
            # request's id belongs to its reply alone) and faults the session;
            # the waiter gets the fault as `Error{PROTOCOL}`.  Under any other
            # non-empty id it is answered UNSUPPORTED; an empty id is a push,
            # noted.  The ledger holds this id in its request table, apart
            # from the replies, so it never collides with a reply's id.
            with self._lock:
                self.unknown_kinds.append(message.id)
                waiter = self._waiters.pop(message.id, None) if message.id else None
                asked = self._asked.pop(message.id, None) if waiter is not None else None
                if waiter is not None:
                    self._invoked.pop(message.id, None)
            if waiter is not None:
                detail = (f"the plugin sent a message of a kind hostsim does not know under the id {message.id!r} "
                          f"of the host's pending {asked or 'request'}, which takes its reply alone "
                          "(control_plane_protocol.md §4 rule 8)")
                waiter.put(pb2.PluginMsg(id=message.id, error=pb2.Error(kind=pb2.Error.PROTOCOL, message=detail)))
                return self._fault_session(detail)
            if message.id:
                self.send(pb2.HostMsg(id=message.id, host_error=pb2.Error(kind=pb2.Error.UNSUPPORTED, message="hostsim predates this message kind")))
            else:
                self.diagnostics.append("a push of a message kind hostsim predates")
            return
        if kind in _REPLY_KINDS and message.id:
            with self._lock:
                asked = self._asked.get(message.id)
                takes = _HOST_REQUEST_REPLIES.get(asked, ())
                pending = message.id in self._waiters
                wrong = pending and bool(takes) and kind not in takes and kind != "error"
                invoked = self._invoked.get(message.id)
                # §8.3: a `CommandDone`
                # echoes the invoked command's id -- on success, on failure and
                # on an argument-validation failure alike.  One without it, or
                # naming another command, is the sender's protocol error.
                misnamed = (pending and not wrong and kind == "command_done" and invoked is not None
                            and message.command_done.command_id != invoked)
                waiter = None if wrong or misnamed else self._waiters.pop(message.id, None)
                if waiter is not None:
                    self._asked.pop(message.id, None)
                    self._invoked.pop(message.id, None)
            if wrong or misnamed:
                # A pending request's id belongs to its reply alone: a known
                # reply of another kind under it is a protocol error that
                # faults the session.  The waiter never receives it; it gets
                # the fault, as an `Error{PROTOCOL}` naming it.  So does a
                # `CommandDone` that does not name the command invoked.
                if wrong:
                    detail = (f"the plugin answered request {message.id!r} ({asked}, which takes "
                              f"{' or '.join(takes)} or error) with {kind}")
                else:
                    named = message.command_done.command_id
                    detail = (f"the plugin answered InvokeCommand {message.id!r} ({invoked!r}) with a CommandDone "
                              + (f"naming {named!r}" if named else "without command_id")
                              + "; CommandDone.command_id echoes the invoked command's id (control_plane_protocol.md §8.3)")
                with self._lock:
                    waiter = self._waiters.pop(message.id, None)
                    self._asked.pop(message.id, None)
                    self._invoked.pop(message.id, None)
                if waiter is not None:
                    waiter.put(pb2.PluginMsg(id=message.id, error=pb2.Error(kind=pb2.Error.PROTOCOL, message=detail)))
                return self._fault_session(detail)
            if waiter is not None:
                waiter.put(message)
                return
            self.unhandled.append(f"{kind} for unknown request {message.id!r}")
            return
        # §9's per-field bounds on every plugin message that is not a reply,
        # as the host applies them before anything else: a request is answered
        # `Error{PROTOCOL}`, a push reported on the site naming the rule.
        found = _over_bound(message)
        if found is not None:
            return self._refuse_over_bound(message, kind, found[0], f"{found[1]} (control_plane_protocol.md §9)")
        if kind in _REQUEST_KINDS and message.id:
            # The host's request ledger: taken in HERE, cleared as its reply
            # leaves (`send`).
            with self._lock:
                self._open_requests[message.id] = False
        if kind == "ready":
            self.ready_id = message.id
            refusal = ready_refusal(message.ready, self.protocol)
            if refusal is not None:
                # §3.1: the host refuses a `Ready` above its own `Hello` and
                # fails the session: the request stream ends here.
                self.handshake_refusal = refusal
                self.diagnostics.append(f"handshake refused: {refusal}")
                self._outbound.put(None)
                return
            # §8.4: the appearance goes once after `Ready`, before anything
            # else the session's surfaces send -- queued here, before a driver
            # waiting on `ready` can open a panel.
            self._send_appearance()
            self.ready = message.ready
            if self.subscriptions & {"view", "regions"}:
                threading.Thread(target=self._subscription_pump, daemon=True, name="hostsim-subs").start()
        elif kind == "panel_patch":
            self._on_panel_patch(message.panel_patch, message.panel_patch.ByteSize())
        elif kind == "work_progress":
            self._on_work_progress(message.work_progress)
        elif kind == "shape_proposal":
            self._on_shape_result(message.shape_proposal, answer=False)
        elif kind == "shape_answer":
            self._on_shape_result(message.shape_answer, answer=True)
        elif kind == "overlay_update":
            self._on_overlay_update(message.overlay_update)
        elif kind == "overlay_remove":
            remove = message.overlay_remove
            self.pushes.append(Push("remove", remove.overlay_id, remove.revision, None))
            if remove.overlay_id in self.chunked:
                # §8.5: on a chunked overlay a remove clears the chunks and the table.
                with self._lock:
                    gone = [k for k in self.chunks if k[0] == remove.overlay_id]
                    for key in gone:
                        del self.chunks[key]
                    held = [self.held_chunks.pop(k) for k in gone if k in self.held_chunks]
                    self.label_tables.pop(remove.overlay_id, None)
                for loan in held:
                    self.release_loan(loan)
        elif kind == "overlay_chunk":
            self._on_overlay_chunk(message.overlay_chunk)
        elif kind == "overlay_label_table":
            self._on_label_table(message.overlay_label_table)
        elif kind in _DATA_ACCESS and self._shutdown_sent:
            # Once teardown has begun, the
            # data-access service drops a request unanswered -- §3.3 lets a
            # request inside the shutdown window go unanswered, and starting
            # its work would race the teardown.  A plugin must not hang on it.
            with self._lock:
                self._open_requests.pop(message.id, None)
                self.dropped_in_shutdown.append((kind, message.id))
        elif kind in _DATA_ACCESS and not self._permitted(message.id, "data_access"):
            pass
        elif kind in _CONTRIBUTE and not self._permitted(message.id, "contribute_rois"):
            pass
        elif kind == "get_view_state":
            view = self.dataset.view_state()
            view.file_state_dir = self.file_state_dir
            self.send(pb2.HostMsg(id=message.id, view_state=view))
        elif kind == "read_data_volume":
            self._answer_read(message.id, message.read_data_volume)
        elif kind == "contribute_rois":
            self._on_contribute(message.id, message.contribute_rois)
        elif kind == "get_rois":
            self._on_get_rois(message.id, message.get_rois)
        elif kind == "extend_roi_poses":
            self._on_extend_poses(message.id, message.extend_roi_poses)
        elif kind == "read_roi_inclusion":
            self._on_read_roi_inclusion(message.id, message.read_roi_inclusion)
        elif kind == "request_status_query":
            self._on_status_query(message.id, message.request_status_query)
        elif kind == "compile_label_volume":
            self._on_compile(message.id, message.compile_label_volume)
        elif kind == "read_mask_voxels":
            self._on_read_mask(message.id, message.read_mask_voxels)
        elif kind == "request_asset":
            self._on_request_asset(message.id, message.request_asset)
        elif kind == "view_request":
            self._on_view_request(message.id, message.view_request)
        elif kind == "offer_follow":
            self._on_offer_follow(message.offer_follow)
        elif kind == "follow_state":
            self._on_follow_state(message.follow_state)
        elif kind == "end_follow":
            with self._follow_lock:
                running = self.follow
                if running is not None and running["follow_id"] == int(message.end_follow.follow_id):
                    self.end_follow("PLUGIN_ENDED", "the plugin ended its Follow")
        elif kind == "request_save_path":
            self._on_request_save_path(message.id, message.request_save_path)
        elif kind == "get_colormap":
            name = message.get_colormap.name
            table = self.colormaps.get(name)
            if table is None:
                self._fail(message.id, pb2.Error.NOT_FOUND, f"no colormap named {name!r}")
            else:
                self.send(pb2.HostMsg(id=message.id, colormap_table=pb2.ColormapTable(name=name, rgba=table)))
        elif kind == "activate_tool":
            self._on_activate_tool(message.id, message.activate_tool)
        elif kind == "grant_ack":
            ack = self._judge_ack(message.grant_ack)
            if ack is None:
                return
            if ack.segment_id in self.v_rings and ack.WhichOneof("outcome") != "installed":
                # §9: a failed replacement leaves the incumbent current and
                # usable, as the host does.
                seg = self.segments.get(ack.segment_id)
                incumbent = seg.supersedes if seg is not None else 0
                self.diagnostics.append(f"V ring {ack.segment_id} failed: {ack}")
                if self.volume_segment == ack.segment_id:
                    self.volume_segment = incumbent if incumbent in self.segments else None
        elif kind == "grant_state":
            with self._lock:
                self.grant_states[message.id] = message.grant_state
        elif kind == "stage_arm_ack":
            ack = message.stage_arm_ack
            with self._lock:
                self.stage_arm_acks[(ack.pipeline, ack.position)] = (ack.revision, ack.stage_id)
        elif kind == "tool_binding_ack":
            with self._lock:
                self.tool_acks[message.tool_binding_ack.tool] = message.tool_binding_ack.revision
        elif kind == "epoch_ack":
            with self._lock:
                for pane in message.epoch_ack.panes:
                    self.epoch_acks[pane.view_id] = pane.epoch
        elif kind == "time_mapping_ack":
            with self._lock:
                self.mapping_acks[message.time_mapping_ack.gen] = message.time_mapping_ack.revision
        elif kind == "contract_violation":
            with self._lock:
                self.violations.append(message.contract_violation)
        elif kind == "segment_shortfall":
            with self._lock:
                self.shortfalls.append(message.segment_shortfall)
        elif kind == "release_deposit":
            deposit = message.release_deposit
            self.released.append((deposit.segment_id, deposit.region, deposit.offset, deposit.len))
            seg = self.segments.get(deposit.segment_id)
            if seg is not None:
                region = seg.region_at(deposit.offset)
                if isinstance(region, HostRing):
                    region.release(deposit.offset, deposit.len)
                    if deposit.segment_id in self._v_draining and not region.outstanding:
                        self._v_draining.discard(deposit.segment_id)
                        self.v_retired_on_drain.append(deposit.segment_id)
                        self.send(pb2.HostMsg(retire_segment=pb2.RetireSegment(segment_id=deposit.segment_id)))
        elif kind == "segment_released":
            gone = message.segment_released.segment_id
            with self._lock:
                self.retired[gone] = message.segment_released
                self.retiring_segments.discard(gone)
                self.drops_at_release[gone] = len(self.dropped_retiring)
            if gone in self.v_rings and gone != self.volume_segment and gone in self.segments:
                # A `V` ring this emulator superseded: the plugin has let go.
                with self._lock:
                    ring = self.segments.pop(gone, None)
                if ring is not None:
                    ring.close()
        elif kind == "shutdown_report":
            self.shutdown_report = message.shutdown_report
        elif kind == "doorbell_final_report":
            with self._lock:
                self.final_report = message.doorbell_final_report
        elif kind == "release_request":
            # A HINT the host owes nothing: recorded for a segment the session holds,
            # dropped for any other, never acted on.
            request = message.release_request
            with self._lock:
                if request.segment_id in self.segments:
                    self.release_requests[request.segment_id] = request.reason
        elif kind == "doorbell_fault":
            fault = message.doorbell_fault
            with self._lock:
                self.plugin_doorbell_faults.append(fault)
            self._end_session_soon(
                f"the plugin reported {_enum_name(pb2.DoorbellFailureClass, fault.failure_class)}",
                reason=pb2.DOORBELL_SHUTDOWN_PEER_FAULT,
                failure_class=fault.failure_class,
            )
        else:
            self.unhandled.append(kind or "?")

    def _judge_ack(self, ack: pb2.GrantAck) -> pb2.GrantAck | None:
        """One `GrantAck`, as the host judges it:
        an ack for a segment never granted is ignored; an `installed` whose
        identity proof is not the one written is a FAILED grant (§8.12 --
        compared whole, an empty proof included); and the FIRST terminal
        answer stands -- a later one that contradicts it is a diagnostic,
        never a change of state.  Returns the ack
        recorded, or ``None`` when nothing changed."""
        segment = ack.segment_id
        with self._lock:
            seg = self.segments.get(segment)
            previous = self.acks.get(segment)
        if seg is None:
            self.diagnostics.append(f"GrantAck for segment {segment}, which was never granted (or is gone)")
            return None
        outcome = ack.WhichOneof("outcome")
        if outcome == "installed" and bytes(ack.installed.identity_proof) != seg.proof:
            ack = pb2.GrantAck(
                segment_id=segment,
                failed=pb2.GrantFailed(failure_class=pb2.GRANT_FAILURE_FAILED, detail="identity proof mismatch"),
            )
            self.diagnostics.append(f"segment {segment}: GrantAck installed with the wrong identity proof; the grant is dead")
        elif outcome is None:
            ack = pb2.GrantAck(
                segment_id=segment,
                failed=pb2.GrantFailed(failure_class=pb2.GRANT_FAILURE_FAILED, detail="GrantAck carried no outcome"),
            )
        if previous is not None:
            if previous.WhichOneof("outcome") != ack.WhichOneof("outcome"):
                self.diagnostics.append(
                    f"segment {segment}: GrantAck {ack.WhichOneof('outcome')} contradicts the earlier "
                    f"{previous.WhichOneof('outcome')}; the first stands"
                )
            return None
        with self._lock:
            self.acks[segment] = ack
        return ack

    def _refuse_over_bound(self, message: pb2.PluginMsg, kind: str, rule: str, detail: str) -> None:
        """A message over a §9 bound, as the host refuses it: a request answered `Error{PROTOCOL}`;
        a push reported on `repeated_field_overflow` for a count, or its own
        refusal site for an inline field; its loans released -- a refused
        push is still a disposition (§5.6)."""
        with self._lock:
            self.over_bound.append((kind, detail))
        if message.id and kind in _REQUEST_KINDS:
            return self._fail(message.id, pb2.Error.PROTOCOL, detail)
        for ref in _segment_refs(message):
            try:
                self.release_loan(self._take_loan(ref))
            except ValueError:
                pass
        if rule == "repeated":
            return self.refuse("repeated_field_overflow", f"{kind}: {detail}")
        site = {
            "overlay_update": "overlay_item_refused",
            "overlay_chunk": "overlay_chunk_refused",
            "overlay_label_table": "overlay_chunk_refused",
            "panel_patch": "panel_patch_refused",
        }.get(kind)
        if site is not None:
            self.refuse(site, f"{kind}: {detail}")
        else:
            self.diagnostics.append(f"{kind} refused over a §9 bound: {detail}")

    def _fail(self, request_id: str, kind: int, message: str) -> None:
        self.send(pb2.HostMsg(id=request_id, host_error=pb2.Error(kind=kind, message=message)))

    def _permitted(self, request_id: str, permission: str) -> bool:
        """§8.7 / §8.8: gated on the manifest, declared-equals-granted."""
        if permission in self.permissions:
            return True
        self._fail(request_id, pb2.Error.UNSUPPORTED, f"plugin does not declare the {permission} permission")
        return False

    def _new_region(self) -> str:
        self._region_seq += 1
        return f"r{self._region_seq}"

    def _roi_refusal(self, roi) -> str | None:
        """§8.8: why the host refuses a contributed `Roi` at the
        wire, or ``None`` -- a NaN or infinite vertex position or direction,
        pose row or frame basis (stored, one poisons every bound, raster and
        measurement the region takes part in), or a pose domain outside
        ``[0, T)`` of the open dataset.  Both `INVALID_ARGUMENT`."""
        for i, vertex in enumerate(roi.vertices):
            for name in ("position", "direction"):
                if name == "direction" and not vertex.HasField("direction"):
                    continue
                v = getattr(vertex, name)
                if not all(np.isfinite((v.z, v.y, v.x))):
                    return f"vertex {i}'s {name} ({v.z}, {v.y}, {v.x}) is not finite"
        for component in ("pose_translation", "pose_rotation", "pose_stretch", "frame_basis"):
            rows = getattr(roi, component)
            if len(rows) and not np.isfinite(np.asarray(rows, dtype=np.float64)).all():
                return f"{component} carries a NaN or infinite value"
        frames = int(self.dataset.shape_tczyx[0])
        for end in ("t_start", "t_end"):
            if roi.HasField(end) and not 0 <= getattr(roi, end) < frames:
                return f"{end}={getattr(roi, end)} is outside [0, {frames}) of the open dataset"
        # What a contributed region may be -- any geometry,
        # under the three rules (`regions.contribution_refusal`).
        return regions.contribution_refusal(roi, self.region_types)

    def _on_contribute(self, request_id: str, batch: pb2.ContributeRois) -> None:
        """§8.8: `contrib_id` is stable inside the plugin's namespace, a
        re-emit replaces the row, and `replace_all` removes the plugin's
        contributed regions omitted, never its compiled labels.  The batch is checked whole before the first write, with the
        wire refusals of `_roi_refusal` (non-finite values, a domain outside
        ``[0, T)``), `INVALID_ARGUMENT`."""
        self.contributions.put(batch)
        origin = f"plugin:{self.plugin_id}"
        for i, contribution in enumerate(batch.rois):
            if not contribution.contrib_id:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "a contribution needs a contrib_id")
            if contribution.contrib_id.startswith("label:"):
                # The prefix is CompileLabelVolume's: a compiled label is
                # `label:<value>`, so no contribution can take its place.
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"contribution [{i}] {contribution.contrib_id!r}: the 'label:' prefix is reserved for compiled labels")
            why = self._roi_refusal(contribution.roi)
            if why is not None:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"contribution [{i}] {contribution.contrib_id!r}: {why}")
        seen = set()
        for contribution in batch.rois:
            existing = next((r for r in self.rois.values() if r.origin == origin and r.contrib_id == contribution.contrib_id), None)
            region_id = existing.region_id if existing is not None else self._new_region()
            record = pb2.RoiRecord(region_id=region_id, origin=origin, contrib_id=contribution.contrib_id, name=contribution.name)
            record.roi.CopyFrom(contribution.roi)
            record.style.CopyFrom(contribution.style)
            if record.roi.type_name in self.region_types:
                record.type_owner = self.plugin_id
            self.rois[region_id] = record
            # §8.8's outcome table: a region contributed again ends its edit.
            self._cancel_drags_of(region_id, "the region was contributed again")
            # The dense domain a later window slices and a later extension
            # grows: the span the contribution declared, when it sent rows.
            # As the host reads it: a region has a domain when its rest
            # form declares one -- `t_start` set -- rows or not.
            if record.roi.HasField("t_start") and record.roi.HasField("t_end"):
                self.pose_domains[region_id] = (int(record.roi.t_start), int(record.roi.t_end))
            else:
                self.pose_domains.pop(region_id, None)
            seen.add(contribution.contrib_id)
        if batch.replace_all:
            # §8.8: `replace_all` replaces the request's own namespace only.
            # The plugin's compiled labels (`label:<n>`, CompileLabelVolume's
            # namespace) are never touched by a ContributeRois.
            for region_id, record in list(self.rois.items()):
                if (record.origin == origin and not record.contrib_id.startswith("label:")
                        and record.contrib_id not in seen):
                    del self.rois[region_id]
                    self.masks.pop(region_id, None)
        self.send(pb2.HostMsg(id=request_id, rois_contributed=pb2.RoisContributed(region_count=len(batch.rois))))

    #: The pose components a region may carry, and how many doubles each
    #: costs per frame (`roi.proto`; §9's arithmetic).
    _POSE_COMPONENTS = (("pose_translation", 3), ("pose_rotation", 3), ("pose_stretch", 6))

    def _on_get_rois(self, request_id: str, request: pb2.GetRois) -> None:
        """§8.8 with the optional pose window, in the shape the
        `1.0` host gives it.

        **Unwindowed**: one `Roi` per region POSED AT THE FRAME ON SCREEN with
        its pose cleared -- no rows, and `t_start`/`t_end` ABSENT,
        because posing bakes the time in and clears it; a region with
        `out_of_span = HIDE` that is not present at that frame is not listed
        at all.  **Windowed**: the REST form with its rows clipped to the
        window and `t_start`/`t_end` naming the rows returned; a window that
        misses the domain entirely returns the shape timeless -- rows AND
        `t_start`/`t_end` cleared.  Half a window is a window, its missing end
        open; an INVERTED one is refused `INVALID_ARGUMENT`.  Every region
        with a domain carries `pose_domain_t_start`/`_t_end` either way,
        and a region declared over `[t_start, t_end]` has a domain
        whether or not it carries rows.

        (Not emulated: re-posing the vertices at the frame on
        screen -- the rows are cleared, which is what a plugin reads.)"""
        window = request.HasField("pose_t_start") or request.HasField("pose_t_end")
        if window:
            lo_w = request.pose_t_start if request.HasField("pose_t_start") else -(2**31)
            hi_w = request.pose_t_end if request.HasField("pose_t_end") else 2**31 - 1
            if lo_w > hi_w:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"pose window [{lo_w}, {hi_w}] is empty")
        out = []
        revisions = self.region_revisions()
        wanted = set(request.region_ids)
        for record in self.rois.values():
            # Only the named regions; an id that does not
            # exist is simply absent.
            if wanted and record.region_id not in wanted:
                continue
            copy = pb2.RoiRecord()
            copy.CopyFrom(record)
            copy.revision = revisions.get(record.region_id, 0)
            copy.content_hash = content_hash(record)
            domain = self.pose_domains.get(record.region_id)
            if not window:
                if domain is not None and record.region_id not in self.masks:
                    t = int(self.dataset.t)
                    if not domain[0] <= t <= domain[1] and record.roi.out_of_span == _OUT_OF_SPAN_HIDE:
                        continue  # absent at the frame on screen: not listed
                    copy.roi.ClearField("t_start")
                    copy.roi.ClearField("t_end")
                for component, _ in self._POSE_COMPONENTS:
                    copy.roi.ClearField(component)
            elif domain is not None:
                lo, hi = domain
                a, b = max(lo_w, lo), min(hi_w, hi)
                if a > b:
                    # the window misses the domain: timeless, and the domain
                    # pair says where the pose is
                    for component, _ in self._POSE_COMPONENTS:
                        copy.roi.ClearField(component)
                    copy.roi.ClearField("t_start")
                    copy.roi.ClearField("t_end")
                else:
                    for component, width in self._POSE_COMPONENTS:
                        rows = list(getattr(record.roi, component))
                        if not rows:
                            continue
                        copy.roi.ClearField(component)
                        getattr(copy.roi, component).extend(rows[(a - lo) * width : (b - lo + 1) * width])
                    # the SLICE's own span, which is what a reader compares
                    # the domain pair against to see that it holds a window
                    copy.roi.t_start, copy.roi.t_end = a, b
            # The pair is
            # carried on every posed record, windowed or not.  A reader tells
            # a slice by comparing it with `roi.t_start`/`t_end`, not by
            # asking whether the pair is there.
            if domain is not None:
                copy.pose_domain_t_start, copy.pose_domain_t_end = domain
            out.append(copy)
        reply = pb2.RoiList(rois=out)
            # An oversized reply is REFUSED, never clipped.  A
        # clipped window would look exactly like a small one -- the domain
        # pair describes the whole either way -- so the plugin would silently
        # hold fewer frames than it asked for.  It resizes and asks again.
        if reply.ByteSize() > self.max_message_bytes:
            return self._fail(
                request_id,
                pb2.Error.OUT_OF_BOUNDS,
                f"the windowed reply is {reply.ByteSize()} bytes, past the {self.max_message_bytes}-byte message ceiling; "
                "ask for fewer frames (every posed record carries its domain, so the size is knowable before asking)",
            )
        self.send(pb2.HostMsg(id=request_id, rois=reply))

    def _on_extend_poses(self, request_id: str, batch: pb2.ExtendRoiPoses) -> None:
        """§8.8's delta form, with the `1.0` host's checks
        in its order.
        First the shape of the whole batch, `INVALID_ARGUMENT`: an EMPTY
        batch, an empty `contrib_id`, one `contrib_id` twice in one batch (no
        defined order once they overlap), `span` 0, a component whose length
        is not width x span, an extension carrying NO component.  Then against
        the store: an unknown region `NOT_FOUND`; a region with no pose domain,
        a component set that differs from the region's, or a GAP,
        `INVALID_ARGUMENT` -- overlap replaces (a re-tracked stretch is a
        correction) and a gap would mint frames nothing defines.  Nothing is
        written until every extension has passed.  (§9's per-field bound --
        1 048 576 doubles per component in ONE extension -- is applied before
        this, to every message, `_over_bound`.)  The shape pass has two more
        `INVALID_ARGUMENT`s: a frame outside ``[0, T)`` of
        the open dataset, and a NaN or infinite value in any row.

        `RoiPosesExtended.frame_count` is what the proto says: the ROWS
        WRITTEN, the sum of the spans, which the host answers too."""
        origin = f"plugin:{self.plugin_id}"
        if not batch.extensions:
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "ExtendRoiPoses carries no extensions")
        seen: set[str] = set()
        for i, ext in enumerate(batch.extensions):
            where = f"pose extension [{i}]"
            if not ext.contrib_id.strip():
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{where}: contrib_id must be non-empty")
            if ext.contrib_id in seen:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{where}: contrib_id {ext.contrib_id!r} appears twice in one batch")
            seen.add(ext.contrib_id)
            if ext.span <= 0:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{where}: span must be positive")
            for component, width in self._POSE_COMPONENTS:
                n = len(getattr(ext, component))
                if n and n != width * ext.span:
                    return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{where}: {component} is {n} doubles, expected {width * ext.span} for span {ext.span}")
            if not any(len(getattr(ext, component)) for component, _ in self._POSE_COMPONENTS):
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{where}: carries no pose component")
            # §8.8's table: every frame inside [0, T) of the open
            # dataset, and no NaN or infinite value in any row
            frames = int(self.dataset.shape_tczyx[0])
            if ext.t_start < 0 or ext.t_start + ext.span > frames:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{where}: frames [{ext.t_start}, {ext.t_start + ext.span - 1}] leave [0, {frames}) of the open dataset")
            for component, _ in self._POSE_COMPONENTS:
                rows = getattr(ext, component)
                if len(rows) and not np.isfinite(np.asarray(rows, dtype=np.float64)).all():
                    return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{where}: {component} carries a NaN or infinite value")
        planned = []
        for ext in batch.extensions:
            record = next((r for r in self.rois.values() if r.origin == origin and r.contrib_id == ext.contrib_id), None)
            if record is None:
                return self._fail(request_id, pb2.Error.NOT_FOUND, f"this plugin has contributed no region {ext.contrib_id!r}")
            domain = self.pose_domains.get(record.region_id)
            if domain is None:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"region {ext.contrib_id!r} has no pose domain; contribute one before extending it")
            have = {c for c, _ in self._POSE_COMPONENTS if len(getattr(record.roi, c))}
            carried = {c for c, _ in self._POSE_COMPONENTS if len(getattr(ext, c))}
            if have != carried:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"region {ext.contrib_id!r} carries pose components {sorted(have) or ['none']} and the extension carries {sorted(carried)}; they must match")
            lo, hi = domain
            a, b = ext.t_start, ext.t_start + ext.span - 1
            if a > hi + 1 or b < lo - 1:
                return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"region {ext.contrib_id!r} spans [{lo}, {hi}] and the extension spans [{a}, {b}]; a gap between them would leave frames with no pose")
            planned.append((record, ext, (lo, hi), (a, b)))
        frames = 0
        for record, ext, (lo, hi), (a, b) in planned:
            nlo, nhi = min(lo, a), max(hi, b)
            for component, width in self._POSE_COMPONENTS:
                rows = list(getattr(ext, component))
                if not rows:
                    continue
                old = list(getattr(record.roi, component))
                grown = [0.0] * (width * (nhi - nlo + 1))
                grown[(lo - nlo) * width : (lo - nlo) * width + len(old)] = old
                grown[(a - nlo) * width : (a - nlo) * width + len(rows)] = rows
                record.roi.ClearField(component)
                getattr(record.roi, component).extend(grown)
            self.pose_domains[record.region_id] = (nlo, nhi)
            self._touch(record.region_id)  # edited in place: a new revision
            # the region's declared span IS its pose domain, and an extension
            # grows both (the host replaces t_start/t_end)
            record.roi.t_start, record.roi.t_end = nlo, nhi
            self.pose_extensions.append(ext)
            frames += int(ext.span)
        self.send(pb2.HostMsg(id=request_id, roi_poses_extended=pb2.RoiPosesExtended(region_count=len(planned), frame_count=frames)))

    def _on_status_query(self, request_id: str, request: pb2.RequestStatusQuery) -> None:
        """§8.9's `RequestStatusQuery`: the host's own
        bookkeeping for one request and nothing more.  PENDING is never a
        promise the request will finish, and UNKNOWN is not evidence the work
        failed -- only that this host is not holding it.

        Read from the host's request LEDGER alone, as the host's answer
        is: PENDING for ANY request taken in and not
        yet answered -- a read parked by `hold_reads`, a command's
        `GetRois` in flight -- with `awaiting_user` only for the one a
        person is holding (the picker)."""
        self.status_queries.append(request.subject_id)
        with self._lock:
            pending = request.subject_id in self._open_requests and request.subject_id != request_id
            waiting = pending and self._open_requests[request.subject_id]
        self.send(pb2.HostMsg(id=request_id, request_status=pb2.RequestStatus(
            subject_id=request.subject_id,
            disposition=pb2.RequestStatus.PENDING if pending else pb2.RequestStatus.UNKNOWN,
            awaiting_user=waiting,
        )))

    def hold_picker(self) -> None:
        """Keep the file picker's dialog up: the next `RequestAsset` blocks in
        the host until `answer_picker`, which is what holds the plugin's
        command deadline open.  A driver uses this to watch a
        plugin wait, and to ask `RequestStatusQuery` while it does."""
        self._picker_held = threading.Event()
        self._picker_abandoned = False

    def answer_picker(self, abandoned: bool = False) -> None:
        """Let a held picker answer.  With ``abandoned`` the reply is sent
        knowing the plugin has already given up on it -- §8.9 says it goes out
        against the retired id and the receiver drops it silently, WITHOUT
        faulting, which is the thing worth testing."""
        self._picker_abandoned = abandoned
        if self._picker_held is not None:
            self._picker_held.set()

    def await_picker(self, timeout: float = 5.0) -> str:
        """Block until a held picker has a request waiting on it; answers the
        request id, which is what `RequestStatusQuery` names."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._picker_pending:
                    return next(iter(self._picker_pending))
            time.sleep(0.01)
        raise AssertionError("no RequestAsset reached the held picker")

    def _on_compile(self, request_id: str, request: pb2.CompileLabelVolume) -> None:
        """§8.8 and `examples/interactive_seg/SPEC.md` note 3: the item
        must be a voxel selection -- `frame` voxel, unit scale, identity
        rotation, an integer position -- with segment-backed labels and LUT,
        which are read out of the overlay's arena and disposed of (one more
        use of each, §5.6).  One native mask per nonzero label, colored by
        the LUT rule (`lut_row`) and named from `label_names[id]` -- keyed by
        LABEL ID, so labels sharing a row keep their own names."""
        self.compiles.append(request)
        item = request.item
        if request.overlay_id not in self.overlay_ids:
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"overlay {request.overlay_id!r} is not declared")
        if request.overlay_id in self.chunked:
            # §5.7 / §8.8: a chunked overlay never compiles -- the item's
            # loans, if any, are released as a refused request's are.
            lv = item.label_volume
            for buf in (lv.labels, lv.colormap):
                if buf.WhichOneof("source") == "segment":
                    try:
                        self.release_loan(self._take_loan(buf.segment))
                    except ValueError:
                        pass
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"overlay {request.overlay_id!r} is chunked; a chunked overlay never compiles (§8.8)")
        if item.WhichOneof("geometry") != "label_volume":
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "the item is not a label volume")
        tf = item.transform
        if tf.frame not in ("", "voxel") or (tf.scale_zyx and list(tf.scale_zyx) != [1.0, 1.0, 1.0]) or (tf.rotation_xyzw and list(tf.rotation_xyzw) != [0.0, 0.0, 0.0, 1.0]):
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "a compiled item is a voxel selection: frame voxel, unit scale, identity rotation")
        position = list(tf.position_zyx) or [0.0, 0.0, 0.0]
        if any(v != int(v) for v in position):
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"position_zyx {position} is not integral")
        lv = item.label_volume
        if lv.labels.WhichOneof("source") != "segment" or lv.colormap.WhichOneof("source") != "segment":
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "a compiled item's labels and colormap are segment-backed")
        try:
            labels_loan = self._take_loan(lv.labels.segment)
            lut_loan = self._take_loan(lv.colormap.segment)
        except (KeyError, ValueError) as exc:
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"descriptor: {exc}")
        dtype = {pb2.DTYPE_U8: np.uint8, pb2.DTYPE_U16: np.uint16}.get(lv.labels.dtype)
        if dtype is None:
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "labels are u8 or u16")
        labels = np.frombuffer(labels_loan.payload, dtype=dtype).reshape(tuple(lv.labels.shape))
        lut = np.frombuffer(lut_loan.payload, dtype=np.uint8).reshape(256, 4)
        _, _, depth, height, width = self.dataset.shape_tczyx
        origin = tuple(int(v) for v in position)
        if any(o < 0 for o in origin) or origin[0] + labels.shape[0] > depth or origin[1] + labels.shape[1] > height or origin[2] + labels.shape[2] > width:
            return self._fail(request_id, pb2.Error.OUT_OF_BOUNDS, "the label volume does not lie inside the dataset")
        ids = [int(v) for v in np.unique(labels) if v != 0]
        if len(ids) > 500:
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"{len(ids)} labels; at most 500")
        origin_name = f"plugin:{self.plugin_id}"
        if request.replace_all:
            # §8.8: removes this plugin's prior regions ABSENT from the batch; a
            # label present keeps its row (and its region id), as ContributeRois does.
            present = {f"label:{label}" for label in ids}
            for region_id, record in list(self.rois.items()):
                if record.origin == origin_name and region_id in self.masks and record.contrib_id not in present:
                    del self.masks[region_id]
                    del self.rois[region_id]
        for label in ids:
            where = labels == label
            zz, yy, xx = np.nonzero(where)
            lo = (int(zz.min()), int(yy.min()), int(xx.min()))
            hi = (int(zz.max()) + 1, int(yy.max()) + 1, int(xx.max()) + 1)
            membership = where[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
            name = lv.label_names.get(label) or f"Label {label}"
            color = tuple(int(v) for v in lut[lut_row(label)])
            contrib_id = f"label:{label}"
            existing = next((r for r, rec in self.rois.items() if rec.origin == origin_name and rec.contrib_id == contrib_id), None)
            region_id = existing or self._new_region()
            self.masks[region_id] = Mask(region_id, label, name, color, tuple(o + l for o, l in zip(origin, lo)), np.ascontiguousarray(membership))
            record = pb2.RoiRecord(region_id=region_id, origin=origin_name, contrib_id=contrib_id, name=name)
            record.style.color = "#%02x%02x%02x" % color[:3]
            self.rois[region_id] = record
        # Disposed of: one use of each; the release follows at zero.
        self.release_loan(labels_loan)
        self.release_loan(lut_loan)
        self.send(pb2.HostMsg(id=request_id, label_volume_compiled=pb2.LabelVolumeCompiled(region_count=len(ids))))

    def _on_read_mask(self, request_id: str, request: pb2.ReadMaskVoxels) -> None:
        """§8.7: a bbox-local u8 membership array deposited into the `V`
        ring, with the mask's origin."""
        if request.region_id not in self.rois:
            return self._fail(request_id, pb2.Error.NOT_FOUND, f"no region {request.region_id!r}")
        mask = self.masks.get(request.region_id)
        if mask is None:
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, f"region {request.region_id!r} is not a mask")
        payload = mask.membership.astype(np.uint8).tobytes()
        deposited = self._deposit_v(request_id, payload)
        if deposited is None:
            return
        seg, offset = deposited
        self._install_ahead(seg)
        reply = pb2.MaskVoxels(
            segment=pb2.SegmentRef(segment_id=seg.segment_id, offset=offset, len=len(payload), shape_zyx=list(mask.membership.shape), dtype=pb2.DTYPE_U8),
            origin_zyx=list(mask.origin_zyx),
            region_id=mask.region_id,
            open_generation=self.dataset.open_generation,
        )
        self.send(pb2.HostMsg(id=request_id, mask_voxels=reply))

    def _on_read_roi_inclusion(self, request_id: str, request: pb2.ReadRoiInclusion) -> None:
        """Which voxels a region covers at frame `t`, the
        host resolving the shape, its inclusion mode, the pose row at `t` and
        the parent chain -- so a plugin measuring what it contributed does not
        re-implement the rasterizer and report different numbers from the
        same picture.  Answered like `MaskVoxels`: a `V` deposit the plugin
        releases (§5.6).

        The answers, as the `1.0` host gives them:

        - an unknown region: `NOT_FOUND`;
        - a `t` outside the DATASET (``0 <= t < T``): `OUT_OF_BOUNDS`;
        - a `t` inside the dataset but past the region's pose domain: the
          region is HELD at its nearest end row -- `out_of_span = HOLD` is
          the wire default -- and answered present with a deposit; only a
          `HIDE` region answers ``present = false``, with no segment, which
          owes no release;
        - a region present but covering no voxel of the dataset (clipped
          away entirely): ``present = true``, NO segment, ``voxel_count = 0``.

        **The clip box** (`roi_min_zyx`/`roi_max_zyx`,
        §8.7's table) -- level 0, min inclusive, max exclusive, both empty
        for the whole dataset, exactly `ReadDataVolume`'s semantics: other
        than 3 values at either end is `INVALID_ARGUMENT`; an empty box or
        one leaving the dataset is `OUT_OF_BOUNDS`; the answer is the
        region's voxels INSIDE the box (a region missing it is present with
        ``voxel_count = 0``).  A region whose clipped box holds more voxels
        than ``inclusion_voxel_budget`` is `OUT_OF_BOUNDS`.

        This emulation covers the shapes the examples use: a stored mask's
        membership, and a unit box at the region's pose row at `t`.
        """
        record = self.rois.get(request.region_id)
        if record is None:
            return self._fail(request_id, pb2.Error.NOT_FOUND, f"unknown region {request.region_id!r}")
        boxed = bool(request.roi_min_zyx or request.roi_max_zyx)
        if boxed and (len(request.roi_min_zyx) != 3 or len(request.roi_max_zyx) != 3):
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "roi_min_zyx/roi_max_zyx must each be 3 values")
        t = int(request.t)
        frames = self.dataset.shape_tczyx[0]
        if not 0 <= t < frames:
            return self._fail(request_id, pb2.Error.OUT_OF_BOUNDS, f"t={t} is outside the dataset's {frames} frames")
        extent = tuple(int(n) for n in self.dataset.shape_tczyx[2:])
        if boxed:
            box_lo = tuple(int(v) for v in request.roi_min_zyx)
            box_hi = tuple(int(v) for v in request.roi_max_zyx)
            if any(a >= b or b > n for a, b, n in zip(box_lo, box_hi, extent)):
                return self._fail(request_id, pb2.Error.OUT_OF_BOUNDS, f"box {list(box_lo)}..{list(box_hi)} is empty or leaves the dataset's level-0 shape {list(extent)}")
        else:
            box_lo, box_hi = (0, 0, 0), extent
        generation = self.dataset.open_generation
        mask = self.masks.get(request.region_id)
        if mask is not None:
            membership, origin = mask.membership.astype(np.uint8), tuple(mask.origin_zyx)
        else:
            domain = self.pose_domains.get(request.region_id)
            if domain is not None and not domain[0] <= t <= domain[1] and record.roi.out_of_span == _OUT_OF_SPAN_HIDE:
                # posed, HIDE, and not at this frame: a truthful "not here"
                return self.send(pb2.HostMsg(id=request_id, roi_inclusion=pb2.RoiInclusion(
                    region_id=request.region_id, t=t, open_generation=generation, present=False)))
            membership, origin = self._box_inclusion(record, t)
            if membership is None:
                return self._fail(request_id, pb2.Error.UNSUPPORTED,
                                  f"region {request.region_id!r} is a shape this emulation does not rasterize")
        # the region's bounding box clipped to the box (the dataset when none)
        lo = tuple(max(o, b) for o, b in zip(origin, box_lo))
        hi = tuple(min(o + n, b) for o, n, b in zip(origin, membership.shape, box_hi))
        if any(a >= b for a, b in zip(lo, hi)):
            membership, origin = np.zeros((0, 0, 0), dtype=np.uint8), (0, 0, 0)
        else:
            voxels = int(np.prod([b - a for a, b in zip(lo, hi)]))
            if voxels > self.inclusion_voxel_budget:
                return self._fail(request_id, pb2.Error.OUT_OF_BOUNDS, f"region {request.region_id!r} clipped to {list(lo)}..{list(hi)} is {voxels} voxels, past the host's budget of {self.inclusion_voxel_budget}; ask for a smaller box")
            membership = np.ascontiguousarray(membership[tuple(slice(a - o, b - o) for a, b, o in zip(lo, hi, origin))])
            origin = lo
        self.inclusion_requests.append((request.region_id, t))
        count = int(np.count_nonzero(membership))
        if count == 0:
            # present, covering nothing: no deposit, nothing to release
            return self.send(pb2.HostMsg(id=request_id, roi_inclusion=pb2.RoiInclusion(
                origin_zyx=[0, 0, 0], region_id=request.region_id, t=t, open_generation=generation,
                present=True, voxel_count=0)))
        deposited = self._deposit_v(request_id, membership.tobytes())
        if deposited is None:
            return
        seg, offset = deposited
        self._install_ahead(seg)
        self.send(pb2.HostMsg(id=request_id, roi_inclusion=pb2.RoiInclusion(
            segment=pb2.SegmentRef(segment_id=seg.segment_id, offset=offset, len=int(membership.nbytes),
                                   shape_zyx=list(membership.shape), dtype=pb2.DTYPE_U8),
            origin_zyx=list(origin), region_id=request.region_id, t=t,
            open_generation=generation, present=True,
            voxel_count=count)))

    def _aabb_inclusion(self, record: "pb2.RoiRecord"):
        """A ``"box"`` region with two or more vertices, as the EMULATOR
        rasterizes it (not vizi's rasterizer): every voxel whose center, an
        integer index, lies within the axis-aligned bounds of the vertices'
        positions, inclusive, clipped to the dataset.  Timeless: no pose.
        What `examples/region_quant/` measures (its SPEC says how its driver
        builds these)."""
        pos = np.array([(v.position.z, v.position.y, v.position.x) for v in record.roi.vertices], dtype=np.float64)
        extent = np.array(self.dataset.shape_tczyx[2:])
        lo = np.maximum(np.ceil(pos.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(pos.max(axis=0)).astype(int) + 1, extent)
        if np.any(hi <= lo):
            return np.zeros((0, 0, 0), dtype=np.uint8), (0, 0, 0)
        return np.ones(tuple(hi - lo), dtype=np.uint8), tuple(int(v) for v in lo)

    def _box_inclusion(self, record: "pb2.RoiRecord", t: int):
        """A unit box at the region's root, moved by its `pose_translation`
        row at `t` -- CLAMPED to the nearest end row outside the domain, the
        host's HOLD -- and clipped to the dataset: a box that lands outside
        covers nothing, which is ``(0, 0, 0)`` membership.  Enough for a
        driver to see that the host resolved the pose; not vizi's
        rasterizer."""
        if record.roi.type_name == "box" and len(record.roi.vertices) >= 2:
            return self._aabb_inclusion(record)
        rows = list(record.roi.pose_translation)
        shift = (0.0, 0.0, 0.0)
        domain = self.pose_domains.get(record.region_id)
        if rows and domain is not None:
            k = min(max(t, domain[0]), domain[1]) - domain[0]
            i = k * 3
            if 0 <= i <= len(rows) - 3:
                shift = tuple(rows[i : i + 3])
        origin = tuple(int(round(v)) for v in shift)
        extent = self.dataset.shape_tczyx[2:]
        if any(not 0 <= o < n for o, n in zip(origin, extent)):
            return np.zeros((0, 0, 0), dtype=np.uint8), (0, 0, 0)
        return np.ones((1, 1, 1), dtype=np.uint8), origin

    def _on_request_asset(self, request_id: str, request: pb2.RequestAsset) -> None:
        """§8.9: the user picks a file (or declines); the host records its
        size and digest at import time.

        While this is outstanding the session's command
        deadline is HELD -- the plugin is blocked on the host's own dialog, in
        front of the host's own user, which is not the "may be wedged" that
        the budget exists to catch.  `hold_picker` keeps the dialog up so a
        driver can exercise that, and `answer_picker` releases it.

        **One picker per session at a time** (§8.9): a
        `RequestAsset` while this session's dialog is still up is answered
        `OUT_OF_BOUNDS` and opens nothing (``picker_refusals``) -- a person
        answers one dialog at a time.  Only a held picker is ever "up" here;
        an unheld one is answered before the next message is read."""
        with self._lock:
            up = bool(self._picker_pending)
        if up:
            self.picker_refusals.append(request_id)
            return self._fail(request_id, pb2.Error.OUT_OF_BOUNDS, "a file picker is already open for this session; ask again once it is answered (§8.9)")
        self.asset_requests.append(request)
        if self._picker_held is not None:
            # The dialog is up.  Wait on a thread, not here: a real host keeps
            # serving while its own modal is on screen, and the plugin's
            # `RequestStatusQuery` is exactly the message that has to get
            # through while this is outstanding.
            with self._lock:
                self._picker_pending[request_id] = request.slot
                if request_id in self._open_requests:
                    self._open_requests[request_id] = True  # a person holds it
            held = self._picker_held
            threading.Thread(target=self._finish_picker, args=(request_id, request, held), daemon=True).start()
            return
        self._send_asset_result(request_id, request)

    def _on_request_save_path(self, request_id: str, request: pb2.RequestSavePath) -> None:
        """§8.9, as the host answers it: a
        `suggested_name` with a separator is `INVALID_ARGUMENT` ("a file name,
        not a path"); otherwise the save dialog, which shares `RequestAsset`'s
        one-picker-per-session rule and its held deadline (`hold_picker`).
        The "person" is `save_path_answer`: a path, or ``None`` to decline.
        The extension is appended when the chosen name lacks it.  The host
        creates nothing."""
        if "/" in request.suggested_name or "\\" in request.suggested_name:
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "suggested_name is a file name, not a path")
        with self._lock:
            up = bool(self._picker_pending)
        if up:
            self.picker_refusals.append(request_id)
            return self._fail(request_id, pb2.Error.OUT_OF_BOUNDS, "a file picker from this plugin is already open; ask again once it is answered")
        self.save_requests.append(request)

        def answer() -> None:
            chosen = self.save_path_answer
            if not chosen:
                result = pb2.SavePathResult(declined=True)
            else:
                path = Path(chosen).absolute()
                ext = request.extension.lstrip(".")
                if ext and path.suffix.lstrip(".").lower() != ext.lower():
                    path = path.with_name(f"{path.name}.{ext}")
                result = pb2.SavePathResult(path=str(path))
            self.send(pb2.HostMsg(id=request_id, save_path_result=result))

        if self._picker_held is not None:
            with self._lock:
                self._picker_pending[request_id] = "(save)"
                if request_id in self._open_requests:
                    self._open_requests[request_id] = True
            held = self._picker_held

            def later() -> None:
                held.wait()
                with self._lock:
                    self._picker_pending.pop(request_id, None)
                answer()

            threading.Thread(target=later, daemon=True).start()
            return
        answer()

    # -- set-view (§8.15), as the host judges it --

    _CAUSE_LIFETIME_S = 2.0
    _TARGET_TOUCHES = {
        "t": ("t",), "z": ("z",), "select_region": ("selection",),
        "frame_region": ("selection", "camera"), "look_at": ("camera",), "orbit_point": ("camera",),
    }
    #: Every target.
    _APPLIED_TARGETS = frozenset({"t", "z", "select_region", "frame_region", "look_at", "orbit_point"})

    def _panel_spec(self, panel_id: str) -> dict:
        return next(p for p in self.manifest.get("contributions", {}).get("panels", []) if p["id"] == panel_id)

    def invoke_widget(self, panel_id: str, widget_id: str, timeout: float = COMMAND_WINDOW_S, *, row: str = "", **args) -> pb2.CommandDone:
        """The person interacts with a panel widget: its `command` fires with
        ``args`` (``row`` added for a list's cell or a selectable list), and
        for a `moves_view` widget with a FRESH `cause` (§8.15), minted as
        the host mints one -- a newer cause from the same widget
        and row supersedes an unanswered older one.  ``last_cause`` is it."""
        kind, spec = self._panel_widgets[panel_id][widget_id]
        list_id = spec.get("_list")
        if list_id is not None and "command" not in spec:
            spec = self._panel_widgets[panel_id][list_id][1]
        command = spec["command"]
        if kind == "roi_picker" and "value" in args:
            # As the host's picker does: a region choice sends its name beside the
            # id, and one that follows the selection IS the person's
            # selection, made before the command fires.
            region_id = str(args["value"])
            with self._lock:
                record = self.rois.get(region_id)
            args = {**args, "name": record.name if record is not None else region_id}
            if spec.get("follows_selection"):
                self.select_region_as_person(region_id)
        if kind == "slider" and "value" in args:
            # The host's slider stops only where its step allows (§3.3.1);
            # ``last_readout`` is what it shows beside the handle.
            args = {**args, "value": slider_value(spec, args["value"])}
            self.last_readout = slider_readout(spec, args["value"])
        cause = 0
        if spec.get("moves_view") and self.dataset_open:
            self._forget_expired_causes()
            stream = (panel_id, widget_id, row)
            for c in self._causes.values():
                if c["stream"] == stream and c["answer"] is None:
                    c["superseded"] = True
            cause = self._next_cause
            self._next_cause += 1
            self._causes[cause] = {
                "surface": panel_id, "targets": frozenset(self._panel_spec(panel_id).get("view_targets", [])),
                "stream": stream, "minted_at": time.monotonic(), "serial": self._touch_serial,
                "generation": int(self.dataset.open_generation), "superseded": False, "answer": None,
            }
        self.last_cause = cause
        if row:
            args = {"row": row, **args}
        typed = {key: _value(value) for key, value in args.items()}
        return self.request(timeout, invoke_command=pb2.InvokeCommand(command_id=command, args=typed, cause=cause)).command_done

    def _forget_expired_causes(self) -> None:
        """As the host forgets them: a cause is kept ten
        lifetimes (20 s) so a late repeat gets its answer, then forgotten --
        and a request naming it is `INVALID`, not `STALE`."""
        horizon = time.monotonic() - 10 * self._CAUSE_LIFETIME_S
        for causes in (self._causes, self._tool_causes):
            for key in [k for k, c in causes.items() if c["minted_at"] < horizon]:
                del causes[key]

    def select_region_as_person(self, region_id: str) -> None:
        """The person selects a region in a control vizi draws -- the Regions
        panel, or a `roi_picker` that follows the selection: it
        touches the selection (supersedes older selection causes, ends a
        Follow driving it) and `selected_region_id` follows."""
        self.person_touches("selection")
        self.dataset.selected_region_id = region_id

    def person_touches(self, *touches: str) -> None:
        """The person changed these (``"t"``, ``"z"``, ``"selection"``,
        ``"camera"``): a cause minted before may not change them
        (`SUPERSEDED`, "the person always wins")."""
        self._touch_serial += 1
        for touch in touches:
            self._touched[touch] = self._touch_serial
        # A Follow driving any of them ends, all of it (§8.15).
        with self._follow_lock:
            if self.follow is not None and self.follow["touches"] & set(touches):
                self.end_follow("USER_TOOK_CONTROL", "the person took control of the view")

    def _switch_panels(self) -> list:
        """The panels that draw the switch: those declaring `view_targets` or
        `follow_targets`."""
        return [p["id"] for p in self.manifest.get("contributions", {}).get("panels", [])
                if p.get("view_targets") or p.get("follow_targets")]

    def _panel_showing(self, panel_id: str) -> bool:
        return panel_id in self.opened_panels and panel_id not in self.hidden_panels

    def _switch_on(self, cause: dict) -> bool:
        """As the host reads it: the person's switch is the first row of a
        panel -- host-drawn, never sent -- and counts only while that panel
        is open and showing.  A panel cause needs its own panel's switch; a
        tool cause, the switch of a panel that declares `view_targets` --
        not one that only offers a Follow (the manifest rule already
        requires that panel)."""
        if cause["surface"] is not None:
            return self.view_toggles.get(cause["surface"], False) and self._panel_showing(cause["surface"])
        governing = [p["id"] for p in self.manifest.get("contributions", {}).get("panels", []) if p.get("view_targets")]
        return any(self.view_toggles.get(p, False) and self._panel_showing(p) for p in governing)

    def _judge_view(self, req: pb2.ViewRequest) -> tuple[str, str, str, dict]:
        targets = {}
        for name in ("t", "z", "select_region", "frame_region", "look_at", "orbit_point"):
            if req.HasField(name):
                targets[name] = getattr(req, name)
        has_tool = req.HasField("tool_cause")
        if bool(req.cause) == has_tool:
            return "REFUSED", "INVALID", "a view request names exactly one cause", {}
        if req.decline and targets:
            return "REFUSED", "INVALID", "a decline moves nothing, so names no target", {}
        if not req.decline and not targets:
            return "REFUSED", "INVALID", "a view request names a target or declines", {}
        if has_tool:
            cause = self._tool_causes.get((int(req.tool_cause.tool), int(req.tool_cause.interaction)))
        else:
            cause = self._causes.get(int(req.cause))
        if cause is None:
            return "REFUSED", "INVALID", "not a cause this host minted for this plugin", {}
        fingerprint = req.SerializeToString(deterministic=True)
        if cause["answer"] is not None:
            earlier, judgement = cause["answer"]
            if earlier == fingerprint:
                return (*judgement[:3], {})  # the same answer, not applied twice
            judgement = ("REFUSED", "CONSUMED", "that cause was answered", {})
        elif self.view_script is not None:
            judgement = (*self.view_script, targets if self.view_script[0] == "APPLIED" else {})
        else:
            judgement = self._view_rule(cause, req, targets)
        cause["answer"] = (fingerprint, judgement)
        return judgement

    def _view_rule(self, cause: dict, req: pb2.ViewRequest, targets: dict):
        if time.monotonic() - cause["minted_at"] > self._CAUSE_LIFETIME_S:
            return "REFUSED", "STALE", f"the cause is older than {self._CAUSE_LIFETIME_S:g} s", {}
        if cause["generation"] != int(self.dataset.open_generation):
            return "REFUSED", "DATASET_CHANGED", "a different dataset since the cause", {}
        if req.decline:
            return "DECLINED", "NONE", "", {}
        undeclared = sorted(set(targets) - cause["targets"])
        if undeclared:
            return "REFUSED", "NOT_DECLARED", f"the panel does not declare {', '.join(undeclared)}", {}
        unapplied = sorted(set(targets) - self._APPLIED_TARGETS)
        if unapplied:
            return "REFUSED", "UNSUPPORTED", f"this version of vizi does not move {', '.join(unapplied)} yet", {}
        if not self._switch_on(cause):
            return "REFUSED", "NOT_ENABLED", "the person has not let this panel move the view", {}
        if self.export_active:
            return "REFUSED", "EXPORT_ACTIVE", "a movie export owns the view", {}
        touched = {t for name in targets for t in self._TARGET_TOUCHES[name]}
        with self._follow_lock:
            if self.follow is not None and self.follow["touches"] & touched:
                return "REFUSED", "FOLLOW_ACTIVE", "a Follow is driving that", {}
        conflict = _targets_conflict(targets)
        if conflict is not None:
            return "REFUSED", "INVALID", conflict, {}
        if cause["superseded"] or any(self._touched.get(t, 0) > cause["serial"] for t in touched):
            return "REFUSED", "SUPERSEDED", "the person has changed that since, and the person wins", {}
        for name, value in targets.items():
            problem = self._check_target(name, value)
            if problem is not None:
                return "REFUSED", problem[0], problem[1], {}
        return "APPLIED", "NONE", "", targets

    def _check_target(self, name: str, value) -> "tuple[str, str] | None":
        """As the host checks a target: ``None``, or ``(reason, detail)`` for a
        target that cannot apply -- shared by set-view and Follow."""
        frames, _, depth, height, width = self.dataset.shape_tczyx
        if name in ("t", "z"):
            size = frames if name == "t" else depth
            if not 0 <= int(value) < size:
                return ("OUT_OF_RANGE", f"{name}={value} is outside 0..{size - 1}")
        if name in ("select_region", "frame_region") and str(value) not in self.rois:
            return ("UNKNOWN_REGION", f"no region {value!r}")
        if name in ("look_at", "orbit_point"):
            point = list(value.target_zyx if name == "look_at" else value.zyx)
            if len(point) != 3:
                return ("INVALID", f"{name} needs 3 ZYX values")
            if not all(-0.5 <= v <= n - 0.5 for v, n in zip(point, (depth, height, width))):
                return ("OUT_OF_RANGE", f"{name} {point} is outside the dataset")
            if name == "look_at":
                if value.fit_extent < 0:
                    return ("INVALID", "look_at fit_extent is negative")
                direction = list(value.direction_zyx)
                if direction and (len(direction) != 3 or not any(direction)):
                    return ("INVALID", "look_at direction needs 3 values, not all 0")
        return None

    def _apply_view(self, targets: dict) -> None:
        """Move the emulator's view as vizi's gestures would: `t`/`z` on the
        dataset, the region targets as the selection, the camera targets in
        ``camera``.  Set-view and Follow both land here."""
        if "t" in targets:
            self.dataset.t = int(targets["t"])
        if "z" in targets:
            self.dataset.z = int(targets["z"])
        for name in ("select_region", "frame_region"):
            if name in targets:
                self.dataset.selected_region_id = str(targets[name])
        for name in ("look_at", "orbit_point"):
            if name in targets:
                self.camera[name] = targets[name]
        if "look_at" in targets and "orbit_point" not in targets:
            # §8.15: the look_at target becomes the orbit point unless the
            # same request sets one.
            self.camera["orbit_point"] = pb2.OrbitPoint(zyx=list(targets["look_at"].target_zyx))

    def _on_view_request(self, request_id: str, req: pb2.ViewRequest) -> None:
        """One `ViewRequest`'s one answer (§8.15).  APPLIED moves the
        emulator's view as vizi's gestures would: `t`/`z` on the dataset, and
        `select_region`/`frame_region` as the selection; `ViewChanged` follows
        on the next tick.  Every request and its result is in
        ``view_requests``; every applied move in ``view_moves``."""
        outcome, reason, detail, targets = self._judge_view(req)
        if outcome == "APPLIED" and targets:
            self._apply_view(targets)
            with self._lock:
                self.view_moves.append(dict(targets))
        result = pb2.ViewRequestResult(
            cause=req.cause,
            outcome=pb2.ViewOutcome.Value(f"VIEW_OUTCOME_{outcome}"),
            reason=pb2.ViewRefusal.Value(f"VIEW_REFUSAL_{reason}"),
            detail=detail,
        )
        if req.HasField("tool_cause"):
            result.tool_cause.CopyFrom(req.tool_cause)
        with self._lock:
            self.view_requests.append((req, result))
        self.send(pb2.HostMsg(id=request_id, view_request_result=result))

    # -- Follow (§8.15), as the host runs it --

    #: §8.15: states are applied at up to this rate; a faster stream coalesces.
    FOLLOW_APPLY_INTERVAL_S = 1.0 / 30.0

    @property
    def view_toggle(self) -> bool:
        """Whether any panel's switch is on.  Setting it sets every panel's
        switch at once (a driver's shorthand; `set_view_toggle` for one)."""
        return any(self.view_toggles.values())

    @view_toggle.setter
    def view_toggle(self, on: bool) -> None:
        for panel_id in self._switch_panels():
            self.set_view_toggle(panel_id, on)

    def set_view_toggle(self, panel_id: str, on: bool) -> None:
        """The person allows or withdraws ``panel_id``'s plugin moving the
        view (never sent).  vizi keeps this choice per plugin; the emulator
        keeps it per panel and starts it off at every open.  Turning it off
        ends a Follow the plugin is running (`STOPPED`): it is the Follow
        button's master."""
        self.view_toggles[panel_id] = bool(on)
        if not on:
            with self._follow_lock:
                if self.follow is not None:
                    self.end_follow("STOPPED", "the person turned off the plugin's view control")

    def _on_offer_follow(self, offer: pb2.OfferFollow) -> None:
        """An offer, as the host takes it: one from a panel with no
        `follow_targets` is logged and ignored; ``""`` withdraws, ending a
        Follow running from that offer (`WITHDRAWN`)."""
        spec = next((p for p in self.manifest.get("contributions", {}).get("panels", []) if p["id"] == offer.panel_id), None)
        if spec is None or not spec.get("follow_targets"):
            self.diagnostics.append(f"a Follow offered from panel {offer.panel_id!r}, which declares no follow_targets")
            return
        with self._follow_lock:
            if offer.label:
                self.follow_offers[offer.panel_id] = offer.label
                return
            self.follow_offers.pop(offer.panel_id, None)
            if self.follow is not None and self.follow["panel_id"] == offer.panel_id:
                self.end_follow("WITHDRAWN", "the plugin withdrew its offer")

    def press_follow(self, panel_id: str, *, resumed: bool = False) -> "int | None":
        """The person presses Follow.  Nothing happens
        unless the panel is open, the switch is on, the panel has an offer and
        no export runs -- the button is not there to press.  A running Follow
        from another panel is ended `REPLACED` (one runs host-wide); pressing
        the button of the panel whose Follow is running presses Stop, and
        ends it `STOPPED`.  Returns the
        new follow_id, or ``None``."""
        spec = self._panel_spec(panel_id)
        with self._follow_lock:
            if self.follow is not None and self.follow["panel_id"] == panel_id:
                self.end_follow("STOPPED", "the person pressed Stop")
                return None
            if (
                not self._panel_showing(panel_id)
                or not self.view_toggles.get(panel_id, False)
                or not self.follow_offers.get(panel_id)
                or not self.dataset_open
                or self.export_active
            ):
                return None
            if self.follow is not None:
                self.end_follow("REPLACED", "another Follow started")
            targets = list(spec.get("follow_targets", []))
            follow_id = self._next_follow
            self._next_follow += 1
            self.follow = {
                "follow_id": follow_id, "panel_id": panel_id, "targets": frozenset(targets),
                "touches": frozenset(t for name in targets for t in self._TARGET_TOUCHES[name]),
                "last_seq": 0, "pending": None, "last_applied_at": None,
                "generation": int(self.dataset.open_generation),
            }
            self.resumable = None
            started = pb2.FollowStarted(follow_id=follow_id, panel_id=panel_id, targets=targets, resumed=resumed)
            self.follows_started.append(started)
            if not self._follow_pump_on:
                self._follow_pump_on = True
                threading.Thread(target=self._follow_pump, daemon=True, name="hostsim-follow").start()
        self.send(pb2.HostMsg(follow_started=started))
        return follow_id

    def resume_follow(self) -> "int | None":
        """The person presses Resume, offered after they took control: the
        same offer started again with ``resumed: true``."""
        if self.resumable is None:
            return None
        return self.press_follow(self.resumable, resumed=True)

    def stop_follow(self) -> None:
        """The person presses Stop."""
        self.end_follow("STOPPED", "the person pressed Stop")

    def close_panel(self, panel_id: str) -> None:
        """The person closes the panel's dock.  In vizi that HIDES it:
        its patches and images keep applying, and reopening it from its
        menu entry is an open -- `PanelOpened`, the switch off.  While
        hidden its switch counts for nothing, and a Follow running from it
        ends `PANEL_CLOSED`."""
        with self._lock:
            self.hidden_panels.add(panel_id)
        with self._follow_lock:
            if self.follow is not None and self.follow["panel_id"] == panel_id:
                self.end_follow("PANEL_CLOSED", "the panel was closed or hidden")

    def end_follow(self, reason: str, detail: str = "") -> None:
        """End the running Follow for ``reason`` (a `FollowEndReason` name
        without its prefix), telling the plugin once.  Drivers script the
        window's own claims with it: `AUTOSPIN`, `EXPORT`, `DATASET_CHANGED`."""
        with self._follow_lock:
            running, self.follow = self.follow, None
            if running is None:
                return
            if reason == "USER_TOOK_CONTROL":
                self.resumable = running["panel_id"]
            ended = pb2.FollowEnded(follow_id=running["follow_id"], reason=pb2.FollowEndReason.Value(f"FOLLOW_END_{reason}"), detail=detail)
            self.follows_ended.append(ended)
        if not self.stream_ended.is_set():
            self.send(pb2.HostMsg(follow_ended=ended))

    def _on_follow_state(self, state: pb2.FollowState) -> None:
        """A state, as the host accepts it: held for the next application if
        it is for the running Follow and newer than both the last applied and
        the one pending; otherwise dropped."""
        with self._follow_lock:
            running = self.follow
            why = None
            if running is None or int(state.follow_id) != running["follow_id"]:
                why = "not running"
            elif int(state.seq) <= running["last_seq"]:
                why = "seq not above the last applied"
            elif running["pending"] is not None and int(running["pending"].seq) >= int(state.seq):
                why = "seq not above the pending one"
            if why is not None:
                self.follow_dropped.append((int(state.follow_id), int(state.seq), why))
                return
            running["pending"] = state
        self._tick_follow()

    def _follow_pump(self) -> None:
        while not self.stream_ended.is_set():
            try:
                self._tick_follow()
            except Exception as exc:  # noqa: BLE001 -- the pump outlives one bad tick
                self.diagnostics.append(f"follow tick raised {exc!r}")
            time.sleep(0.033)  # vizi's 33 ms follow timer
        self._follow_pump_on = False

    @staticmethod
    def _state_targets(state) -> dict:
        return {name: getattr(state, name) for name in ("t", "z", "select_region", "frame_region", "look_at", "orbit_point") if state.HasField(name)}

    def _tick_follow(self) -> None:
        """As the host applies Follow: the newest pending state, at most 30
        times a second; one that cannot apply ends the Follow `INVALID`."""
        with self._follow_lock:
            running = self.follow
            if running is None:
                return
            if int(self.dataset.open_generation) != running["generation"]:
                self.end_follow("DATASET_CHANGED", "a different dataset is open")
                return
            if running["pending"] is None:
                return
            now = time.monotonic()
            last = running["last_applied_at"]
            if last is not None and now - last < 0.9 * self.FOLLOW_APPLY_INTERVAL_S:
                return
            state, running["pending"] = running["pending"], None
            running["last_seq"] = int(state.seq)
            running["last_applied_at"] = now
            targets = self._state_targets(state)
            undeclared = sorted(set(targets) - running["targets"])
            problem = None
            if not targets:
                problem = "a Follow state names no target"
            elif undeclared:
                problem = f"the Follow does not drive {', '.join(undeclared)}"
            elif _targets_conflict(targets) is not None:
                problem = _targets_conflict(targets)
            else:
                for name, value in targets.items():
                    found = self._check_target(name, value)
                    if found is not None:
                        problem = found[1]
                        break
            if problem is not None:
                self.end_follow("INVALID", problem)
                return
            self._apply_view(targets)
            self.follow_applied.append((running["follow_id"], int(state.seq), dict(targets)))

    # -- the subscribed pushes (§8.7) --

    def close_dataset(self) -> None:
        """The person closes the dataset: no cause is minted
        until one opens, and a subscribed `ViewChanged` says generation 0.
        An asked edit in progress ends `CANCELLED` (§8.8)."""
        self.dataset_open = False
        for region_id in {d.region_id for d in self.shape_drags.values() if d.ended is None}:
            self._cancel_drags_of(region_id, "the dataset closed")

    def open_dataset(self) -> None:
        """A dataset opens again: a NEW generation, even for the same file
        (§8.7)."""
        self.dataset.open_generation = int(self.dataset.open_generation) + 1
        self.dataset_open = True

    def region_revisions(self) -> dict[str, int]:
        """Every region's `revision`:
        new whenever a different record sits under the id or `_touch` retired
        the last, from one counter, never reused."""
        with self._lock:
            out: dict[str, int] = {}
            for region_id, record in self.rois.items():
                entry = self._region_revisions.get(region_id)
                if entry is None or entry[0] is not record:
                    self._region_counter += 1
                    entry = (record, self._region_counter)
                    self._region_revisions[region_id] = entry
                out[region_id] = entry[1]
            for gone in self._region_revisions.keys() - out.keys():
                del self._region_revisions[gone]
            return out

    def _touch(self, region_id: str) -> None:
        with self._lock:
            self._region_revisions.pop(region_id, None)

    def add_region(self, roi, name: str = "", origin: str = "user") -> str:
        """A region the PERSON made (a driver's stand-in): stored, and seen
        by the next `RegionsChanged` tick."""
        record = pb2.RoiRecord(region_id=self._new_region(), origin=origin, name=name)
        record.roi.CopyFrom(roi)
        with self._lock:
            self.rois[record.region_id] = record
        return record.region_id

    def delete_region(self, region_id: str) -> None:
        with self._lock:
            self.rois.pop(region_id, None)
            self.masks.pop(region_id, None)
        self._cancel_drags_of(region_id, "the region was deleted")

    def select_region(self, region_id: str) -> None:
        """The region selection changes (``""`` for none): `ViewState` and the
        next `ViewChanged` say so.  It does not model the person's touch; see
        `select_region_as_person` for that."""
        self.dataset.selected_region_id = region_id

    def edit_region(self, region_id: str, *, name: "str | None" = None, color: "str | None" = None, visible: "bool | None" = None) -> None:
        """The person renames, recolors or hides a region IN PLACE:
        the record under the id is the same region, so its `revision`
        is new and its `content_hash` (over the `Roi`) is unchanged."""
        with self._lock:
            record = self.rois[region_id]
            if name is not None:
                record.name = name
            if color is not None:
                record.style.color = color
            if visible is not None:
                record.style.visible = visible
        self._touch(region_id)

    # -- generic regions and the types a plugin shapes --

    def drag_handle(self, region_id: str, handle: int, p1) -> int:
        """The person drags ``handle`` of a DECLARED region to ``p1``: the
        host enacts it from the region's data alone (§8.8; the ROI scheme's
        §4 order, then one level of `LinearBinding`s), as one edit.  Returns
        the `Resolution`.  An asked region's drag is :meth:`begin_shape_drag`."""
        with self._lock:
            record = self.rois[region_id]
            if record.roi.type_name in regions.BUILTIN_TYPES or record.type_owner:
                raise ValueError(f"region {region_id} ({record.roi.type_name!r}) is not a declared region")
            if not record.roi.vertices[handle].HasField("handle"):
                raise ValueError(f"vertex {handle} of region {region_id} is not a handle")
            moved, resolution = regions.drag_declared(record.roi, handle, p1)
            fresh = pb2.RoiRecord()
            fresh.CopyFrom(record)
            fresh.roi.CopyFrom(moved)
            self.rois[region_id] = fresh  # a new record: a new revision
        return resolution

    def begin_shape_drag(self, region_id: str, handle: int) -> ShapeDrag:
        """The person presses on ``handle`` of an ASKED region (its type one
        of the plugin's `region_types`): a new `drag_id`, nothing sent yet."""
        with self._lock:
            record = self.rois[region_id]
            if record.type_owner != self.plugin_id:
                raise ValueError(f"region {region_id} ({record.roi.type_name!r}) is not shaped by {self.plugin_id}")
            if not record.roi.vertices[handle].HasField("handle"):
                raise ValueError(f"vertex {handle} of region {region_id} is not a handle")
            if any(d.region_id == region_id and d.ended is None for d in self.shape_drags.values()):
                raise ValueError(f"region {region_id} is already being edited")
            self._drag_seq += 1
            drag = ShapeDrag(self, self._drag_seq, record, handle)
            self.shape_drags[drag.drag_id] = drag
        return drag

    def spinbox_edit(self, region_id: str, handle: int, p1, *, supervised: bool = True, timeout: "float | None" = None) -> tuple[ShapeDrag, pb2.ShapeEditEnded]:
        """A spinbox edit in the generic vertex editor: a drag with no
        `shape_drag` before its `shape_commit` (§8.8)."""
        drag = self.begin_shape_drag(region_id, handle)
        p = regions.project_onto_dof(drag.mode, drag.p0, drag.axis, p1)
        return drag, drag._commit(p, supervised, timeout)

    def _end_drag(self, drag: ShapeDrag, outcome: int, detail: str, positions=None) -> pb2.ShapeEditEnded:
        with self._lock:
            if drag.ended is not None:
                return drag.ended
            if positions is not None and drag.region_id in self.rois:
                fresh = pb2.RoiRecord()
                fresh.CopyFrom(self.rois[drag.region_id])
                regions.set_positions(fresh.roi, positions)
                self.rois[drag.region_id] = fresh  # one undo step, a new revision
            drag.ended = pb2.ShapeEditEnded(drag_id=drag.drag_id, outcome=outcome, detail=detail)
        drag._answered.set()
        self.send(pb2.HostMsg(shape_edit_ended=drag.ended))
        return drag.ended

    def _cancel_drags_of(self, region_id: str, why: str) -> None:
        for drag in [d for d in self.shape_drags.values() if d.region_id == region_id and d.ended is None]:
            self._end_drag(drag, pb2.ShapeEditEnded.CANCELLED, why)

    def _on_shape_result(self, result: pb2.ShapeResult, *, answer: bool) -> None:
        """A `shape_proposal` or `shape_answer` (§8.8): out of step is a
        `shape_out_of_step` violation (a `drag_id` never sent, a `seq` never
        sent in that drag, a second answer to one commit, or -- this
        emulator's reading -- an answer before any commit); a reply for an
        ended drag, a proposal after the commit, and a proposal older than
        the one on screen are dropped silently; a bad shape is
        `shape_refused`."""
        with self._lock:
            (self.shape_answers if answer else self.shape_proposals).append(result)
            drag = self.shape_drags.get(int(result.drag_id))
        kind = "shape_answer" if answer else "shape_proposal"
        if drag is None:
            return self.refuse("shape_out_of_step", f"{kind} names drag {result.drag_id}, which this session never began")
        if answer:
            if not drag.committing:
                if drag.ended is not None:
                    return  # the drag ended before any commit: dropped silently
                return self.refuse("shape_out_of_step", f"shape_answer for drag {drag.drag_id} before its shape_commit")
            drag.answers += 1
            if drag.answers > 1:
                return self.refuse("shape_out_of_step", f"a second shape_answer to drag {drag.drag_id}'s commit")
            if drag.ended is not None:
                return  # ended first (canceled, deleted, timed out): dropped silently
            if result.resolution == roi_pb2.REJECTED:
                return self._end_drag(drag, pb2.ShapeEditEnded.REVERTED, result.message or "the plugin rejected the shape")
            why = regions.shape_result_problem(drag.roi, result)
            if why is not None:
                self.refuse("shape_refused", f"drag {drag.drag_id}: {why}")
                return self._end_drag(drag, pb2.ShapeEditEnded.REVERTED, f"shape refused: {why}")
            positions = np.array([[p.z, p.y, p.x] for p in result.positions], dtype=np.float64)
            return self._end_drag(drag, pb2.ShapeEditEnded.COMMITTED, "", positions)
        if drag.ended is not None or drag.committing:
            return  # the drag ended, or its commit is out: dropped silently
        if int(result.seq) not in drag.sent:
            return self.refuse("shape_out_of_step", f"shape_proposal for drag {drag.drag_id} names seq {result.seq}, never sent")
        if result.seq <= drag.on_screen_seq or result.resolution == roi_pb2.REJECTED:
            return  # older than the shape on screen, or no shape for it: the last stays
        why = regions.shape_result_problem(drag.roi, result)
        if why is not None:
            return self.refuse("shape_refused", f"drag {drag.drag_id} seq {result.seq}: {why}")
        with self._lock:
            drag.on_screen = np.array([[p.z, p.y, p.x] for p in result.positions], dtype=np.float64)
            drag.on_screen_seq = int(result.seq)

    def _subscription_pump(self) -> None:
        while not self.stream_ended.is_set():
            try:
                self._tick_subscriptions()
            except Exception as exc:  # noqa: BLE001 -- the pump outlives one bad tick
                self.diagnostics.append(f"subscription tick raised {exc!r}")
            time.sleep(0.04)  # the host's pump tick (doorbell_protocol.md §11)

    def _tick_subscriptions(self) -> None:
        """As the host sends subscriptions, one tick: each subscribed state
        restated and sent only when it changed.  ``reset`` on the first push
        and on a new `open_generation`, with every region listed; a deleted
        region is `deleted` with revision 0 and origin ``""``."""
        gen = int(self.dataset.open_generation)
        if "view" in self.subscriptions:
            if not self.dataset_open:
                # §8.7: a close is a push at generation 0 with
                # nothing else set.
                state = (0,)
                msg = pb2.ViewChanged(open_generation=0)
            else:
                view = self.dataset.view_state()
                applied = view.filter.SerializeToString(deterministic=True) if view.HasField("filter") else None
                state = (gen, int(view.t), int(view.z), tuple(view.visible_channels), str(view.selected_region_id), applied)
                msg = pb2.ViewChanged(open_generation=state[0], t=state[1], z=state[2], visible_channels=list(state[3]), selected_region_id=state[4])
                if applied is not None:
                    msg.filter.CopyFrom(view.filter)
            if self._view_sent != state:
                self.send(pb2.HostMsg(view_changed=msg))
                with self._lock:
                    self.view_pushes.append(msg)
                self._view_sent = state
        if "regions" in self.subscriptions:
            revisions = self.region_revisions()
            with self._lock:
                origins = {rid: r.origin for rid, r in self.rois.items()}
            sent = self._regions_sent
            reset = sent is None or sent[0] != gen
            before = {} if reset else sent[1]
            changes = [
                pb2.RegionChange(region_id=rid, revision=rev, origin=origins.get(rid, ""))
                for rid, rev in revisions.items()
                if before.get(rid) != rev
            ]
            changes += [pb2.RegionChange(region_id=rid, deleted=True) for rid in before.keys() - revisions.keys()]
            if reset or changes:
                msg = pb2.RegionsChanged(open_generation=gen, reset=reset, changes=changes)
                self.send(pb2.HostMsg(regions_changed=msg))
                with self._lock:
                    self.region_pushes.append(msg)
                self._regions_sent = (gen, revisions)

    def _finish_picker(self, request_id: str, request: pb2.RequestAsset, held: threading.Event) -> None:
        held.wait()
        with self._lock:
            self._picker_pending.pop(request_id, None)
        if self._picker_abandoned:
            # §8.9: the plugin gave up first, the reply still goes
            # out against the retired id, and the receiver drops it silently
            # rather than faulting.  That is the thing worth driving.
            self.diagnostics.append(f"late asset_result for {request_id}, sent after the waiter left")
        self._send_asset_result(request_id, request)

    def _send_asset_result(self, request_id: str, request: pb2.RequestAsset) -> None:
        import hashlib

        path = self.assets.get(request.slot)
        if path is None:
            return self.send(pb2.HostMsg(id=request_id, asset_result=pb2.AssetResult(slot=request.slot, declined=True)))
        data = Path(path).read_bytes()
        self.send(pb2.HostMsg(id=request_id, asset_result=pb2.AssetResult(slot=request.slot, path=str(path), size=len(data), sha256=hashlib.sha256(data).hexdigest())))

    def _take_loan(self, ref: pb2.SegmentRef) -> Loan:
        """Read the bytes a push's descriptor names, out of the plugin's arena,
        bounds-checked against the region -- what the host does before it
        believes a single byte (§11).  A descriptor into a segment this
        session does not hold is a ValueError like any other bad descriptor."""
        with self._lock:
            seg = self.segments.get(ref.segment_id)
        if seg is None:
            raise ValueError(f"descriptor names segment {ref.segment_id}, which is not granted")
        region = seg.region_at(ref.offset)
        if not isinstance(region, PluginArena):
            raise ValueError(f"descriptor into segment {ref.segment_id} names no plugin arena")
        return Loan(ref.segment_id, region.index, ref.offset, ref.len, region.read(ref.offset, ref.len), int(ref.uses))

    def _drop_if_retiring(self, what: str, refs) -> bool:
        """A message naming a segment this host
        retired and the plugin has not yet released is dropped quietly, its
        loans returned.  ``refs`` are the `SegmentRef`s it names."""
        with self._lock:
            hit = next((r.segment_id for r in refs if r.segment_id in self.retiring_segments), None)
        if hit is None:
            return False
        for ref in refs:
            try:
                self.release_loan(self._take_loan(ref))
            except ValueError:
                pass
        with self._lock:
            self.dropped_retiring.append((what, hit))
        return True

    def _on_overlay_update(self, update: pb2.OverlayUpdate) -> None:
        refs = [buf.segment for buf in _geom_buffers(update) if buf.WhichOneof("source") == "segment"]
        if self._drop_if_retiring(f"overlay {update.overlay_id}", refs):
            return
        push = Push("update", update.overlay_id, update.revision, update)
        for buf in _geom_buffers(update):
            if buf.WhichOneof("source") == "segment":
                try:
                    push.loans.append(self._take_loan(buf.segment))
                except ValueError as exc:
                    # vizi refuses the push on `overlay_item_refused` and
                    # disposes of what it could note.
                    for loan in push.loans:
                        self.release_loan(loan)
                    return self.refuse("overlay_item_refused", f"{update.overlay_id}: descriptor: {exc}")
        if update.overlay_id in self.chunked:
            # §5.7 rule 3 / §8.5: a chunked overlay's unit of state is the
            # chunk; an `OverlayUpdate` to one is refused, its loans released.
            for loan in push.loans:
                self.release_loan(loan)
            return self.refuse("overlay_chunk_refused", f"OverlayUpdate to chunked overlay {update.overlay_id!r}: its state is OverlayChunk and OverlayLabelTable")
        self.pushes.append(push)
        # §5.6: applied is a disposition; every loan the message named is
        # decremented, and released once its declared uses reach zero.
        for loan in push.loans:
            self.release_loan(loan)

    def _on_panel_patch(self, patch: pb2.PanelPatch, size: int = 0) -> None:
        """A `PanelPatch`, op by op, as the host applies it: each
        op is its own statement about one widget, so a refused op does not
        take its neighbors with it, and every op NOT applied is reported to
        the plugin as `ContractViolation{panel_patch_refused}`
        (kept in ``refusals``).  Refused, as the host refuses them:

        - the whole patch, for a panel this session has not OPENED -- the
          host holds no live panel for it (`open_panel` first);
        - an op naming a widget the panel does not declare;
        - an op the widget's kind does not take: `set_value` on anything but
          slider (a number), checkbox, dropdown (an option it holds, or an
          index in range), line_edit, label, button; `set_progress` on
          anything but progress; `set_options` on anything but dropdown;
          `set_image` on anything but image;
        - a `set_image` whose ref is not into this panel's `P` segment, or
          whose bytes are not a PNG.

        A refused op is not in ``patches``.  Every image ref the patch
        names is released either way (§5.6: refused is a disposition).

        First of all, §8.4's bound: a `PanelPatch` carrying a table op is at
        most `TABLE_MAX_PATCH_BYTES`, read on the message as the plugin sent
        it (``size``, the encoded `PanelPatch`), before the host splits it per
        widget.  Past it, each `set_table` and `patch_table` is refused
        `INVALID_ARGUMENT` and the patch's other ops still apply.  Then the
        transport: with ``hold_table_drain`` set, the table ops wait, in
        order, for `drain_tables`."""
        panel_id = patch.panel_id
        if size > TABLE_MAX_PATCH_BYTES:
            tables = [op for op in patch.ops if op.WhichOneof("op") in ("set_table", "patch_table")]
            for op in tables:
                self.refuse(
                    "panel_patch_refused",
                    f"panel {panel_id!r} op {op.WhichOneof('op')!r} on widget {op.widget_id!r}: "
                    f"INVALID_ARGUMENT: the PanelPatch carrying this table op is {size} bytes; "
                    f"one carrying a table op is at most {TABLE_MAX_PATCH_BYTES}",
                )
            if tables:
                kept = [op for op in patch.ops if op.WhichOneof("op") not in ("set_table", "patch_table")]
                if not kept:
                    return
                patch = pb2.PanelPatch(panel_id=panel_id, ops=kept)
        patch = self._hold_in_transport(patch)
        if patch is not None:
            self._deliver_panel_patch(patch)

    def _is_table_op(self, panel_id: str, op: pb2.PatchOp) -> bool:
        which = op.WhichOneof("op")
        if which in ("set_table", "patch_table"):
            return True
        found = self._panel_widgets.get(panel_id, {}).get(op.widget_id)
        return which == "set_value" and found is not None and found[0] == "table"

    def _hold_in_transport(self, patch: pb2.PanelPatch) -> "pb2.PanelPatch | None":
        """§8.4: the transport holds every op of a table in order -- its
        selection included -- none superseding or displacing another, and
        NOT counted against the 256 held-whole ops: bounded per widget by
        `TABLE_HELD_BYTES` of encoded ops instead.  The op that would pass
        it faults the session, naming the widget and the op.  Holds only
        while ``hold_table_drain`` is set or ops are still waiting, so none
        overtakes another.  Returns the patch's other ops, which go on as
        usual (``None`` for none)."""
        panel_id = patch.panel_id
        tables = [op for op in patch.ops if self._is_table_op(panel_id, op)]
        if not tables:
            return patch
        fault = None
        with self._lock:
            if not (self.hold_table_drain or self.table_transport):
                return patch
            for op in tables:
                key, size = (panel_id, op.widget_id), op.ByteSize()
                held = self._table_transport_bytes.get(key, 0)
                if held + size > TABLE_HELD_BYTES:
                    fault = (
                        f"table widget {op.widget_id!r} of panel {panel_id!r} holds {TABLE_HELD_BYTES >> 20} MiB of ops "
                        f"between two drains, the next a {op.WhichOneof('op')}: past what the host holds for one table "
                        "(control_plane_protocol.md §8.4)"
                    )
                    break
                kept = pb2.PatchOp()
                kept.CopyFrom(op)
                self.table_transport.append((panel_id, kept))
                self._table_transport_bytes[key] = held + size
        if fault is not None:
            self._fault_session(fault)
            return None
        rest = [op for op in patch.ops if not self._is_table_op(panel_id, op)]
        return pb2.PanelPatch(panel_id=panel_id, ops=rest) if rest else None

    def drain_tables(self) -> int:
        """The transport's drain (§8.4): every table op ``hold_table_drain``
        held, applied in the order it arrived -- each applied, or refused,
        alone, open panel or closed.  Returns how many were applied or
        refused.  Clear ``hold_table_drain`` first to stop holding new ones;
        any that arrive while this runs wait behind the rest and are drained
        too, so none overtakes an op still held."""
        count = 0
        while True:
            with self._lock:
                if not self.table_transport:
                    self._table_transport_bytes.clear()
                    return count
                panel_id, op = self.table_transport[0]
            self._deliver_panel_patch(pb2.PanelPatch(panel_id=panel_id, ops=[op]))
            with self._lock:
                self.table_transport.pop(0)
                key = (panel_id, op.widget_id)
                self._table_transport_bytes[key] = self._table_transport_bytes.get(key, 0) - op.ByteSize()
            count += 1

    def _deliver_panel_patch(self, patch: pb2.PanelPatch) -> None:
        panel_id = patch.panel_id
        # As the host does, BEFORE anything else: a patch's image
        # ops naming a retiring segment are dropped quietly and their loans
        # returned; the rest of the patch is still the plugin's statement.
        retiring = [op for op in patch.ops if op.WhichOneof("op") == "set_image" and op.set_image.segment_id in self.retiring_segments]
        if retiring:
            for op in retiring:
                self._drop_if_retiring(f"panel {panel_id} {op.widget_id}", [pb2.SegmentRef(segment_id=op.set_image.segment_id, offset=op.set_image.offset, len=op.set_image.len)])
            kept = [op for op in patch.ops if op not in retiring]
            if not kept:
                return
            patch = pb2.PanelPatch(panel_id=panel_id, ops=kept)
        widgets = self._panel_widgets.get(panel_id)
        if widgets is None:
            for op in patch.ops:
                if op.WhichOneof("op") == "set_image":
                    self._dispose_image_ref(op.set_image)
            return self.refuse("panel_patch_refused", f"panel {panel_id!r}: the panel is not declared")
        if panel_id not in self.opened_panels:
            return self._hold_patch(panel_id, widgets, patch)
        self._apply_panel_ops(panel_id, widgets, patch.ops)

    def _hold_patch(self, panel_id: str, widgets: dict, patch: pb2.PanelPatch) -> None:
        """As the host holds a closed panel's patch:
        a declared panel that is not open keeps the latest op per (widget,
        property) -- per row too, for a list's cells -- and applies them when
        it opens.  An undeclared widget is refused; an image is refused, in
        vizi's words, because the panel's `P` segment is granted at open."""
        with self._lock:
            held = self._held_patches.setdefault(panel_id, {})
        for op in patch.ops:
            which = op.WhichOneof("op")
            list_id = widgets[op.widget_id][1].get("_list") if op.widget_id in widgets else None
            rows_op = held.get((list_id, "set_rows", "")) if list_id is not None else None
            if op.widget_id not in widgets:
                why = f"widget {op.widget_id!r} is not in this panel"
            elif which == "set_image":
                self._dispose_image_ref(op.set_image)
                why = "the panel is not open, and its image segment is granted when it opens"
            elif which in ("set_table", "patch_table") or (which == "set_value" and widgets[op.widget_id][0] == "table"):
                # A table's ops are judged on arrival against the table the
                # panel will open with, and a bad one is refused alone.
                why = self._hold_table_op(panel_id, widgets, held, op)
                if why is None:
                    continue
            elif which == "set_rows" and len(op.set_rows.keys) > int(widgets[op.widget_id][1].get("max_rows", 256)):
                why = f"list {op.widget_id!r} holds at most {widgets[op.widget_id][1].get('max_rows', 256)} rows, not {len(op.set_rows.keys)}"
            elif op.row_key and rows_op is not None and op.row_key not in rows_op.set_rows.keys:
                # Held per row only for rows the
                # list will have, so the store is bounded by max_rows.
                why = f"list {list_id!r} has no row {op.row_key!r}"
            elif (
                op.row_key
                and rows_op is None
                and list_id is not None
                and op.row_key not in (rows := {k[2] for k in held if k[2] and widgets.get(k[0], (None, {}))[1].get("_list") == list_id})
                and len(rows) >= int(widgets[list_id][1].get("max_rows", 256))
            ):
                why = f"list {list_id!r} holds at most {widgets[list_id][1].get('max_rows', 256)} rows"
            else:
                if which == "set_rows":
                    keep = set(op.set_rows.keys)
                    with self._lock:
                        for key in [k for k in held if k[2] and k[2] not in keep and widgets.get(k[0], (None, {}))[1].get("_list") == op.widget_id]:
                            del held[key]
                key = (op.widget_id, which, op.row_key)
                kept = pb2.PatchOp()
                kept.CopyFrom(op)
                with self._lock:
                    held.pop(key, None)  # re-inserted last: the newer op is the later one
                    held[key] = kept
                continue
            self.refuse("panel_patch_refused", f"panel {panel_id!r} op {which!r} on widget {op.widget_id!r}: {why}")

    def _hold_table_op(self, panel_id: str, widgets: dict, held: dict, op: pb2.PatchOp) -> "str | None":
        """A table's op for a panel that has never opened, as the host holds
        it: the held table starts EMPTY, and each op is applied to it on
        arrival by the reference rules (`tables.apply_op`, `tables.select`),
        so a bad op is refused alone, at once, and is not held: the reason is
        returned.  The hold keeps ONE `set_table` -- the held table's whole
        rows -- and, once a selection has been sent, the selection the
        reference leaves, both applied at the open (`_mark_panel_open`), the
        rows first.  The held `set_table` was never a plugin's message, so
        no message bound applies to it: a table sent as a statement and its
        appends arrives whole."""
        table_id = op.widget_id
        kind, spec = widgets[table_id]
        which = op.WhichOneof("op")
        why = self._patch_op_refusal(panel_id, widgets, op, which)
        if why is not None:
            return why
        rows_key, value_key = (table_id, "table", ""), (table_id, "set_value", "")
        with self._lock:
            pending, chosen = held.get(rows_key), held.get(value_key)
            state = _tables.TableState()
            if pending is not None:
                state.keys = [row.key for row in pending.set_table.rows]
                state.rows = {row.key: row for row in pending.set_table.rows}
            if chosen is not None:
                state.selection = list(chosen.set_value.strings.values)
            if which == "set_value":
                why = _tables.select(state, op.set_value, bool(spec.get("multiple")), table_id)
            else:
                why = _tables.apply_op(state, op, len(spec.get("columns", [])), table_id)
            if why is not None:
                return why
            if which != "set_value":
                rows = pb2.PatchOp(widget_id=table_id)
                rows.set_table.rows.extend(state.rows[k] for k in state.keys)
                held.pop(rows_key, None)
                held[rows_key] = rows
            if which == "set_value" or chosen is not None:
                if which == "set_value":
                    held.pop(value_key, None)
                held[value_key] = pb2.PatchOp(widget_id=table_id, set_value=pb2.Value(strings=pb2.StringList(values=state.selection)))
        return None

    def _apply_panel_ops(self, panel_id: str, widgets: dict, ops) -> None:
        for op in ops:
            which = op.WhichOneof("op")
            why = self._patch_op_refusal(panel_id, widgets, op, which)
            if which == "set_image" and self._drop_if_retiring(f"panel {panel_id} {op.widget_id}", [pb2.SegmentRef(segment_id=op.set_image.segment_id, offset=op.set_image.offset, len=op.set_image.len)]):
                continue
            if which == "set_image":
                loan = None
                if why is None:
                    try:
                        loan = self._take_loan(pb2.SegmentRef(segment_id=op.set_image.segment_id, offset=op.set_image.offset, len=op.set_image.len))
                    except ValueError as exc:
                        why = f"the image ref could not be read: {exc}"
                    else:
                        if not loan.payload.startswith(b"\x89PNG\r\n\x1a\n"):
                            why = "the image is not a PNG"
                if why is None:
                    self.images.append((panel_id, op.widget_id, loan))
                    if op.row_key:
                        # An image cell, per row.
                        self.row_images[(panel_id, op.widget_id, op.row_key)] = loan
                    else:
                        self.patches.append((panel_id, op.widget_id, loan))
                if loan is not None:
                    self.release_loan(loan)
                else:
                    self._dispose_image_ref(op.set_image)
            if why is None and which in ("set_table", "patch_table"):
                why = self._apply_table_op(panel_id, op.widget_id, widgets[op.widget_id][1], op)
            elif why is None and which == "set_value" and widgets[op.widget_id][0] == "table":
                why = self._select_table(panel_id, op.widget_id, widgets[op.widget_id][1], op.set_value)
            if why is not None:
                self.refuse("panel_patch_refused", f"panel {panel_id!r} op {which!r} on widget {op.widget_id!r}: {why}")
                continue
            if which == "set_image":
                continue
            if which in ("set_table", "patch_table"):
                self.patches.append((panel_id, op.widget_id, op))
                continue
            if which == "set_value" and widgets[op.widget_id][0] == "table":
                # Never fires the command (§8.4's no-echo rule).
                with self._lock:
                    keys = list(self.tables[(panel_id, op.widget_id)].selection)
                self.patches.append((panel_id, op.widget_id, keys))
                continue
            if which == "set_progress":
                # A state, not a measurement with a sentinel.
                # `indeterminate` outranks `value`, and the host clamps rather
                # than refusing, so a plugin that reports 1.2 shows full.
                # A NaN value shows EMPTY (IDL and schema), and
                # the value is ignored while indeterminate.
                p = op.set_progress
                value = float(p.value)
                shown = None if p.indeterminate else (0.0 if value != value else min(1.0, max(0.0, value)))
                self.progress[(panel_id, op.widget_id)] = shown
                self.progress_log.append((panel_id, op.widget_id, shown))
                self.patches.append((panel_id, op.widget_id, op))
            elif which == "set_options":
                self._dropdown_options[(panel_id, op.widget_id)] = [str(o) for o in op.set_options.options]
                self.patches.append((panel_id, op.widget_id, op))
            elif which == "set_rows":
                keys = list(op.set_rows.keys)
                self.list_rows[(panel_id, op.widget_id)] = keys
                sel = (panel_id, op.widget_id)
                if self.list_selection.get(sel) not in (None, "", *keys):
                    self.list_selection[sel] = ""  # the selected row left
                self.patches.append((panel_id, op.widget_id, keys))
            elif op.row_key:
                self.row_patches.append((panel_id, op.widget_id, op.row_key, _plain(op.set_value) if which == "set_value" else op))
            else:
                if which == "set_value" and widgets[op.widget_id][0] == "list":
                    self.list_selection[(panel_id, op.widget_id)] = _plain(op.set_value)
                if which == "set_value" and widgets[op.widget_id][0] == "slider":
                    # The host moves the handle where it can stop and shows the
                    # value there, never the position (§3.3.1).
                    slider_spec = widgets[op.widget_id][1]
                    self.last_readout = slider_readout(slider_spec, slider_value(slider_spec, float(_plain(op.set_value))))
                self.patches.append((panel_id, op.widget_id, _plain(op.set_value) if which == "set_value" else op))

    def _patch_op_refusal(self, panel_id: str, widgets: dict, op: pb2.PatchOp, which: str | None) -> str | None:
        """Why the host would not apply ``op``, or ``None`` (see
        `_on_panel_patch`)."""
        found = widgets.get(op.widget_id)
        if found is None:
            return f"widget {op.widget_id!r} is not in this panel"
        kind, _spec = found
        # §8.4: the row rules come before the kind's own.
        list_id = _spec.get("_list")
        if list_id is not None:
            if not op.row_key:
                return f"widget {op.widget_id!r} is a cell of list {list_id!r}; a cell op needs a row_key"
            if op.row_key not in self.list_rows.get((panel_id, list_id), []):
                return f"list {list_id!r} holds no row {op.row_key!r}"
        elif op.row_key:
            return f"row_key {op.row_key!r} on widget {op.widget_id!r}, which is not a list's cell"
        if which == "set_rows":
            if kind != "list":
                return f"set_rows is not supported by widget {op.widget_id!r} (kind {kind!r})"
            keys = list(op.set_rows.keys)
            most = int(_spec.get("max_rows", 256))
            if len(keys) > most:
                return f"{len(keys)} rows is more than list {op.widget_id!r}'s max_rows {most}; refused whole, never truncated"
            if any(not k for k in keys):
                return "a row key is empty"
            if len(set(keys)) != len(keys):
                return "the row keys are not distinct"
            if any(len(k) > 128 for k in keys):
                return "a row key is longer than 128 characters"
            return None
        if kind == "list":
            if which in ("set_enabled", "set_visible"):
                return None
            if which == "set_value" and _spec.get("selectable"):
                v = _plain(op.set_value)
                ok = op.set_value.WhichOneof("kind") == "s" and (v == "" or v in self.list_rows.get((panel_id, op.widget_id), []))
                return None if ok else f"list {op.widget_id!r} selects a row key it holds, or \"\""
            return f"{which} is not supported by widget {op.widget_id!r} (kind 'list')"
        if which in ("set_table", "patch_table"):
            # The table's own checks go with applying it (`_apply_table_op`).
            return None if kind == "table" else f"{which} is not supported by widget {op.widget_id!r} (kind {kind!r})"
        if kind == "table" and which == "set_value":
            # As the host checks it: a value of no kind this host knows first,
            # then whether the table is selectable; the selection's own rule
            # goes with applying it (`tables.select`).
            if op.set_value.WhichOneof("kind") is None:
                return f"UNSUPPORTED: the value for widget {op.widget_id!r} is of a kind this host does not know; nothing was applied"
            if not _spec.get("selectable"):
                return f"table {op.widget_id!r} is not selectable, so takes no set_value"
            return None
        if which in ("set_enabled", "set_visible"):
            return None
        unsupported = f"{which} is not supported by widget {op.widget_id!r} (kind {kind!r})"
        if which == "set_progress":
            return None if kind == "progress" else unsupported
        if which == "set_options":
            return None if kind == "dropdown" else unsupported
        if which == "set_image":
            if kind != "image":
                return unsupported
            p_segments = [s for s in self.segments.values() if s.purpose == "P" and s.subject == panel_id]
            if op.set_image.segment_id not in {s.segment_id for s in p_segments}:
                return f"the image ref names segment {op.set_image.segment_id}; it is not this panel's P segment"
            return None
        if which == "set_value":
            value = op.set_value.WhichOneof("kind")
            v = _plain(op.set_value)
            if kind == "slider":
                return None if value in ("i", "f", "b") else unsupported
            if kind == "dropdown":
                options = self._dropdown_options.get((panel_id, op.widget_id), [])
                if value == "s":
                    return None if v in options else f"{v!r} is not one of dropdown {op.widget_id!r}'s options"
                if value in ("i", "b"):
                    return None if 0 <= int(v) < len(options) else f"index {v} is outside dropdown {op.widget_id!r}'s {len(options)} options"
                return unsupported
            # The host pickers (manifest_schema.md §3.3.1).
            # Checked against the emulator's own choices, as the host checks
            # against its.
            channels = self.dataset.shape_tczyx[1]
            if kind == "channel_picker":
                if _spec.get("multiple"):
                    if value != "ints":
                        return f"channel_picker {op.widget_id!r} is multiple: its value is a list of channels"
                    bad = [c for c in v if not 0 <= int(c) < channels]
                    return None if not bad else f"channels {bad} are not among the dataset's {channels}"
                if value != "i":
                    return unsupported
                return None if 0 <= int(v) < channels else f"channel {v} is not among the dataset's {channels}"
            if kind == "color_picker":
                # The schema's #RRGGBB; vizi's swatch (QColor) also takes
                # other spellings, which the emulator refuses.
                return None if value == "s" and _is_rgb_hex(v) else f"color_picker {op.widget_id!r} takes a #RRGGBB color"
            if kind == "colormap_picker":
                if value != "s":
                    return unsupported
                return None if v in self.colormaps else f"{v!r} is not a colormap this host offers"
            if kind == "roi_picker":
                if value != "s":
                    return unsupported
                with self._lock:
                    known = v in self.rois
                return None if known else f"{v!r} is not a region this host has"
            return None if kind in ("checkbox", "line_edit", "label", "button") else unsupported
        return f"op {which!r} is not one this host applies"

    # -- the `table` widget (§8.4, manifest_schema.md §3.3.1) --

    def _apply_table_op(self, panel_id: str, table_id: str, spec: dict, op: pb2.PatchOp) -> str | None:
        """A `set_table` or `patch_table`, applied whole by `tables.apply_op`
        or refused whole with its reason.  The rows still there keep their
        selection, and the plugin is not told.  A cell of a kind the host
        does not show is blank, never a refusal."""
        width = len(spec.get("columns", []))
        key = (panel_id, table_id)
        with self._lock:
            state = self.tables.setdefault(key, _tables.TableState())
            why = _tables.apply_op(state, op, width, table_id)
            if why is not None:
                return why
            cells = self._table_cells.setdefault(key, {})
            if op.WhichOneof("op") == "set_table":
                cells.clear()
                rows = op.set_table.rows
            else:
                for gone in op.patch_table.delete:
                    cells.pop(gone, None)
                rows = op.patch_table.upsert
            for row in rows:
                shown = [_table_cell(v, self.table_blank_kinds) for v in row.cells]
                cells[row.key] = tuple(shown + [None] * (width - len(shown)))
        return None

    def _select_table(self, panel_id: str, table_id: str, spec: dict, value: pb2.Value) -> str | None:
        """A selectable table's `set_value`, applied by `tables.select` -- one
        key (``Value.s``, ``""`` for none) or ``Value.strings``, duplicates
        collapsed -- or refused with its reason, the selection unchanged."""
        with self._lock:
            state = self.tables.setdefault((panel_id, table_id), _tables.TableState())
            return _tables.select(state, value, bool(spec.get("multiple")), table_id)

    @property
    def table_selection(self) -> dict[tuple[str, str], list[str]]:
        """Each table's selected keys, by (panel, table)."""
        with self._lock:
            return {key: list(state.selection) for key, state in self.tables.items()}

    def _table_spec(self, panel_id: str, table_id: str) -> dict:
        found = self._panel_widgets.get(panel_id, {}).get(table_id)
        if found is None or found[0] != "table":
            raise ValueError(f"panel {panel_id!r} has no table {table_id!r}")
        return found[1]

    def table_rows(self, panel_id: str, table_id: str) -> list[tuple[str, tuple]]:
        """The table's rows in the plugin's order, as ``(key, cells)`` with
        ``None`` for a blank cell -- what the host holds, not what is sorted."""
        self._table_spec(panel_id, table_id)
        key = (panel_id, table_id)
        with self._lock:
            state, cells = self.tables.get(key), self._table_cells.get(key, {})
            return [] if state is None else [(k, cells[k]) for k in state.keys]

    def table_shown(self, panel_id: str, table_id: str) -> list[tuple[str, tuple[str, ...]]]:
        """Each row's cells as the host draws them: a number in a number
        column to its ``precision``, a true bool in a bool column as ``✓`` and
        a false one empty, a blank cell empty, and any other cell -- one whose
        kind is not its column's type included -- as its text."""
        columns = self._table_spec(panel_id, table_id).get("columns", [])
        return [(k, tuple(_table_shown(c, v) for c, v in zip(columns, cells))) for k, cells in self.table_rows(panel_id, table_id)]

    def table_sorted(self, panel_id: str, table_id: str, column: str, descending: bool = False) -> list[str]:
        """The keys in the order the person sees after clicking ``column``'s
        header: a number column numerically, a bool column false before true,
        a string column by its text without case.  ``descending`` is a second
        click.  A blank cell -- and a NaN, and a cell whose kind is not its
        column's type, which sort the same way -- goes after every value in
        BOTH directions: where one side is blank the comparison answers by the
        order's own direction, and two blanks are equal.  Stable, as Qt's
        sort is, so equal cells (blanks and NaNs among them) keep the
        plugin's order.  Only a `sortable` table sorts; the plugin's order
        underneath is unchanged."""
        spec = self._table_spec(panel_id, table_id)
        if not spec.get("sortable"):
            raise ValueError(f"table {table_id!r} is not sortable")
        columns = spec.get("columns", [])
        index = next((i for i, c in enumerate(columns) if c.get("id") == column), None)
        if index is None:
            raise ValueError(f"table {table_id!r} has no column {column!r}")
        rows = self.table_rows(panel_id, table_id)
        keyed = [(_table_sort_key(columns[index], cells[index]), k) for k, cells in rows]

        def less(a, b) -> bool:
            # The host's comparison, blanks last both ways.
            a_blank, b_blank = a[0] != 0, b[0] != 0
            if a_blank or b_blank:
                if a_blank and b_blank:
                    return False
                return a_blank if descending else b_blank
            return a < b

        def compare(x, y) -> int:
            # Qt sorts ascending by lessThan(x, y) and descending by
            # lessThan(y, x), stably.
            first, second = (y[0], x[0]) if descending else (x[0], y[0])
            return -1 if less(first, second) else (1 if less(second, first) else 0)

        return [k for _, k in sorted(keyed, key=functools.cmp_to_key(compare))]

    def select_table_rows(self, panel_id: str, table_id: str, keys, timeout: float = COMMAND_WINDOW_S) -> pb2.CommandDone:
        """The person selects rows of a `selectable` table: the selection
        becomes ``keys`` and the table's command fires with ``{"rows":
        [key, ...]}`` (`Value.strings`), a list even of one or of none (a
        cleared selection), in the table's order -- through `invoke_widget`,
        so a `moves_view` table carries a fresh cause.  A person cannot select
        what the host would not offer, so more than one key on a table
        without `multiple`, or a key it does not hold, raises."""
        spec = self._table_spec(panel_id, table_id)
        if not spec.get("selectable"):
            raise ValueError(f"table {table_id!r} is not selectable")
        keys = [str(k) for k in keys]
        if len(keys) > 1 and not spec.get("multiple"):
            raise ValueError(f"table {table_id!r} selects one row, not {len(keys)}")
        held = [k for k, _ in self.table_rows(panel_id, table_id)]
        missing = [k for k in keys if k not in held]
        if missing:
            raise ValueError(f"table {table_id!r} has no row {missing[0]!r}")
        chosen = set(keys)
        ordered = [k for k in held if k in chosen]
        with self._lock:
            state = self.tables[(panel_id, table_id)]
            state.selection = list(ordered)
        return self.invoke_widget(panel_id, table_id, timeout, rows=pb2.Value(strings=pb2.StringList(values=ordered)))

    def _dispose_image_ref(self, ref) -> None:
        """Release the loan an image ref names, if it names one this host can
        find -- a refused op is still a disposition (§5.6)."""
        try:
            self.release_loan(self._take_loan(pb2.SegmentRef(segment_id=ref.segment_id, offset=ref.offset, len=ref.len)))
        except ValueError:
            pass

    def _on_activate_tool(self, request_id: str, request: pb2.ActivateTool) -> None:
        """§8.10: a plugin asking, answered as the `1.0` host answers it
.

        Refused `INVALID_ARGUMENT` for a tool the manifest does not declare,
        and for any change while a gesture is open (``activation_refusals``).
        Otherwise, and in this order -- the arrangement FIRST, then the reply:

        - **activate** a tool that is not engaged: the engaged one, if any,
          is DISPLACED -- its binding retired -- and this one is bound under a
          NEW handle from the session counter; then `ToolActivationChanged
          {active: true}`.  Already engaged: the reply alone.
        - **deactivate** the engaged tool: its binding is retired, then
          `{active: false}`.  A tool that is not engaged: the reply alone.

        Nothing is awaited on this thread -- the acks arrive on it.
        """
        with self._lock:
            self.activations.append(request)
        refusal = None
        if request.tool_id not in self.tool_ids:
            refusal = f"plugin {self.plugin_id!r} does not declare tool {request.tool_id!r}"
        elif self._open_interactions:
            refusal = "tool activation cannot change during an active gesture"
        if refusal is not None:
            with self._lock:
                self.activation_refusals.append((request.tool_id, pb2.Error.INVALID_ARGUMENT, refusal))
            return self._fail(request_id, pb2.Error.INVALID_ARGUMENT, refusal)
        engaged = self.engaged == request.tool_id
        if request.deactivate:
            if engaged:
                self._send_retirement(request.tool_id)
        elif not engaged:
            self._engage(request.tool_id, **self._manifest_binding(request.tool_id))
        self.send(pb2.HostMsg(id=request_id, tool_activation_changed=pb2.ToolActivationChanged(
            tool_id=request.tool_id, active=not request.deactivate)))

    def _manifest_binding(self, tool_id: str) -> dict:
        """What the host binds a tool with when the PLUGIN engages it: the
        manifest's `mode`, `cancel_on_camera_move` and every declared action
        with its key.  No triggers: which of those the
        host assigns depends on every other plugin's claims, and a trigger
        left out is §8.10's *"a refused binding is not a refused tool"* --
        always a legal binding.  A driver that wants triggers engages with
        `bind_tool` instead."""
        tool = next((t for t in self.manifest.get("contributions", {}).get("tools", []) if t.get("id") == tool_id), {})
        return {
            "mode": pb2.TOOL_MODE_FIRE if tool.get("mode") == "fire" else pb2.TOOL_MODE_GESTURE,
            "cancel_on_camera_move": bool(tool.get("cancel_on_camera_move", True)),
            "actions": [(a["id"], str(a.get("key", ""))) for a in tool.get("actions", [])],
        }

    def _v_for(self, length: int) -> Segment:
        """The `V` ring a deposit of ``length`` goes into.

        **A driver that set `volume_segment` gets that ring and nothing else**
        -- one fixed ring, pre-granted, which is how the ride-along scenarios
        run.

        **A driver that left it unset gets the `1.0` host's policy**: no `V` until the first read,
        then a ring sized to exactly that deposit; when a deposit does not
        fit, a new ring of ``max(deposit, 2 x capacity)`` at a NEW id, naming
        the old in `InstallSegment.supersedes` (`shared_memory_protocol.md`
        §9.5).  `M` goes at the grant, the installation rides the first reply
        into the ring (§8.12), and no acknowledgement is waited for.  A
        second reply into a ring whose install has already ridden goes BARE
        for the plugin to hold (§8.12's receiver rule), and nothing is ever
        written into a ring after its replacement is granted (§9.1), as the
        host does.  The superseded ring is closed here when the plugin's
        `SegmentReleased` for it arrives.

        **`supersedes` names only a drained ring**
        (`shared_memory_protocol.md` §9.5).  When the ring being outgrown still
        has deposits LENT -- replies the plugin may not have copied yet -- the
        new ring names nothing in `supersedes`, and the old one is sent
        `RetireSegment` once its last deposit comes back.  Naming it would let
        the plugin retire it under an uncopied deposit and fail the read;
        `v_supersede_undrained` does that on purpose, to reproduce the failure.
        """
        current = self.segments.get(self.volume_segment) if self.volume_segment is not None else None
        if self.volume_segment is not None and self.volume_segment not in self.v_rings:
            return current
        if current is not None and current.ring.fits(length):
            return current
        need = (length + HostRing.ALIGN - 1) // HostRing.ALIGN * HostRing.ALIGN
        capacity = need if current is None else max(need, current.ring.extent * 2)
        seg = Segment.host_ring(self.new_segment_id(), "V", 32 + capacity, self.directory)
        if current is not None and (self.v_supersede_undrained or not current.ring.outstanding):
            seg.supersedes = current.segment_id
        elif current is not None:
            self._v_draining.add(current.segment_id)
        self.add(seg)
        self.v_rings.append(seg.segment_id)
        self.ride_along.add(seg.segment_id)
        self.volume_segment = seg.segment_id
        self.grant_capability(seg.segment_id)
        return seg

    def _deposit_v(self, request_id: str, payload: bytes) -> tuple[Segment, int] | None:
        """Deposit into `V` (`_v_for`), or refuse the request when the ring
        is full.

        A fixed ring never grows, so a plugin holding every deposit in it
        gets a refusal it can report."""
        with self._read_lock:
            return self._deposit_v_locked(request_id, payload)

    def _deposit_v_locked(self, request_id: str, payload: bytes) -> tuple[Segment, int] | None:
        seg = self._v_for(len(payload))
        if seg.segment_id in self.v_rings and seg.segment_id not in self.ride_along and seg.segment_id not in self.acks:
            # Its installation already went, with an earlier reply, and the
            # plugin has not acknowledged it: this reply goes bare (§8.12).
            self.bare_replies += 1
        try:
            return seg, seg.ring.deposit(payload)
        except RingFull as exc:
            self.send(pb2.HostMsg(id=request_id, host_error=pb2.Error(kind=pb2.Error.INTERNAL, message=str(exc))))
            return None

    def _install_ahead(self, seg: Segment) -> None:
        """§8.12: a reply with no `install` field to ride
        (`MaskVoxels`, `RoiInclusion`) into a ring whose installation has not
        gone yet is PRECEDED by the standalone `InstallSegment`, and the
        receiver holds the descriptor until both halves are in."""
        if seg.segment_id in self.ride_along:
            self.ride_along.discard(seg.segment_id)
            self.send(pb2.HostMsg(install_segment=seg.install()))

    def hold_reads(self) -> None:
        """Park every `ReadDataVolume` from now until `release_reads` (see
        `_reads_held`): recorded on arrival, answered on release."""
        with self._lock:
            self._reads_held = True

    def await_reads(self, count: int = 1, timeout: float = 10.0) -> int:
        """Block until at least ``count`` reads are parked; returns how many.
        After this the plugin's operation is PROVABLY in flight -- waiting on
        a read the host has not answered -- which is the moment a busy-guard
        scenario fires its second request."""
        return self._await(lambda: len(self._held_reads) if len(self._held_reads) >= count else None, timeout, f"{count} parked read(s)")

    def release_reads(self) -> int:
        """Stop holding and answer every parked read, in arrival order, from
        the calling thread.  Returns how many were answered."""
        with self._lock:
            self._reads_held = False
            held, self._held_reads = self._held_reads, []
        for request_id, request in held:
            self._serve_read(request_id, request)
        return len(held)

    def _answer_read(self, request_id: str, request: pb2.ReadDataVolume) -> None:
        self.reads.append((request.t, request.c, request.level))
        self.read_requests.append(
            ReadRequest(
                t=request.t,
                c=request.c,
                level=request.level,
                roi=(
                    (tuple(request.roi_min_zyx), tuple(request.roi_max_zyx))
                    if request.roi_min_zyx or request.roi_max_zyx
                    else None
                ),
                raw=request.raw,
            )
        )
        with self._lock:
            if self._reads_held:
                self._held_reads.append((request_id, request))
                return
        self._serve_read(request_id, request)

    def _serve_read(self, request_id: str, request: pb2.ReadDataVolume) -> None:
        # One read served at a time: parked reads are answered from a
        # driver's thread while new ones arrive on the reader thread, and
        # both deposit into the same `V`.
        with self._read_lock:
            self._serve_read_locked(request_id, request)

    def _serve_read_locked(self, request_id: str, request: pb2.ReadDataVolume) -> None:
        if self.read_delay:
            # A slow host, for scenarios that need a computation to still be
            # in flight when the next request lands; never a protocol matter.
            time.sleep(self.read_delay)
        if self.self_serve_guard:
            self.send(pb2.HostMsg(id=request_id, host_error=pb2.Error(kind=pb2.Error.UNSUPPORTED, message="the active dataset is served by this plugin's reader; reading it back through the host would deadlock")))
            return
        shape_t, shape_c = self.dataset.shape_tczyx[:2]
        if request.t >= shape_t or request.c >= shape_c:
            # The host's first check.
            self.send(pb2.HostMsg(id=request_id, host_error=pb2.Error(kind=pb2.Error.OUT_OF_BOUNDS, message=f"(t={request.t}, c={request.c}) exceeds dataset shape {tuple(self.dataset.shape_tczyx)}")))
            return
        if not 0 <= request.level < self.dataset.levels:
            self.send(pb2.HostMsg(id=request_id, host_error=pb2.Error(kind=pb2.Error.OUT_OF_BOUNDS, message=f"hostsim serves levels 0..{self.dataset.levels - 1} only")))
            return
        if (request.roi_min_zyx or request.roi_max_zyx) and (len(request.roi_min_zyx) != 3 or len(request.roi_max_zyx) != 3):
            # As the host checks it: a one-sided or short box is
            # `INVALID_ARGUMENT`.
            self._fail(request_id, pb2.Error.INVALID_ARGUMENT, "roi_min_zyx/roi_max_zyx must each be 3 values")
            return
        volume = self.dataset.volume_at(request.t, request.c, request.level)
        # §8.7: with a filter shared, a read is its output
        # and says so in `DataVolume.filter`; a `raw` read is the source and
        # carries no `filter`.
        filtered = self.dataset.applied_filter is not None and not request.raw
        if filtered and self.dataset.filter_fn is not None:
            volume = np.ascontiguousarray(self.dataset.filter_fn(volume, request.t, request.c), dtype=volume.dtype)
        if request.roi_min_zyx:
            # §8.7: the ROI is in the coordinates OF THE REQUESTED LEVEL, min
            # inclusive, max exclusive; a box past the level's shape is refused
            # rather than clipped, since the plugin declared it.
            lo, hi = list(request.roi_min_zyx), list(request.roi_max_zyx)
            if any(a < 0 or b > n or a >= b for a, b, n in zip(lo, hi, volume.shape)):
                self.send(pb2.HostMsg(id=request_id, host_error=pb2.Error(kind=pb2.Error.OUT_OF_BOUNDS, message=f"roi {lo}..{hi} is outside level {request.level}'s shape {list(volume.shape)}")))
                return
            volume = np.ascontiguousarray(volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
        payload = volume.tobytes()
        deposited = self._deposit_v(request_id, payload)
        if deposited is None:
            return
        seg, offset = deposited
        reply = pb2.DataVolume(
            segment=pb2.SegmentRef(segment_id=seg.segment_id, offset=offset, len=len(payload), shape_zyx=list(volume.shape), dtype=self.dataset.wire_dtype),
            t=request.t,
            c=request.c,
            level=request.level,
            open_generation=self.dataset.open_generation,
        )
        if filtered:
            reply.filter.CopyFrom(self.dataset.applied_filter)
        if seg.segment_id in self.ride_along:
            # §3: where a grant already has a gRPC exchange, the region table
            # rides it rather than being a new message -- and §5.5's exception
            # says the receiver applies the arrangement BEFORE interpreting the
            # rest of the message it arrived on.
            self.ride_along.discard(seg.segment_id)
            reply.install.CopyFrom(seg.install())
        self.send(pb2.HostMsg(id=request_id, data_volume=reply))

    # -- helpers for scenarios --

    def _kill_launch(self) -> None:
        """§9: a launch past its budget is killed (and the next request
        would be a fresh launch)."""
        if self._plugin is not None and self._plugin.poll() is None:
            self._plugin.kill()
            self._plugin.wait()

    def await_ready(self) -> pb2.Ready:
        """Wait for `Ready` within what is left of the launch budget (§9:
        one deadline over the bind and `Ready`); past it the
        plugin is killed and the launch fails, as vizi's does."""
        while self.ready is None:
            if self.handshake_refusal is not None:
                raise RuntimeError(f"vizi refuses the handshake: {self.handshake_refusal}")
            if self.stream_ended.is_set():
                code = self._plugin.poll() if self._plugin is not None else None
                raise RuntimeError(
                    "the session ended before Ready "
                    f"(process {'still running' if code is None else f'exited {code}'}; stream error: {self.stream_error!r})"
                )
            if self.launch_deadline is not None and time.monotonic() > self.launch_deadline:
                self._kill_launch()
                raise RuntimeError(f"did not become ready within {self.launch_budget:g}s (waiting for Ready)")
            time.sleep(0.01)
        return self.ready

    def check_handshake(self) -> str:
        """§3.1 as the real host checks it: the version (the major matches and
        the `Ready` minor does not exceed this host's `Hello` minor -- checked
        as the `Ready` arrives, see :func:`ready_refusal`), `Ready.plugin_id`
        (which "echoes the manifest identity"), and the identity string's
        FORM and nothing else -- never its content."""
        self.await_ready()
        got = (self.ready.protocol_major, self.ready.protocol_minor)
        # A `Ready` minor below this host's is the plugin clamping to an older
        # SDK and runs; one above it was refused on arrival.
        assert ready_refusal(self.ready, self.protocol) is None, f"handshake {got} against {self.protocol}"
        assert self.ready.plugin_id == self.plugin_id, self.ready.plugin_id
        ready_id = self.ready_id or ""
        assert ready_id.startswith(READY_PREFIX), f"Ready.id {ready_id[:80]!r} does not start with {READY_PREFIX!r}"
        assert len(ready_id) > len(READY_PREFIX), "Ready.id carries no version string"
        assert len(ready_id.encode("utf-8")) <= IDENTITY_MAX_BYTES, "Ready.id over 2048 bytes"
        assert not any(unicodedata.category(ch) == "Cc" for ch in ready_id), "Ready.id carries a control character"
        # And nothing else: §3.1 says the receiver "never matches the string
        # against anything, never parses the version string, and never
        # compares it to the manifest or to its own".  The version string is
        # logged here, as the host logs
        # it, and that is all a receiver does with it.
        return f"protocol {got[0]}.{got[1]} from {self.ready.plugin_id}; {ready_id[len(READY_PREFIX):]!r}"

    def status_of(self, panel: str, widget: str = "status") -> str:
        for panel_id, widget_id, value in reversed(self.patches):
            if panel_id == panel and widget_id == widget:
                return str(value)
        return ""

    def await_status(self, panel: str, needle: str, timeout: float = 5.0, widget: str = "status") -> str:
        """The label once it says ``needle``, or whatever it says at the
        deadline.  A handler that sets the label and then returns emits two
        messages on two queues, and §7 drains the reply's queue ahead of the
        coalescing one -- so the `CommandDone` can arrive first."""
        deadline = time.monotonic() + timeout
        while needle not in self.status_of(panel, widget) and time.monotonic() < deadline:
            time.sleep(0.02)
        return self.status_of(panel, widget)

    def await_progress(self, panel: str, widget: str, until=None, timeout: float = 5.0) -> float | None:
        """Wait for a `set_progress` and answer the newest value -- a fraction,
        or ``None`` for indeterminate.

        Wait rather than read: a panel patch is a COALESCING push, so the
        command's reply can overtake it and a run of values may arrive as one.
        The number of pushes is never assertable.

        **`until` is how you wait for a PARTICULAR state, and an SDK with a
        widget mirror needs it.**  Such a mirror re-pushes its whole state
        when the panel opens, as §6 asks, so the bar's declared opening value
        lands before the work starts and a bare wait answers *that* -- the
        first value present, not the last one the plugin sent.  Pass a
        predicate (``until=lambda v: v is None`` for indeterminate) to wait
        for the state you mean.  Without one this answers as soon as any
        value has arrived, which is only the final state when nothing pushed
        before the work began.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (panel, widget) in self.progress:
                value = self.progress[(panel, widget)]
                if until is None or until(value):
                    return value
            time.sleep(0.01)
        seen = self.progress.get((panel, widget), "nothing")
        raise AssertionError(
            f"no set_progress for {panel}/{widget} within {timeout}s"
            if until is None
            else f"{panel}/{widget} never reached the wanted state within {timeout}s; newest was {seen}"
        )

    def await_pushes(self, count: int, timeout: float = 5.0, kind: str | None = None) -> list[Push]:
        """The control-plane pushes once there are ``count`` of them."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pushes = [p for p in self.pushes if kind is None or p.kind == kind]
            if len(pushes) >= count:
                return pushes
            time.sleep(0.02)
        return [p for p in self.pushes if kind is None or p.kind == kind]

    def await_violations(self, site: str, count: int = 1, timeout: float = 5.0) -> list[pb2.ContractViolation]:
        """Wait until at least ``count`` `ContractViolation`s at ``site``
        (a `ViolationSite` spelling such as ``"unusable_context"``) have
        arrived, and return them."""
        where = pb2.ViolationSite.Value("VIOLATION_SITE_" + site.upper())

        def matching():
            found = [v for v in self.violations if v.where == where]
            return found if len(found) >= count else None

        return self._await(matching, timeout, f"no {count} ContractViolation(s) at {site}")

    def await_live(self, count: int, timeout: float = 5.0) -> list[LiveValue]:
        """Consumed live values once there are ``count`` of them -- from the
        plugin's hints, and from the host's own scan while waiting, which is
        §5's unconditional re-check and what keeps a dropped hint costless."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.live) >= count:
                    return list(self.live)
            self.scan()
            time.sleep(0.02)
        with self._lock:
            return list(self.live)


class _Gesture:
    """One open interaction on one tool and pane."""

    def __init__(self, host: Host, tool: int, view_id: int, interaction: int) -> None:
        self.host = host
        self.tool = tool
        self.view_id = view_id
        self.interaction = interaction
        self.seq = 0
        self.closed = False

    def _send(self, phase: int, position, **fields) -> None:
        if self.closed:
            raise RuntimeError("the interaction is closed")
        self.host._event(phase, self.tool, self.view_id, self, position=position, **fields)
        if phase in (PHASE_RELEASE, PHASE_CANCEL):
            self.closed = True
            self.host._open_interactions.pop(self.interaction, None)

    def drag(self, position, **fields) -> None:
        self._send(PHASE_DRAG, position, **fields)

    def release(self, position, **fields) -> None:
        self._send(PHASE_RELEASE, position, **fields)

    def cancel(self, **fields) -> None:
        """A `CANCEL` carrying `NONE`."""
        self._send(PHASE_CANCEL, None, **fields)


class _IdLedger:
    """Ids are never reused.  The ids of the plugin's messages that deserve a
    reply (its requests, and an unknown kind carrying an id) and, apart, of
    its replies to this host's requests (a `GrantState` answering a
    `GrantStateQuery` included).  A second message under a recorded id is
    ``"resent"`` when byte-identical, else ``"reused"``; anything else is
    ``"new"``.  The tables hold the last ``WINDOW`` ids received, the host's
    bound; an older id is not checked.  Called on the reader thread only."""

    WINDOW = 65_536

    def __init__(self) -> None:
        self.requests: dict[str, bytes] = {}
        self.replies: dict[str, bytes] = {}
        self._window: collections.deque = collections.deque()

    def observe(self, message, kind) -> str:
        if not message.id:
            return "new"
        if kind is None or kind in _REQUEST_KINDS:
            table = self.requests
        elif kind in _REPLY_KINDS or kind == "grant_state":
            table = self.replies
        else:
            return "new"
        digest = hashlib.sha256(message.SerializeToString(deterministic=True)).digest()
        seen = table.get(message.id)
        if seen is not None:
            return "resent" if seen == digest else "reused"
        table[message.id] = digest
        self._window.append((table, message.id))
        if len(self._window) > self.WINDOW:
            old_table, old = self._window.popleft()
            del old_table[old]
        return "new"


#: Plugin-message arms that answer a host request, routed by id (§4).
_REPLY_KINDS = frozenset({"reader_opened", "series_list", "segment", "metadata", "closed", "error", "command_done", "export_opened", "chunk_written", "export_done", "stage_planned"})
#: The reply kinds each host request takes, besides `error`: a pending
#: request's id belongs to its reply alone, so any other known reply kind under
#: it faults the session.  A request not listed takes any reply.
_HOST_REQUEST_REPLIES = {
    "open_reader": ("reader_opened", "series_list"),
    "read_volume": ("segment",),
    "read_plane": ("segment",),
    "metadata": ("metadata",),
    "close_reader": ("closed",),
    "invoke_command": ("command_done",),
    "open_export": ("export_opened",),
    "export_chunk": ("chunk_written",),
    "finish_export": ("export_done",),
    "abort_export": ("export_done",),
    "plan_stage": ("stage_planned",),
}
#: `roi.proto`'s `OutOfSpan.HIDE`: a region not present outside its domain.
#: `HOLD` (0, the wire default) clamps to the nearest end row.
_OUT_OF_SPAN_HIDE = 1
#: Plugin->host arms that are REQUESTS: a plugin thread is blocked on each one's reply, so each is
#: answered whatever happens, and each is PENDING in the ledger until it is.
_REQUEST_KINDS = frozenset({
    "get_view_state", "read_data_volume", "get_rois", "read_mask_voxels", "read_roi_inclusion",
    "contribute_rois", "extend_roi_poses", "compile_label_volume", "activate_tool", "request_asset",
    "request_status_query", "request_save_path", "get_colormap", "view_request",
})
#: Plugin requests gated on `data_access` and on `contribute_rois` (§8.7, §8.8).
_DATA_ACCESS = frozenset({"get_view_state", "read_data_volume", "get_rois", "read_mask_voxels", "request_asset", "read_roi_inclusion"})
_CONTRIBUTE = frozenset({"contribute_rois", "compile_label_volume", "extend_roi_poses"})


def _segment_refs(message):
    """Every `SegmentRef` / `ImageRef` below a message, as `SegmentRef`s --
    the loans a refused push still disposes of."""
    for descriptor, value in message.ListFields():
        if descriptor.type != descriptor.TYPE_MESSAGE:
            continue
        for sub in _message_children(descriptor, value):
            name = sub.DESCRIPTOR.name
            if name == "SegmentRef":
                yield sub
            elif name == "ImageRef":
                yield pb2.SegmentRef(segment_id=sub.segment_id, offset=sub.offset, len=sub.len)
            else:
                yield from _segment_refs(sub)


def _value(value) -> pb2.Value:
    if isinstance(value, pb2.Value):
        return value
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        # `Value.strings`, a table's selected row keys.
        return pb2.Value(strings=pb2.StringList(values=list(value)))
    if isinstance(value, (list, tuple)):
        # `Value.ints`, a multi-select channel_picker's value.
        return pb2.Value(ints=pb2.IntList(values=[int(v) for v in value]))
    if isinstance(value, bool):
        return pb2.Value(b=value)
    if isinstance(value, int):
        return pb2.Value(i=value)
    if isinstance(value, float):
        return pb2.Value(f=value)
    return pb2.Value(s=str(value))


def _plain(value: pb2.Value):
    kind = value.WhichOneof("kind")
    if kind == "ints":
        return list(value.ints.values)
    if kind == "strings":
        return list(value.strings.values)
    return getattr(value, kind) if kind else None


#: The kinds a table cell shows (§8.4); any other is a blank cell.
_TABLE_CELL_KINDS = frozenset({"s", "f", "i", "b"})


def _table_cell(value: pb2.Value, said: set):
    """A wire cell as the host reads it: ``None`` (blank)
    for no kind, or a kind other than ``s f i b`` -- a list, or one this host
    predates, which decodes as no kind -- recorded once in ``said``, the
    host's local diagnostic.  Never a refusal of its row."""
    kind = value.WhichOneof("kind")
    if kind is None:
        return None
    if kind not in _TABLE_CELL_KINDS:
        said.add(kind)
        return None
    return getattr(value, kind)


def _table_shown(column: dict, value) -> str:
    """A cell as drawn: a number in a number column to its ``precision``, a
    bool in a bool column as ``✓`` or empty, a blank empty, and any other
    cell -- one whose kind is not its column's type included -- as its
    text."""
    if value is None:
        return ""
    if column.get("type") == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return f"{float(value):.{int(column.get('precision', 3))}f}"
    if column.get("type") == "bool" and isinstance(value, bool):
        return "✓" if value else ""
    return str(value)


#: The cell kinds each column type holds (§8.4).  A cell of any other kind in
#: the column is shown as its text and sorts as a blank.
_TABLE_COLUMN_KINDS = {"string": (str,), "number": (int, float), "bool": (bool,)}


def _table_cell_matches(column: dict, value) -> bool:
    """The cell's kind is its column type's: ``s`` in a string column, ``i``
    or ``f`` in a number column, ``b`` in a bool column."""
    kinds = _TABLE_COLUMN_KINDS.get(column.get("type", "string"), (str,))
    if isinstance(value, bool):
        return bool in kinds
    return isinstance(value, kinds)


def _table_sort_key(column: dict, value):
    """How a header click orders a cell: blanks after every value; a number
    column's numbers by value (NaN after them, before blanks), a bool
    column's bools false first, a string column's strings by their
    case-folded text.  A cell whose kind is not its column's type sorts as a
    blank."""
    if value is None or not _table_cell_matches(column, value):
        return (2, 0.0, "")
    if column.get("type") == "number":
        number = float(value)
        return (1, 0.0, "") if math.isnan(number) else (0, number, "")
    if column.get("type") == "bool":
        return (0, float(value), "")
    return (0, 0.0, str(value).casefold())


def _is_rgb_hex(v) -> bool:
    return isinstance(v, str) and len(v) == 7 and v[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in v[1:])


def _targets_conflict(targets: dict) -> "str | None":
    """As the host judges targets: framing a region selects
    it, so a different `select_region` beside it could only be dropped."""
    framed, selected = targets.get("frame_region"), targets.get("select_region")
    if framed and selected and framed != selected:
        return "frame_region selects the region it frames; a different select_region cannot apply"
    return None


def _default_colormaps() -> dict[str, bytes]:
    """vizi's curated colormap names, 256
    RGBA entries each.  ``gray`` is exact; the others are the emulator's own
    ramps (their names are vizi's, their colors are not)."""
    ramp = np.arange(256, dtype=np.uint8)
    full = np.full(256, 255, dtype=np.uint8)
    zero = np.zeros(256, dtype=np.uint8)
    floor = ramp.copy()
    floor[0] = 0
    # vizi's eight, in its order.
    tables = {
        "gray": (ramp, ramp, ramp, full),
        "gray > min": (ramp, ramp, ramp, np.where(ramp == 0, 0, 255).astype(np.uint8)),
        "gray-stain": (255 - ramp // 3, 255 - ramp // 2, 255 - ramp // 4, full),
        "hot": (np.clip(ramp.astype(int) * 3, 0, 255).astype(np.uint8), np.clip(ramp.astype(int) * 3 - 255, 0, 255).astype(np.uint8), np.clip(ramp.astype(int) * 3 - 510, 0, 255).astype(np.uint8), full),
        "viridis": (ramp // 4, ramp, 255 - ramp // 2, full),
        "inferno": (ramp, ramp // 2, zero, full),
        "vizi-dark": (ramp // 2, ramp // 2, ramp, full),
        "vizi-light": (255 - ramp // 2, 255 - ramp // 2, 255 - ramp // 4, full),
    }
    return {name: np.stack(cols, axis=1).astype(np.uint8).tobytes() for name, cols in tables.items()}


def content_hash(record) -> str:
    """`RoiRecord.content_hash`: 16 hex characters over what
    decides the region's voxels -- here the `Roi` itself, so a rename or a
    restyle leaves it unchanged.  The emulator's digest, not vizi's: a plugin
    compares two hashes from one host, never computes one."""
    import hashlib

    return hashlib.sha256(record.roi.SerializeToString(deterministic=True)).hexdigest()[:16]
