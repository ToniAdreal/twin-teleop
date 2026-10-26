"""Replay utilities for recorded TwinTeleop telemetry streams.

A recorded stream is simply concatenated binary frames (see telemetry.py).
This module turns such a stream into:
  * an iterator of decoded frames (file or in-memory bytes)
  * per-joint time series for a digital-twin front end
  * a deterministic replay schedule at a fixed rate (the transport layer —
    e.g. a WebSocket server — sleeps on the offsets; replay itself does
    no sleeping, so it is fully testable and reproducible)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from telemetry import TelemetryFrame, decode, encode

_HEADER_SIZE = 12  # 2s magic + B version + B joint_count + Q timestamp


def frame_size(first_bytes: bytes) -> int:
    """Size in bytes of one frame, derived from its header joint count."""
    if len(first_bytes) < _HEADER_SIZE:
        raise ValueError("not enough bytes for a frame header")
    n = first_bytes[3]
    return _HEADER_SIZE + 8 * n + 2  # positions+torques (float32 x2) + crc16


def iter_frames_from_bytes(data: bytes) -> Iterator[TelemetryFrame]:
    offset = 0
    while offset < len(data):
        size = frame_size(data[offset : offset + _HEADER_SIZE])
        chunk = data[offset : offset + size]
        if len(chunk) != size:
            raise ValueError("truncated frame in stream")
        yield decode(chunk)
        offset += size


def read_stream(path: str | Path) -> list[TelemetryFrame]:
    return list(iter_frames_from_bytes(Path(path).read_bytes()))


def write_stream(path: str | Path, frames: Sequence[TelemetryFrame]) -> int:
    data = b"".join(encode(f) for f in frames)
    Path(path).write_bytes(data)
    return len(data)


def to_time_series(frames: Sequence[TelemetryFrame]) -> dict[int, dict[str, tuple]]:
    """Per-joint time series: {joint: {"t_s": (...), "position": (...), "torque": (...)}}"""
    if not frames:
        return {}
    n = len(frames[0].positions)
    t0 = frames[0].timestamp_us
    return {
        j: {
            "t_s": tuple((f.timestamp_us - t0) / 1_000_000 for f in frames),
            "position": tuple(f.positions[j] for f in frames),
            "torque": tuple(f.torques[j] for f in frames),
        }
        for j in range(n)
    }


def replay_schedule(
    frames: Sequence[TelemetryFrame], hz: float = 50.0
) -> Iterator[tuple[float, TelemetryFrame]]:
    """Yield (offset_seconds_from_start, frame) at a fixed replay rate.

    Deterministic: offsets are i / hz, independent of wall clock.
    A transport server does: `for offset, frame in replay_schedule(...):
    sleep_until(start+offset); ws.send(encode(frame))`.
    """
    if hz <= 0:
        raise ValueError("hz must be positive")
    for i, frame in enumerate(frames):
        yield i / hz, frame
