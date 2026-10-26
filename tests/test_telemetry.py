import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telemetry import TelemetryFrame, decode, encode  # noqa: E402


def sample() -> TelemetryFrame:
    return TelemetryFrame(
        timestamp_us=1_726_000_000_123_456,
        positions=(0.0, 0.5, -1.25, 2.0, 0.125, -0.75),
        torques=(1.5, -2.5, 0.25, 0.0, 3.75, -0.5),
    )


def test_roundtrip():
    frame = sample()
    out = decode(encode(frame))
    assert out.timestamp_us == frame.timestamp_us
    assert out.positions == pytest.approx(frame.positions)
    assert out.torques == pytest.approx(frame.torques)


def test_corrupt_frame_rejected():
    raw = bytearray(encode(sample()))
    raw[14] ^= 0xFF  # flip a payload byte; CRC must catch it
    with pytest.raises(ValueError):
        decode(bytes(raw))


def test_truncated_frame_rejected():
    with pytest.raises(ValueError):
        decode(encode(sample())[:-3])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        TelemetryFrame(timestamp_us=0, positions=(0.0,), torques=(0.0, 1.0))


def test_frame_size_scales_with_joints():
    f = TelemetryFrame(timestamp_us=1, positions=(0.0,) * 7, torques=(0.0,) * 7)
    assert len(encode(f)) == struct.calcsize("<2sBBQ") + 8 * 7 + 2
