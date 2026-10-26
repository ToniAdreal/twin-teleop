"""Binary telemetry frame codec for multi-axis manipulator teleoperation.

Frame layout (little-endian):
    magic        2 bytes   b"TT"
    version      uint8     currently 1
    joint_count  uint8     N (1..16)
    timestamp_us uint64    capture time, microseconds
    positions    N x float32  joint angles, rad
    torques      N x float32  joint torques, N*m
    crc16        uint16    CRC-16/CCITT-FALSE over all preceding bytes
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"TT"
VERSION = 1
MAX_JOINTS = 16
_HEADER = struct.Struct("<2sBBQ")
_CRC = struct.Struct("<H")


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class TelemetryFrame:
    timestamp_us: int
    positions: tuple[float, ...]
    torques: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (1 <= len(self.positions) <= MAX_JOINTS):
            raise ValueError("joint count out of range")
        if len(self.positions) != len(self.torques):
            raise ValueError("positions/torques length mismatch")


def encode(frame: TelemetryFrame) -> bytes:
    n = len(frame.positions)
    payload = _HEADER.pack(MAGIC, VERSION, n, frame.timestamp_us)
    payload += struct.pack(f"<{n}f", *frame.positions)
    payload += struct.pack(f"<{n}f", *frame.torques)
    return payload + _CRC.pack(crc16_ccitt(payload))


def decode(raw: bytes) -> TelemetryFrame:
    if len(raw) < _HEADER.size + _CRC.size:
        raise ValueError("frame too short")
    magic, version, n, ts = _HEADER.unpack_from(raw)
    if magic != MAGIC or version != VERSION:
        raise ValueError("bad magic/version")
    expected = _HEADER.size + 8 * n + _CRC.size
    if len(raw) != expected:
        raise ValueError("frame length mismatch")
    body, crc_raw = raw[:-_CRC.size], raw[-_CRC.size:]
    (crc_stored,) = _CRC.unpack(crc_raw)
    if crc16_ccitt(body) != crc_stored:
        raise ValueError("crc mismatch")
    off = _HEADER.size
    positions = struct.unpack_from(f"<{n}f", raw, off)
    torques = struct.unpack_from(f"<{n}f", raw, off + 4 * n)
    return TelemetryFrame(timestamp_us=ts, positions=positions, torques=torques)
