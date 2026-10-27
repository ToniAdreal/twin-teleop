import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stream import encode_ws_frame, serve_frames, websocket_accept  # noqa: E402
from telemetry import TelemetryFrame, decode  # noqa: E402


def make_frames(n=5, joints=3):
    return [
        TelemetryFrame(
            timestamp_us=1_000_000 + i * 20_000,
            positions=tuple(0.1 * i + 0.01 * j for j in range(joints)),
            torques=tuple(0.5 for _ in range(joints)),
        )
        for i in range(n)
    ]


def read_exact_from(sock, buf, n):
    """Read exactly n bytes, first draining any buffered leftover bytes."""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed early")
        buf += chunk
    return buf[:n], buf[n:]


def ws_client(port):
    """Raw-socket WebSocket client: handshake, returning (sock, leftover).

    The handshake recv may coalesce bytes past the header terminator, so
    any bytes after \\r\\n\\r\\n are returned as leftover for the frame
    reader instead of being discarded.
    """
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = "dGhlIHNhbXBsZSBub25jZQ=="  # RFC 6455 example key
    sock.sendall(
        (
            "GET /replay HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
    )
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(4096)
    head, leftover = resp.split(b"\r\n\r\n", 1)
    assert b"101" in head.split(b"\r\n", 1)[0]
    assert websocket_accept(key).encode("ascii") in head
    return sock, leftover


def read_ws_messages(sock, buf=b""):
    """Read (opcode, payload) pairs until the Close frame."""
    messages = []
    while True:
        head, buf = read_exact_from(sock, buf, 2)
        fin_opcode = head[0]
        opcode = fin_opcode & 0x0F
        assert fin_opcode & 0x80, "server must set FIN"
        length = head[1] & 0x7F
        assert not (head[1] & 0x80), "server frames must be unmasked"
        if length == 126:
            raw, buf = read_exact_from(sock, buf, 2)
            length = int.from_bytes(raw, "big")
        elif length == 127:
            raw, buf = read_exact_from(sock, buf, 8)
            length = int.from_bytes(raw, "big")
        payload, buf = read_exact_from(sock, buf, length) if length else (b"", buf)
        if opcode == 0x8:
            break
        messages.append((opcode, payload))
    return messages


def test_replay_stream_order_and_content():
    frames = make_frames(5, joints=3)
    server = serve_frames(frames, hz=200.0)
    try:
        sock, leftover = ws_client(server.port)
        try:
            messages = read_ws_messages(sock, leftover)
        finally:
            sock.close()
        assert [op for op, _ in messages] == [0x2] * 5
        decoded = [decode(payload) for _, payload in messages]
        for got, want in zip(decoded, frames):
            assert got.timestamp_us == want.timestamp_us
            assert got.positions == pytest.approx(want.positions)
            assert got.torques == pytest.approx(want.torques)
    finally:
        server.wait_done(timeout=10)
        server.stop()


def test_extended_length_encoding_for_wide_frames():
    # 16 joints -> 12 + 128 + 2 = 142-byte telemetry frame, forcing the
    # 126 extended-length WebSocket header form.
    frames = make_frames(2, joints=16)
    assert len(encode_ws_frame(b"x" * 142)) == 4 + 142
    server = serve_frames(frames, hz=200.0)
    try:
        sock, leftover = ws_client(server.port)
        try:
            messages = read_ws_messages(sock, leftover)
        finally:
            sock.close()
        assert len(messages) == 2
        assert decode(messages[0][1]).positions == pytest.approx(
            frames[0].positions
        )
    finally:
        server.wait_done(timeout=10)
        server.stop()


def test_known_accept_value():
    # RFC 6455 section 1.3 worked example.
    assert (
        websocket_accept("dGhlIHNhbXBsZSBub25jZQ==")
        == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    )
