"""Minimal WebSocket streaming server for TwinTeleop telemetry replay.

Standard-library only (no third-party websocket dependency): the server
performs the RFC 6455 HTTP Upgrade handshake, then streams recorded
telemetry frames as binary WebSocket messages at a fixed replay rate,
following ``replay.replay_schedule``. This is the transport layer a
Three.js digital-twin front end (or any WebSocket client) connects to in
order to watch a recorded teleoperation session play back.

Only one client is served at a time; after the last frame the server
sends a Close frame and shuts the connection down. Nothing here touches
a real robot: the input is always a recorded frame sequence.
"""

from __future__ import annotations

import base64
import hashlib
import socket
import struct
import threading
import time
from typing import Sequence

from replay import replay_schedule
from telemetry import TelemetryFrame, encode

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def websocket_accept(key: str) -> str:
    """RFC 6455 Sec-WebSocket-Accept value for a client key."""
    digest = hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_ws_frame(payload: bytes, opcode: int = 0x2) -> bytes:
    """Encode one server-to-client WebSocket frame (FIN set, unmasked)."""
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([n])
    elif n < 65536:
        header += bytes([126]) + struct.pack(">H", n)
    else:
        header += bytes([127]) + struct.pack(">Q", n)
    return header + payload


class TelemetryServer:
    """Replay recorded frames to a single WebSocket client at fixed Hz."""

    def __init__(
        self,
        frames: Sequence[TelemetryFrame],
        hz: float = 50.0,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        self._frames = list(frames)
        self._hz = hz
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(1)
        self._ready = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "TelemetryServer":
        self._thread.start()
        return self

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    def wait_done(self, timeout: float = 30.0) -> bool:
        return self._done.wait(timeout)

    def stop(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
        self._done.set()

    def _run(self) -> None:
        self._ready.set()
        try:
            conn, _addr = self._sock.accept()
        except OSError:
            return
        try:
            self._handshake(conn)
            self._stream(conn)
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self._done.set()

    def _handshake(self, conn: socket.socket) -> None:
        request = b""
        while b"\r\n\r\n" not in request:
            chunk = conn.recv(4096)
            if not chunk:
                raise ValueError("client closed during handshake")
            request += chunk
        key = None
        for line in request.decode("latin-1").split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                key = line.split(":", 1)[1].strip()
        if not key:
            raise ValueError("missing Sec-WebSocket-Key")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {websocket_accept(key)}\r\n\r\n"
        )
        conn.sendall(response.encode("ascii"))

    def _stream(self, conn: socket.socket) -> None:
        start = time.monotonic()
        for offset, frame in replay_schedule(self._frames, self._hz):
            wait = start + offset - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            conn.sendall(encode_ws_frame(encode(frame), opcode=0x2))
        conn.sendall(encode_ws_frame(b"", opcode=0x8))  # Close


def serve_frames(
    frames: Sequence[TelemetryFrame], hz: float = 50.0
) -> TelemetryServer:
    """Start a replay server in a background thread and return it."""
    server = TelemetryServer(frames, hz=hz).start()
    if not server.wait_ready():
        raise RuntimeError("telemetry server did not start")
    return server
