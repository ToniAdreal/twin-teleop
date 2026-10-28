# TwinTeleop — Robot Teleoperation Telemetry & Digital-Twin Replay

Seed project of Toni Adreal's embodied-AI portfolio line (started 2026-10-26).

A minimal, runnable re-implementation of the telemetry layer behind a robot
teleoperation digital twin, distilled from production experience at
CyberOrigin (space-computing / robot teleoperation platform, 2025):

- **Binary telemetry frames** — compact little-endian frames carrying
  multi-axis manipulator joint positions (rad) and joint torques (N·m) with a
  microsecond timestamp and CRC16 integrity check, designed for
  WebSocket `ArrayBuffer` transport at high frequency with <30 ms
  end-to-end latency budgets.
- **Parser + validator** (`src/telemetry.py`) — decode/encode frames, reject
  truncated or corrupted frames deterministically.
- **Replay utilities** (`src/replay.py`) — read/write concatenated frame
  streams, turn a recorded stream into per-joint time series for a
  Three.js digital-twin front end, and generate a deterministic replay
  schedule at a fixed Hz for a transport layer (e.g. WebSocket) to follow.
- **Trajectory evaluation harness** (`src/evaluate.py`) — deterministic
  PASS/FAIL scoring of replayed trajectories (timestamp monotonicity,
  gap, torque, joint-velocity and optional target-pose checks) with an
  auditable evidence record, VerifyGate-style: same frames + same
  thresholds always produce the same verdict.

## Run the tests

```bash
cd twin-teleop
python3 -m pytest tests/ -q
```

## Roadmap (built in public, one verifiable milestone at a time)

1. [x] Binary frame codec + CRC + unit tests
2. [x] Replay utilities: stream I/O, per-joint time series, fixed-Hz
       deterministic replay schedule (13 tests passing, verified in an
       isolated copy on 2026-10-26)
3. [x] Trajectory evaluation harness (v1): deterministic PASS/FAIL
       scoring of replayed trajectories with evidence records
4. [x] WebSocket streaming server replaying recorded frames at fixed Hz
       (stdlib-only RFC 6455 server in `src/stream.py`, binary frames a
       Three.js front end can consume; 16 tests passing, verified in an
       isolated copy on 2026-10-27)
5. [x] NVIDIA open-model trajectory assessment via Nebius Token Factory
       (`src/model_assist.py`, OpenAI-compatible client, stdlib-only):
       the deterministic harness verdict is narrated — never overturned —
       by `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` served on Token Factory.
       Real call evidence (2026-10-28, HTTP 200) in
       `NEBIUS_RUNTIME_CALL.md`; hermetic stub-server tests in
       `tests/test_model_assist.py`. This satisfies the Nebius x NVIDIA
       hackathon's "must run on Token Factory + use an NVIDIA open model"
       requirement for the Physical AI track submission.
6. [ ] Three.js digital-twin viewer (InstancedMesh joint rendering, 60 FPS)
7. [ ] World-model / policy evaluation: extend the harness to learned
       policies / predicted trajectories

## Honesty note

This is a from-scratch portfolio re-implementation of patterns used in
production; it contains no proprietary CyberOrigin code or data.
