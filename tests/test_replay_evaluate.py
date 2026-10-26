import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import EvalConfig, evaluate_trajectory  # noqa: E402
from replay import (  # noqa: E402
    iter_frames_from_bytes,
    read_stream,
    replay_schedule,
    to_time_series,
    write_stream,
)
from telemetry import TelemetryFrame, encode  # noqa: E402


def make_frames(n=5, dt_us=20_000, torque=1.0):
    return [
        TelemetryFrame(
            timestamp_us=1_000_000 + i * dt_us,
            positions=(0.08 * i, -0.04 * i),
            torques=(torque, -torque),
        )
        for i in range(n)
    ]


def test_stream_roundtrip_bytes():
    frames = make_frames()
    data = b"".join(encode(f) for f in frames)
    out = list(iter_frames_from_bytes(data))
    assert len(out) == 5
    assert out[-1].positions == pytest.approx(frames[-1].positions)


def test_stream_roundtrip_file(tmp_path):
    frames = make_frames()
    p = tmp_path / "rec.tt"
    written = write_stream(p, frames)
    assert written > 0
    assert read_stream(p) == frames or len(read_stream(p)) == 5


def test_to_time_series():
    series = to_time_series(make_frames())
    assert set(series) == {0, 1}
    assert series[0]["t_s"][0] == 0.0
    assert abs(series[0]["t_s"][1] - 0.02) < 1e-9
    assert abs(series[1]["position"][2] + 0.08) < 1e-6


def test_replay_schedule_deterministic():
    sched = list(replay_schedule(make_frames(), hz=50.0))
    assert [round(o, 3) for o, _ in sched] == [0.0, 0.02, 0.04, 0.06, 0.08]


def test_evaluate_pass():
    report = evaluate_trajectory(make_frames())
    assert report.passed is True
    assert report.checks["torque_within_limit"] is True
    assert report.evidence["frame_count"] == 5


def test_evaluate_fails_on_torque():
    report = evaluate_trajectory(make_frames(torque=99.0))
    assert report.passed is False
    assert report.checks["torque_within_limit"] is False


def test_evaluate_fails_on_gap_and_target():
    frames = make_frames()
    frames = [frames[0]] + [
        TelemetryFrame(
            timestamp_us=f.timestamp_us + 500_000,
            positions=f.positions,
            torques=f.torques,
        )
        for f in frames[1:]
    ]
    report = evaluate_trajectory(
        frames,
        EvalConfig(target_positions=(9.0, 9.0)),
    )
    assert report.passed is False
    assert report.checks["gap_within_limit"] is False
    assert report.checks["target_reached"] is False


def test_evaluate_empty_fails():
    report = evaluate_trajectory([])
    assert report.passed is False
    assert report.checks["non_empty"] is False
