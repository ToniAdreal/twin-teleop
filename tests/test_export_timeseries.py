import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replay import write_stream  # noqa: E402
from telemetry import TelemetryFrame  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "export_timeseries.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("export_timeseries", TOOLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_frames(n=4, dt_us=20_000):
    return [
        TelemetryFrame(
            timestamp_us=i * dt_us,
            positions=(0.1 * i, -0.2 * i),
            torques=(1.0, 0.5),
        )
        for i in range(n)
    ]


def test_export_stream_round_trip(tmp_path):
    tool = load_tool()
    stream = tmp_path / "s.bin"
    write_stream(stream, make_frames())
    out = tmp_path / "s.json"
    data = tool.export_stream(stream, out)
    assert data["source"] == "recorded-stream"
    assert data["joint_count"] == 2
    assert json.loads(out.read_text()) == data
    j0 = data["joints"][0]
    assert j0["t_s"] == [0.0, 0.02, 0.04, 0.06]
    assert j0["position"] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert j0["torque"] == [1.0, 1.0, 1.0, 1.0]


def test_demo_series_is_deterministic_and_labelled():
    tool = load_tool()
    a, b = tool.demo_series(), tool.demo_series()
    assert a == b
    assert a["source"] == "synthetic-demo"
    assert a["joint_count"] == 2
    t = a["joints"][0]["t_s"]
    assert t[0] == 0.0 and t[-1] == 2.0 and len(t) == 101
    assert all(t[i] < t[i + 1] for i in range(len(t) - 1))
    assert len(a["joints"][0]["position"]) == len(t)
