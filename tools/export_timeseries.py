"""Export a TwinTeleop telemetry stream as JSON time series for the viewer.

Reads a recorded stream (concatenated binary frames, see telemetry.py)
and writes the per-joint time series produced by replay.to_time_series
as JSON, so a front end can drive a digital-twin arm without any
telemetry dependency:

    python3 tools/export_timeseries.py demo.bin viewer/sample_timeseries.json

`--demo` writes a deterministic SYNTHETIC 2-joint trajectory instead
(sine sweep, 2 s at 50 Hz). It is labelled "synthetic-demo" in the
output and exists only so the viewer runs out of the box; it is not
recorded hardware data and must never be presented as such.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replay import read_stream, to_time_series  # noqa: E402


def series_to_json(series: dict, source: str) -> dict:
    joints = []
    for j in sorted(series):
        ts = series[j]
        joints.append(
            {
                "t_s": list(ts["t_s"]),
                "position": list(ts["position"]),
                "torque": list(ts["torque"]),
            }
        )
    return {"source": source, "joint_count": len(joints), "joints": joints}


def export_stream(stream_path: str | Path, out_path: str | Path) -> dict:
    data = series_to_json(to_time_series(read_stream(stream_path)), "recorded-stream")
    Path(out_path).write_text(json.dumps(data, indent=2) + "\n")
    return data


def demo_series() -> dict:
    """Deterministic synthetic 2-joint sine sweep, 2 s at 50 Hz."""
    hz, seconds = 50, 2.0
    n = int(hz * seconds) + 1
    t = [i / hz for i in range(n)]
    joints = [
        {
            "t_s": t,
            "position": [0.6 * math.sin(2 * math.pi * 0.5 * x) for x in t],
            "torque": [0.8 * math.cos(2 * math.pi * 0.5 * x) for x in t],
        },
        {
            "t_s": t,
            "position": [0.4 * math.sin(2 * math.pi * 0.5 * x + 0.9) for x in t],
            "torque": [0.5 * math.cos(2 * math.pi * 0.5 * x + 0.9) for x in t],
        },
    ]
    return {"source": "synthetic-demo", "joint_count": 2, "joints": joints}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", nargs="?", help="recorded .bin stream")
    parser.add_argument("out", help="output .json path")
    parser.add_argument("--demo", action="store_true",
                        help="write the synthetic demo series instead of reading a stream")
    args = parser.parse_args(argv)
    if args.demo:
        data = demo_series()
    else:
        if not args.stream:
            parser.error("stream path required unless --demo is given")
        data = export_stream(args.stream, args.out)
        return 0
    Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
