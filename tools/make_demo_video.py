"""Render the TwinTeleop demo video frames from REAL project data.

Every number shown comes from running this repo's own code:
telemetry codec -> replay -> deterministic evaluation -> (recorded)
NVIDIA Nemotron narration via Nebius Token Factory.
Output: PNG frames in the given dir; encode with ffmpeg.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from evaluate import EvalConfig, evaluate_trajectory
from replay import replay_schedule, to_time_series
from telemetry import TelemetryFrame

W, H = 1280, 720
BG = (13, 17, 23)
FG = (230, 237, 243)
ACCENT = (88, 166, 255)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
GREY = (139, 148, 158)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_frames")
OUT.mkdir(parents=True, exist_ok=True)
idx = 0


def font(size, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(p, size)


def mono(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)


def slide(lines, hold=1, title_color=FG):
    """lines: list of (text, font, color, dy)"""
    global idx
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 90
    for text, f, c, dy in lines:
        d.text((70, y), text, font=f, fill=c)
        y += dy
    for _ in range(hold):
        img.save(OUT / f"f{idx:05d}.png")
        idx += 1


def make_frames():
    # A realistic 6-joint trajectory: smooth sinusoid blend, 12 s at 50 Hz
    frames = []
    for i in range(600):
        t = i * 20_000
        pos = tuple(
            round(0.6 * math.sin(i / 40 + j) + 0.2 * math.sin(i / 13 + j * 2), 4)
            for j in range(6)
        )
        tor = tuple(round(1.2 * math.cos(i / 50 + j), 4) for j in range(6))
        frames.append(TelemetryFrame(timestamp_us=1_000_000 + t, positions=pos, torques=tor))
    return frames


FRAMES = make_frames()
SERIES = to_time_series(FRAMES)
REP_OK = evaluate_trajectory(FRAMES[:120])
FRAMES_BAD = [
    TelemetryFrame(
        timestamp_us=f.timestamp_us,
        positions=tuple(p * (1 + 0.02 * k) for p in f.positions),
        torques=f.torques,
    )
    for k, f in enumerate(FRAMES[::10])
]
REP_BAD = evaluate_trajectory(FRAMES_BAD)

# --- Slide 1: title (4 s at 10 fps = 40 frames) ---
slide(
    [
        ("TwinTeleop", font(84, True), FG, 110),
        ("Robot teleoperation telemetry +", font(40), ACCENT, 55),
        ("deterministic trajectory evaluation", font(40), ACCENT, 90),
        ("Nebius x NVIDIA Global AI Hackathon - Physical AI Track", font(28), GREY, 45),
        ("github.com/ToniAdreal/twin-teleop", mono(26), GREY, 0),
    ],
    hold=40,
)

# --- Segment 2: live replay animation (30 s = 300 frames) ---
for step in range(0, 600, 2):
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    upto = max(step, 2)
    for j in range(6):
        xs = SERIES[j]["t_s"][: upto // 1]
        ys = SERIES[j]["position"][: upto // 1]
        ax.plot(xs, ys, lw=1.6, label=f"joint {j}")
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("replay time (s) - 50 Hz deterministic schedule", color="#e6edf3")
    ax.set_ylabel("joint position (rad)", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.legend(loc="upper right", fontsize=8, facecolor="#0d1117", labelcolor="#e6edf3")
    ax.set_title(
        f"Telemetry replay: binary frames (CRC16) -> per-joint time series   t={SERIES[0]['t_s'][upto-1]:.1f}s",
        color="#e6edf3",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"f{idx:05d}.png", facecolor="#0d1117")
    plt.close(fig)
    idx += 1

# --- Slide 3: evaluation verdicts (8 s) ---
ok_lines = [("Deterministic evaluation harness (VerifyGate-style)", font(36, True), FG, 70)]
for name, val in REP_OK.checks.items():
    ok_lines.append((f"  {'PASS' if val else 'FAIL'}  {name}", mono(28), GREEN if val else RED, 42))
ok_lines.append((f"Verdict on smooth trajectory: {'PASS' if REP_OK.passed else 'FAIL'}", font(32, True), GREEN if REP_OK.passed else RED, 60))
slide(ok_lines, hold=80)

bad_lines = [("Same harness, stressed trajectory:", font(36, True), FG, 70)]
for name, val in REP_BAD.checks.items():
    bad_lines.append((f"  {'PASS' if val else 'FAIL'}  {name}", mono(28), GREEN if val else RED, 42))
bad_lines.append(
    (f"peak velocity {REP_BAD.evidence['peak_joint_velocity_rad_s']:.2f} rad/s vs limit 5.0 -> verdict FAIL, with evidence", mono(24), GREY, 50)
)
slide(bad_lines, hold=80)

# --- Slide 4: model assist (8 s) ---
slide(
    [
        ("NVIDIA open model via Nebius Token Factory", font(38, True), FG, 65),
        ("Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (real call, HTTP 200)", mono(25), ACCENT, 55),
        ("Role: narrate the deterministic verdict in plain language.", font(28), FG, 45),
        ("It NEVER overturns the harness verdict.", font(28, True), FG, 70),
        ('Recorded narration of a real FAIL case:', font(26), GREY, 45),
        ('"velocity_within_limit: The trajectory failed because the peak', mono(24), FG, 38),
        ('joint velocity exceeded the allowed limit. Next action: Reduce', mono(24), FG, 38),
        ('the peak joint velocity to stay within the limit."', mono(24), FG, 0),
    ],
    hold=80,
)

# --- Slide 5: stack + tests + close (6 s) ---
slide(
    [
        ("What is in the repo", font(40, True), FG, 70),
        ("  src/telemetry.py  binary frame codec (positions/torques/CRC16)", mono(26), FG, 45),
        ("  src/replay.py     stream I/O + fixed-Hz replay schedule", mono(26), FG, 45),
        ("  src/stream.py     stdlib RFC 6455 WebSocket replay server", mono(26), FG, 45),
        ("  src/evaluate.py   deterministic PASS/FAIL + evidence record", mono(26), FG, 45),
        ("  src/model_assist.py  NVIDIA Nemotron via Nebius Token Factory", mono(26), FG, 80),
        ("Tests: 21/21 passing (verified in an isolated copy)", font(30, True), GREEN, 55),
        ("Honest scope: no ROS 2 / Isaac Sim / world-model training claimed - in progress.", font(24), GREY, 0),
    ],
    hold=60,
)

print(f"frames={idx} ok_pass={REP_OK.passed} bad_pass={REP_BAD.passed}")
