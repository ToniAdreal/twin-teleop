"""Deterministic trajectory evaluation harness (VerifyGate-style).

Given a replayed trajectory, produce a PASS/FAIL verdict decided entirely
by deterministic checks — no model judgement — plus an evidence record
that can be replayed / audited. This is the pattern intended for
robot-policy and world-model evaluation: the same frames + the same
thresholds always produce the same verdict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from telemetry import TelemetryFrame


@dataclass
class EvalConfig:
    max_torque_nm: float = 10.0
    max_joint_velocity_rad_s: float = 5.0
    max_gap_us: int = 100_000  # 100 ms
    target_positions: tuple[float, ...] | None = None
    position_tolerance_rad: float = 0.05


@dataclass
class EvalReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_trajectory(
    frames: Sequence[TelemetryFrame], config: EvalConfig | None = None
) -> EvalReport:
    cfg = config or EvalConfig()
    checks: dict[str, bool] = {}
    evidence: dict = {"frame_count": len(frames)}

    checks["non_empty"] = len(frames) > 0
    if not frames:
        return EvalReport(passed=False, checks=checks, evidence=evidence)

    # Monotonic timestamps + max gap
    gaps = [
        frames[i + 1].timestamp_us - frames[i].timestamp_us
        for i in range(len(frames) - 1)
    ]
    checks["timestamps_monotonic"] = all(g > 0 for g in gaps) if gaps else True
    evidence["max_gap_us"] = max(gaps) if gaps else 0
    checks["gap_within_limit"] = evidence["max_gap_us"] <= cfg.max_gap_us

    # Torque limits
    peak_torque = max(abs(t) for f in frames for t in f.torques)
    evidence["peak_torque_nm"] = peak_torque
    checks["torque_within_limit"] = peak_torque <= cfg.max_torque_nm

    # Joint velocity (finite difference, rad/s)
    peak_vel = 0.0
    for i in range(len(frames) - 1):
        dt_s = (frames[i + 1].timestamp_us - frames[i].timestamp_us) / 1_000_000
        if dt_s <= 0:
            continue
        for j in range(len(frames[i].positions)):
            v = abs(frames[i + 1].positions[j] - frames[i].positions[j]) / dt_s
            peak_vel = max(peak_vel, v)
    evidence["peak_joint_velocity_rad_s"] = peak_vel
    checks["velocity_within_limit"] = peak_vel <= cfg.max_joint_velocity_rad_s

    # Optional: final pose reaches target within tolerance
    if cfg.target_positions is not None:
        final = frames[-1].positions
        errors = [abs(a - b) for a, b in zip(final, cfg.target_positions)]
        evidence["final_position_error_rad"] = max(errors) if errors else 0.0
        checks["target_reached"] = all(
            e <= cfg.position_tolerance_rad for e in errors
        )

    return EvalReport(
        passed=all(checks.values()), checks=checks, evidence=evidence
    )
