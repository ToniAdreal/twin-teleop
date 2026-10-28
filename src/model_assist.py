"""Trajectory assessment via an NVIDIA open model on an OpenAI-compatible
endpoint (Nebius Token Factory in the hackathon submission).

This module is deliberately thin: it takes the deterministic verdict
produced by :mod:`evaluate` (PASS/FAIL + evidence numbers, never model
judgement) and asks an NVIDIA open model to explain that verdict in plain
language and suggest one concrete next action for the operator — e.g.
re-run the trajectory, inspect a joint, or lower a velocity limit.

Design rules, kept honest for the submission:
- No credentials live in this file or in the repo. The endpoint, model
  and key come from the environment (``TWINTELEOP_MODEL_BASE_URL``,
  ``TWINTELEOP_MODEL_NAME``, ``TWINTELEOP_MODEL_API_KEY``). In the
  hackathon demo the call is made through Nebius Token Factory, whose
  OpenAI-compatible ``/v1/chat/completions`` shape this client speaks.
- The deterministic verdict is never overridden by the model: the model
  only narrates it. If the endpoint is unreachable, the caller gets an
  exception and the verdict still stands on its own.
- Stdlib only (``urllib``), so the module runs wherever the rest of
  TwinTeleop runs.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


@dataclass
class ModelConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout_s: float = 120.0


def config_from_env() -> ModelConfig:
    """Build a config from the environment; empty key means 'not configured'."""
    return ModelConfig(
        base_url=os.environ.get("TWINTELEOP_MODEL_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("TWINTELEOP_MODEL_NAME", DEFAULT_MODEL),
        api_key=os.environ.get("TWINTELEOP_MODEL_API_KEY", ""),
    )


def build_assessment_prompt(report: dict) -> str:
    """Render the deterministic evaluation report as a model prompt.

    ``report`` is the ``to_dict()`` output of :class:`evaluate.EvalReport`:
    ``passed``, ``checks`` (name -> bool) and ``evidence`` (numbers).
    """
    checks = report.get("checks", {})
    evidence = report.get("evidence", {})
    verdict = "PASS" if report.get("passed") else "FAIL"
    lines = [
        "You are assisting a robot teleoperation operator. A recorded "
        "manipulator trajectory was evaluated by a deterministic harness "
        "(no model judgement involved). Explain the result in plain "
        "language and suggest exactly one concrete next action.",
        "",
        f"Verdict: {verdict}",
        f"Frames evaluated: {evidence.get('frame_count')}",
        f"Peak torque (N*m): {evidence.get('peak_torque_nm')}",
        f"Peak joint velocity (rad/s): {evidence.get('peak_joint_velocity_rad_s')}",
        f"Max timestamp gap (us): {evidence.get('max_gap_us')}",
        "Checks:",
    ]
    for name in sorted(checks):
        lines.append(f"- {name}: {'ok' if checks[name] else 'FAILED'}")
    if "final_position_error_rad" in evidence:
        lines.append(
            f"Final position error (rad): {evidence['final_position_error_rad']}"
        )
    lines.append("")
    lines.append(
        "Do not change the verdict. If any check FAILED, name the failing "
        "check first, then give the single next action."
    )
    return "\n".join(lines)


def assess_trajectory(report: dict, config: ModelConfig | None = None) -> dict:
    """Send ``report`` to the configured model; return model + content.

    Raises RuntimeError on non-200 responses and ValueError when no API
    key is configured, so a missing endpoint can never silently fake a
    successful assessment.
    """
    cfg = config or config_from_env()
    if not cfg.api_key:
        raise ValueError(
            "No model API key configured: set TWINTELEOP_MODEL_API_KEY "
            "(Nebius Token Factory key for the hackathon demo)."
        )
    payload = {
        "model": cfg.model,
        "messages": [
            {
                "role": "system",
                "content": "You explain robot trajectory evaluation results "
                "concisely and never overturn the harness verdict.",
            },
            {"role": "user", "content": build_assessment_prompt(report)},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {cfg.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Model endpoint returned HTTP {e.code}") from e
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Model endpoint returned no choices")
    message = choices[0].get("message") or {}
    return {
        "model": body.get("model", cfg.model),
        "content": message.get("content") or message.get("reasoning_content"),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": body.get("usage"),
    }
