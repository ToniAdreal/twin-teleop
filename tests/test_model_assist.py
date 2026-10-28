"""Tests for model_assist against a local stub OpenAI-compatible server.

No real endpoint or key is touched: the stub records the request and
returns a canned chat-completions payload, so the tests stay hermetic
and runnable offline.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_assist import (  # noqa: E402
    ModelConfig,
    assess_trajectory,
    build_assessment_prompt,
)

REPORT = {
    "passed": False,
    "checks": {
        "non_empty": True,
        "timestamps_monotonic": True,
        "gap_within_limit": True,
        "torque_within_limit": False,
        "velocity_within_limit": True,
    },
    "evidence": {
        "frame_count": 12,
        "peak_torque_nm": 12.5,
        "peak_joint_velocity_rad_s": 1.2,
        "max_gap_us": 20000,
    },
}


def test_prompt_carries_verdict_and_failing_check():
    prompt = build_assessment_prompt(REPORT)
    assert "Verdict: FAIL" in prompt
    assert "torque_within_limit: FAILED" in prompt
    assert "12.5" in prompt  # peak torque number is in the prompt
    assert "Do not change the verdict" in prompt


def test_prompt_pass_case():
    ok = dict(REPORT, passed=True)
    ok["checks"] = dict(REPORT["checks"], torque_within_limit=True)
    prompt = build_assessment_prompt(ok)
    assert "Verdict: PASS" in prompt
    assert "torque_within_limit: ok" in prompt


def _stub_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def test_assess_posts_openai_shape_and_parses_reply():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/chat/completions"
            length = int(self.headers["Content-Length"])
            seen["body"] = json.loads(self.rfile.read(length).decode())
            seen["auth"] = self.headers.get("Authorization")
            payload = {
                "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Torque check failed; inspect joint torque next.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # keep test output clean
            pass

    server, port = _stub_server(Handler)
    try:
        out = assess_trajectory(
            REPORT,
            ModelConfig(
                base_url=f"http://127.0.0.1:{port}",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                api_key="test-key-not-real",
            ),
        )
    finally:
        server.shutdown()
    assert seen["body"]["model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert "Verdict: FAIL" in seen["body"]["messages"][1]["content"]
    assert seen["auth"] == "Bearer test-key-not-real"
    assert out["model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert "Torque check failed" in out["content"]
    assert out["finish_reason"] == "stop"


def test_assess_without_key_raises_before_any_request():
    with pytest.raises(ValueError, match="No model API key"):
        assess_trajectory(REPORT, ModelConfig(api_key=""))


def test_assess_non200_raises_runtime_error():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server, port = _stub_server(Handler)
    try:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            assess_trajectory(
                REPORT,
                ModelConfig(base_url=f"http://127.0.0.1:{port}", api_key="x"),
            )
    finally:
        server.shutdown()
