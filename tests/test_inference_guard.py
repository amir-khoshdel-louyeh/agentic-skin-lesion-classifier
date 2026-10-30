"""Unit tests for bounded local-inference retry and timeout guards."""

import json
import subprocess
import urllib.error

import pytest

import skin_agent
import tools.vlm_ollama as vlm
from control.inference_guard import (
    DEFAULT_MAX_ATTEMPTS,
    MAX_ATTEMPTS,
    MAX_TIMEOUT_SECONDS,
    InferencePolicy,
    InferenceTimeout,
    run_with_retries,
)


class Retryable(RuntimeError):
    pass


def test_policy_from_env_clamps_unbounded_values(monkeypatch):
    monkeypatch.setenv("TEST_INFERENCE_TIMEOUT", "99999")
    monkeypatch.setenv("TEST_INFERENCE_ATTEMPTS", "99")
    monkeypatch.setenv("TEST_INFERENCE_TOTAL_TIMEOUT", "99999")

    policy = InferencePolicy.from_env(
        "TEST", default_timeout=10, default_attempts=2, default_total=20
    )

    assert policy.timeout_s == MAX_TIMEOUT_SECONDS
    assert policy.max_attempts == MAX_ATTEMPTS
    assert policy.max_total_s == 1800.0

    monkeypatch.setenv("TEST_INFERENCE_TIMEOUT", "nonsense")
    monkeypatch.setenv("TEST_INFERENCE_ATTEMPTS", "-5")
    monkeypatch.setenv("TEST_INFERENCE_TOTAL_TIMEOUT", "nonsense")
    fallback = InferencePolicy.from_env("TEST", default_timeout=10,
                                        default_attempts=2, default_total=20)
    assert fallback.timeout_s == 10
    assert fallback.max_attempts == 1
    assert fallback.max_total_s == 20


def test_policy_rejects_invalid_direct_values():
    with pytest.raises(ValueError):
        InferencePolicy(timeout_s=0)
    with pytest.raises(ValueError):
        InferencePolicy(max_attempts=0)
    with pytest.raises(ValueError):
        InferencePolicy(timeout_s=10, max_total_s=5)
    with pytest.raises(ValueError):
        InferencePolicy(backoff_s=-1)


def test_retries_only_transient_errors_and_caps_attempts():
    clock = [0.0]
    sleeps = []
    calls = []

    def now():
        return clock[0]

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def operation(limit):
        calls.append(limit)
        clock[0] += 0.1
        raise Retryable("temporary")

    policy = InferencePolicy(
        timeout_s=2, max_attempts=3, max_total_s=10,
        backoff_s=0.1, max_backoff_s=0.2,
    )
    with pytest.raises(Retryable):
        run_with_retries(
            operation, policy, lambda exc: isinstance(exc, Retryable),
            sleep=sleep, clock=now,
        )

    assert len(calls) == 3
    assert sleeps == [0.1, 0.2]


def test_non_retryable_and_total_budget_stop_immediately():
    calls = []

    def permanent(_limit):
        calls.append(1)
        raise ValueError("permanent")

    policy = InferencePolicy(timeout_s=2, max_attempts=3, max_total_s=10)
    with pytest.raises(ValueError):
        run_with_retries(permanent, policy, lambda _exc: False)
    assert len(calls) == 1

    clock = [0.0]

    def now():
        return clock[0]

    def slow(_limit):
        calls.append(2)
        clock[0] += 3
        raise Retryable("slow")

    tight = InferencePolicy(
        timeout_s=2, max_attempts=3, max_total_s=2.5,
        backoff_s=0.1,
    )
    with pytest.raises(InferenceTimeout):
        run_with_retries(
            slow, tight, lambda exc: isinstance(exc, Retryable),
            clock=now,
        )
    assert calls[-1] == 2


def test_openclaw_timeout_aborts_without_retry(monkeypatch):
    calls = []

    monkeypatch.setattr(skin_agent, "find_openclaw_executable",
                        lambda: "openclaw")

    def timed_out(cmd, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(skin_agent.subprocess, "run", timed_out)
    policy = InferencePolicy(
        timeout_s=5, max_attempts=3, max_total_s=15,
    )

    with pytest.raises(RuntimeError, match="timed out"):
        skin_agent.run_openclaw_cli("hello", show_command=False, policy=policy)

    assert len(calls) == 1
    assert calls[0]["timeout"] <= 5


def test_openclaw_transient_retries_use_fresh_bounded_sessions(monkeypatch):
    commands = []
    sleeps = []

    monkeypatch.setattr(skin_agent, "find_openclaw_executable",
                        lambda: "openclaw")
    monkeypatch.setattr(skin_agent.time, "sleep", sleeps.append)

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "HTTP 429 Too Many Requests"

    def run(cmd, **kwargs):
        commands.append(cmd)
        if len(commands) < 3:
            return Failed()
        return type("Ok", (), {
            "returncode": 0,
            "stdout": json.dumps({"payloads": [{"text": "ok"}]}),
            "stderr": "",
        })()

    monkeypatch.setattr(skin_agent.subprocess, "run", run)
    policy = InferencePolicy(
        timeout_s=5, max_attempts=3, max_total_s=15,
        backoff_s=0.1,
    )

    result = skin_agent.run_openclaw_cli(
        "hello", show_command=False, policy=policy
    )

    assert result == {"payloads": [{"text": "ok"}]}
    assert len(commands) == 3
    session_ids = [
        cmd[cmd.index("--session-id") + 1] for cmd in commands
    ]
    assert len(set(session_ids)) == 3
    assert sleeps == [0.1, 0.2]
    assert DEFAULT_MAX_ATTEMPTS == 3


def test_openclaw_model_override_is_forwarded(monkeypatch):
    commands = []
    monkeypatch.setattr(skin_agent, "find_openclaw_executable",
                        lambda: "openclaw")

    def run(cmd, **kwargs):
        commands.append(cmd)
        return type("Ok", (), {
            "returncode": 0,
            "stdout": json.dumps({"payloads": [{"text": "ok"}]}),
            "stderr": "",
        })()

    monkeypatch.setattr(skin_agent.subprocess, "run", run)
    policy = InferencePolicy(timeout_s=5, max_attempts=1, max_total_s=5)

    skin_agent.run_openclaw_cli(
        "hello", model="ollama/qwen3:8b", show_command=False,
        policy=policy,
    )
    assert commands[0][commands[0].index("--model") + 1] == "ollama/qwen3:8b"

    commands.clear()
    monkeypatch.setenv("SKIN_AGENT_MODEL", "ollama/qwen3:8b")
    skin_agent.run_openclaw_cli("hello", show_command=False, policy=policy)
    assert commands[0][commands[0].index("--model") + 1] == "ollama/qwen3:8b"


def test_vlm_ollama_retries_429_and_stops_at_attempt_cap(monkeypatch):
    attempts = []
    response = type("Response", (), {
        "read": lambda self: json.dumps({
            "message": {"content": "{}"},
        }).encode(),
        "__enter__": lambda self: self,
        "__exit__": lambda self, *args: False,
    })()

    def urlopen(_req, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(
                "http://localhost:11434", 429, "Too Many Requests", None, None
            )
        return response

    monkeypatch.setattr(vlm.urllib.request, "urlopen", urlopen)
    policy = InferencePolicy(
        timeout_s=5, max_attempts=3, max_total_s=15,
        backoff_s=0,
    )

    result = vlm.chat("qwen3-vl:8b", "image", timeout=5, policy=policy)

    assert result["message"]["content"] == "{}"
    assert len(attempts) == 3
    assert all(limit <= 5 for limit in attempts)
