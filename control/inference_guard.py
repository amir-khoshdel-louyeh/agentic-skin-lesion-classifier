"""Bounded timeout/retry policy for local inference calls.

Every local Ollama/OpenClaw request must have a finite per-attempt timeout,
a finite total budget, and a small attempt cap. Retries are allowed only for
explicitly transient errors; timeouts and permanent failures are never
retried. This module is dependency-free so the policy can be unit-tested
without starting Ollama or OpenClaw.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_ATTEMPTS = 3
MAX_ATTEMPTS = 4
DEFAULT_MAX_TOTAL_SECONDS = 900.0
MAX_TOTAL_SECONDS = 1800.0
DEFAULT_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0

T = TypeVar("T")


class InferenceTimeout(RuntimeError):
    """The per-attempt or total inference budget was exhausted."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class InferencePolicy:
    """Finite limits for one local inference operation."""

    timeout_s: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_total_s: float = DEFAULT_MAX_TOTAL_SECONDS
    backoff_s: float = DEFAULT_BACKOFF_SECONDS
    max_backoff_s: float = MAX_BACKOFF_SECONDS

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.max_attempts > MAX_ATTEMPTS:
            raise ValueError(f"max_attempts must be <= {MAX_ATTEMPTS}")
        if self.max_total_s < self.timeout_s:
            raise ValueError("max_total_s must be >= timeout_s")
        if self.max_total_s > MAX_TOTAL_SECONDS:
            raise ValueError(f"max_total_s must be <= {MAX_TOTAL_SECONDS}")
        if self.backoff_s < 0 or self.max_backoff_s < 0:
            raise ValueError("backoff values cannot be negative")

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        default_attempts: int = DEFAULT_MAX_ATTEMPTS,
        default_total: Optional[float] = None,
    ) -> "InferencePolicy":
        """Read finite limits from `<PREFIX>_INFERENCE_*` environment vars.

        Invalid, negative, or oversized values are clamped instead of
        producing an unbounded request.
        """
        timeout = min(
            max(_env_float(f"{prefix}_INFERENCE_TIMEOUT", default_timeout), 1.0),
            MAX_TIMEOUT_SECONDS,
        )
        attempts = min(
            max(_env_int(f"{prefix}_INFERENCE_ATTEMPTS", default_attempts), 1),
            MAX_ATTEMPTS,
        )
        total_default = max(default_total or DEFAULT_MAX_TOTAL_SECONDS, timeout)
        total = min(
            max(_env_float(f"{prefix}_INFERENCE_TOTAL_TIMEOUT", total_default), timeout),
            MAX_TOTAL_SECONDS,
        )
        return cls(
            timeout_s=timeout,
            max_attempts=attempts,
            max_total_s=total,
        )

    def delay_for(self, failed_attempt: int) -> float:
        """Exponential backoff for a zero-based failed attempt index."""
        if failed_attempt < 0:
            raise ValueError("failed_attempt must be non-negative")
        return min(
            self.backoff_s * (2 ** failed_attempt),
            self.max_backoff_s,
        )


def run_with_retries(
    operation: Callable[[float], T],
    policy: InferencePolicy,
    retryable: Callable[[Exception], bool],
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
) -> T:
    """Run `operation(timeout_limit)` under the policy's finite budgets.

    `operation` receives the smaller of the per-attempt timeout and the
    remaining total budget. Only exceptions accepted by `retryable` are
    retried; the final attempt always re-raises.
    """
    started = clock()
    last_error: Optional[Exception] = None
    for attempt in range(policy.max_attempts):
        remaining = policy.max_total_s - (clock() - started)
        if remaining <= 0:
            raise InferenceTimeout(
                f"inference total budget of {policy.max_total_s:g}s exhausted"
            ) from last_error
        try:
            return operation(min(policy.timeout_s, remaining))
        except Exception as exc:  # noqa: BLE001 - policy decides retryability
            last_error = exc
            if attempt + 1 >= policy.max_attempts or not retryable(exc):
                raise
            delay = policy.delay_for(attempt)
            remaining = policy.max_total_s - (clock() - started)
            if delay >= remaining:
                raise InferenceTimeout(
                    f"no budget left for retry after {attempt + 1} attempts"
                ) from exc
            if on_retry is not None:
                on_retry(attempt + 1, delay, exc)
            sleep(delay)
    raise InferenceTimeout("inference retry budget exhausted") from last_error
