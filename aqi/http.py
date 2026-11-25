"""A small HTTP helper: polite throttling and retries for free public APIs."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status = status
        self.url = url


class Http:
    """GET JSON with a minimum gap between calls and retries on 429/5xx.

    `session` only needs a requests-style `.get()`; tests and the offline demo pass
    a fake one so the real parsing code runs without network access.
    """

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        min_interval: float = 0.0,
        retries: int = 4,
        timeout: float = 60,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.headers = {"User-Agent": "aqi-forecast (github.com)", **(headers or {})}
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.sleep = sleep
        self._last_call = float("-inf")

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(self.retries + 1):
            gap = self.min_interval - (time.monotonic() - self._last_call)
            if gap > 0:
                self.sleep(gap)
            self._last_call = time.monotonic()

            try:
                resp = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                delay = 3 * 2**attempt
                log.warning("Network error on %s (%s); retrying in %ss", url, exc, delay)
                self.sleep(delay)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.retries:
                    raise HttpError(resp.status_code, url, resp.text)
                delay = _retry_delay(resp.headers, attempt)
                log.warning("HTTP %s from %s; retrying in %.0fs", resp.status_code, url, delay)
                self.sleep(delay)
                continue
            if resp.status_code >= 400:
                raise HttpError(resp.status_code, url, resp.text)
            return resp.json()
        raise AssertionError("unreachable")


def _retry_delay(headers: Any, attempt: int) -> float:
    """Honour Retry-After or OpenAQ's x-ratelimit-reset (seconds), else back off."""
    for name in ("retry-after", "x-ratelimit-reset"):
        value = headers.get(name) if headers else None
        try:
            if value is not None and float(value) >= 0:
                return min(float(value) + 1, 90)
        except (TypeError, ValueError):
            pass
    return min(5 * 2**attempt, 90)
