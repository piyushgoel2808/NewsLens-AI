"""Security, API Authentication, PIN Brute-Force Lockout, and Distributed Rate Limiting."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


def get_client_ip(request: Request) -> str:
    """Resolve client IP address, trusting X-Forwarded-For only when configured."""
    settings = get_settings()
    if getattr(settings, "trust_forwarded_for", False):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            first_ip = forwarded.split(",")[0].strip()
            if first_ip:
                return first_ip
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


async def require_api_auth(
    api_key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> str:
    """Validate X-API-Key or Authorization Bearer token when API_AUTH_KEY is configured."""
    settings = get_settings()
    expected_key = (getattr(settings, "api_auth_key", None) or "").strip()

    # When API_AUTH_KEY is not configured, allow access seamlessly for local dev / test suites
    if not expected_key:
        return "anonymous-dev"

    candidate = (api_key or (bearer.credentials if bearer else "")).strip()
    if not candidate or not hmac.compare_digest(
        candidate.encode("utf-8"),
        expected_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "authenticated"


class PinAttemptTracker:
    """Brute-force protection for Sovereign Laptop Node Admin PIN verification."""

    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_SECONDS = 900  # 15 minutes

    def __init__(self) -> None:
        self._failed_attempts: dict[str, list[float]] = defaultdict(list)

    def _prune(self, ip: str, now: float) -> None:
        cutoff = now - self.LOCKOUT_SECONDS
        self._failed_attempts[ip] = [t for t in self._failed_attempts[ip] if t > cutoff]
        if not self._failed_attempts[ip]:
            self._failed_attempts.pop(ip, None)

    def is_locked_out(self, ip: str) -> bool:
        now = time.monotonic()
        self._prune(ip, now)
        return len(self._failed_attempts.get(ip, [])) >= self.MAX_FAILED_ATTEMPTS

    def record_failure(self, ip: str) -> int:
        now = time.monotonic()
        self._prune(ip, now)
        self._failed_attempts[ip].append(now)
        return len(self._failed_attempts[ip])

    def reset(self, ip: str) -> None:
        self._failed_attempts.pop(ip, None)


pin_attempt_tracker = PinAttemptTracker()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Distributed Redis-backed (with bounded in-memory fallback) rate limiter middleware."""

    EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    MAX_IN_MEMORY_IPS = 4096

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._buckets: dict[str, list[float]] = {}
        self._redis_client: Any = None
        self._redis_init_attempted: bool = False

    async def _get_redis(self) -> Any:
        if not self._redis_init_attempted:
            self._redis_init_attempted = True
            try:
                import redis.asyncio as aioredis

                settings = get_settings()
                self._redis_client = aioredis.Redis.from_url(
                    settings.redis_url,
                    socket_connect_timeout=0.25,
                    socket_timeout=0.25,
                    decode_responses=True,
                )
            except Exception:
                self._redis_client = None
        return self._redis_client

    async def _check_rate_limit(self, ip: str, limit_rpm: int) -> tuple[bool, int]:
        """Return (allowed, current_count) for the 60-second window."""
        if limit_rpm <= 0:
            return True, 0

        window_bucket = int(time.time() // 60)
        redis_key = f"newslens:ratelimit:{ip}:{window_bucket}"

        client = await self._get_redis()
        if client is not None:
            try:
                count = int(await client.incr(redis_key))
                if count == 1:
                    await client.expire(redis_key, 65)
                return count <= limit_rpm, count
            except Exception:
                pass

        # Fallback: bounded in-memory sliding window
        now = time.monotonic()
        cutoff = now - 60.0
        if len(self._buckets) >= self.MAX_IN_MEMORY_IPS and ip not in self._buckets:
            # Evict oldest entry to prevent memory exhaustion under IP spoofing
            oldest_ip = next(iter(self._buckets))
            self._buckets.pop(oldest_ip, None)

        timestamps = [t for t in self._buckets.get(ip, []) if t > cutoff]
        timestamps.append(now)
        self._buckets[ip] = timestamps
        return len(timestamps) <= limit_rpm, len(timestamps)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        settings = get_settings()
        limit_rpm = getattr(settings, "rate_limit_rpm", 120)
        client_ip = get_client_ip(request)

        allowed, _count = await self._check_rate_limit(client_ip, limit_rpm)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please retry in a minute."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
