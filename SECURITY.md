# Security Policy

## 🔒 Supported Versions

Only the latest `main` branch receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability or sensitive secret disclosure in NewsLens-AI, **please do NOT report it in public GitHub issues or discussions**.

Instead, please report security vulnerabilities responsibly:
1. Open a **Private Security Advisory** on GitHub under `Security` > `Advisories` > `Report a vulnerability`.
2. Or contact the maintainers directly via email with details, reproduction steps, and potential remediation.

Please include:
- A description of the issue and its potential impact.
- Affected components or API endpoints.
- Proof of Concept (PoC) scripts or reproduction steps.

We will acknowledge receipt within 48 hours and work with you on a coordinated disclosure timeline.

---

## 🛡️ Security Architecture & Hardening Controls (v0.6.0)

1. **API Key & Bearer Token Authentication (`backend/app/core/security.py`)**:
   - All mutating and administrative endpoints (`POST /api/ingest/*`, `PUT/POST /api/settings/*`, `POST /api/newspapers`, `PUT/DELETE /api/newspapers/{id}`, `PATCH/DELETE /api/issues/{id}`, `POST .../reingest`) are protected by `require_api_auth`.
   - Validates `X-API-Key` or `Authorization: Bearer <token>` using constant-time `secrets.compare_digest` against `API_AUTH_KEY`.
   - In production (`APP_DEBUG=false`), mutating endpoints fail closed (`HTTP 403`) if `API_AUTH_KEY` is unconfigured.
2. **Distributed Sliding-Window Rate Limiting (`RateLimitMiddleware`)**:
   - Protects high-cost LLM and OCR endpoints (`/api/query*`, `/api/timeline*`, `/api/ingest*`) with a Redis-backed atomic `INCR` + `EXPIRE` 60-second sliding window (`RATE_LIMIT_RPM=60`) and automatic in-memory fallback when Redis is unavailable.
   - Prevents `X-Forwarded-For` IP spoofing unless `TRUST_FORWARDED_FOR=true` is explicitly enabled behind a trusted reverse proxy.
3. **Sovereign Laptop Node PIN Brute-Force Lockout (`PinAttemptTracker`)**:
   - `POST /api/settings/laptop-node/toggle` enforces constant-time PIN comparison (`secrets.compare_digest`) and locks out offending client IPs for 300 seconds (`HTTP 429`) after 5 failed attempts within 5 minutes.
4. **Multi-Layer Subprocess AST Sandbox (`backend/app/agent/sandbox.py`, `sandbox_runner.py`)**:
   - Static AST folding (`_fold_string_concat`) detects and blocks string-concatenated dunder bypasses (e.g. `getattr(obj, "__sub" + "classes__")`).
   - Runtime `_safe_getattr` wrapper blocks dynamic access to any attribute starting with `_`.
   - Enforces MySQL engine-level read-only isolation (`SET SESSION TRANSACTION READ ONLY`) and wraps the database session in `_ReadOnlySessionProxy` to block `.commit()`, `.flush()`, and `.close()`, paired with an unconditional `finally: session.rollback()`.
5. **Docker Build Context Secret Isolation (`.dockerignore`)**:
   - Root, `backend/.dockerignore`, and `frontend/.dockerignore` strictly exclude `.env*`, `service-account*.json`, `.git`, virtual environments, and test caches from container build contexts.

---

## 🛡️ Best Practices for Deployments

- **Never commit `.env` or cloud service keys**: `.env` and `service-account*.json` are strictly ignored in `.gitignore` and `.dockerignore`.
- **Production Secrets**: Always set `APP_SECRET_KEY` and `API_AUTH_KEY` to cryptographically random 64-character strings in production environments.
- **Debug Mode**: Never set `APP_DEBUG=true` in public or production environments.
- **Reverse Proxy Headers**: Only set `TRUST_FORWARDED_FOR=true` when deployed behind a trusted ingress (such as Google Cloud Run or Nginx).
- **CORS Protection**: Configure `CORS_ALLOWED_ORIGINS` to specify only trusted front-end domains.
