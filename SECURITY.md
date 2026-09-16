# Security Policy

## 🔒 Supported Versions

Only the latest `main` branch receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

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

## 🛡️ Best Practices for Deployments

- **Never commit `.env` or cloud service keys**: `.env` and `service-account*.json` are strictly ignored in `.gitignore`.
- **Production Secrets**: Ensure `APP_SECRET_KEY` is set to a secure, randomly generated 64-character string in production environments.
- **Debug Mode**: Never set `APP_DEBUG=true` in public or production environments.
- **CORS Protection**: Configure `CORS_ALLOWED_ORIGINS` to specify only trusted front-end domains.
