"""Cloud Run Celery Worker Entrypoint.

Starts a background HTTP health server on $PORT to satisfy Cloud Run service
readiness/liveness probes while running the Celery task consumer loop in the foreground.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        clean_path = self.path.split("?")[0].rstrip("/")
        if clean_path in ("", "/health", "/ready", "/live", "/healthz", "/_health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = json.dumps({
                "status": "up",
                "service": "celery-worker",
                "pid": os.getpid(),
            }).encode("utf-8")
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Suppress noisy periodic Cloud Run health probe logs
        pass


def start_health_server() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


def main() -> None:
    # 1. Start HTTP health check thread for Cloud Run
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()

    # 2. Build Celery worker command using the active Python interpreter
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "app.ingestion.celery_app",
        "worker",
        "--loglevel=info",
        "--concurrency=2",
        "--max-tasks-per-child=10",
        "--time-limit=600",
        "--soft-time-limit=540",
    ]

    # If any CLI arguments are forwarded, append or override
    if len(sys.argv) > 1:
        cmd = sys.argv[1:]

    # 3. Spawn Celery worker and forward Cloud Run signals gracefully
    proc = subprocess.Popen(cmd)

    def _shutdown(signum: int, _frame: object) -> None:
        if proc.poll() is None:
            proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    exit_code = proc.wait()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

