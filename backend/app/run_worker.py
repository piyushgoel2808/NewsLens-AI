"""Cloud Run Celery Worker Entrypoint.

Starts a background HTTP health server on $PORT to satisfy Cloud Run service
readiness/liveness probes while running the Celery task consumer loop in the foreground.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/health", "/ready", "/live"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"up","service":"celery-worker"}')
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

    # 2. Build Celery worker command
    cmd = [
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

    # 3. Run Celery in foreground. If it exits, exit script so Cloud Run restarts instance.
    exit_code = subprocess.call(cmd)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
