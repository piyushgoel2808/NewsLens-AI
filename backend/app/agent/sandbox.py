"""AST safety verification and isolated execution sandbox for dynamically generated tools."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Forbidden Imports, Calls, and Attributes
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES: frozenset[str] = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "pty",
    "ctypes",
    "importlib",
    "pathlib",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "pickle",
    "marshal",
    "shelve",
    "builtins",
    "code",
    "codeop",
    "posix",
    "nt",
    "_thread",
    "threading",
    "multiprocessing",
    "signal",
    "selectors",
    "webbrowser",
})

FORBIDDEN_CALLS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "globals",
    "locals",
    "__import__",
    "setattr",
    "delattr",
    "exit",
    "quit",
})

FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
    "__builtins__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__class__",
    "__bases__",
    "__mro__",
})

FORBIDDEN_SQL_PATTERNS: tuple[str, ...] = (
    "DROP TABLE",
    "DROP DATABASE",
    "DROP SCHEMA",
    "TRUNCATE TABLE",
    "ALTER TABLE",
    "DELETE FROM",
    "INSERT INTO",
    "UPDATE ",
    "GRANT ",
    "REVOKE ",
    "CREATE USER",
    "DROP USER",
)


@dataclass
class ScanResult:
    """Result of AST static analysis."""

    is_safe: bool
    violations: list[str] = field(default_factory=list)


class ASTSafetyScanner:
    """Static AST analyzer verifying safety of generated Python code."""

    def __init__(
        self,
        forbidden_modules: frozenset[str] = FORBIDDEN_MODULES,
        forbidden_calls: frozenset[str] = FORBIDDEN_CALLS,
        forbidden_attrs: frozenset[str] = FORBIDDEN_ATTRIBUTES,
    ) -> None:
        self._forbidden_modules = forbidden_modules
        self._forbidden_calls = forbidden_calls
        self._forbidden_attrs = forbidden_attrs

    def scan(self, code: str) -> ScanResult:
        """Parse and scan Python source code for security violations."""
        violations: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ScanResult(is_safe=False, violations=[f"Syntax error on line {e.lineno}: {e.msg}"])

        has_analyze_func = False

        for node in ast.walk(tree):
            # Check function definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "analyze":
                    has_analyze_func = True

            # Check imports: import X, Y
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in self._forbidden_modules:
                        violations.append(f"Forbidden module import: '{alias.name}'")

            # Check from-imports: from X import Y
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in self._forbidden_modules:
                        violations.append(f"Forbidden module import: 'from {node.module}'")

            # Check function calls: func(...)
            elif isinstance(node, ast.Call):
                call_name = self._resolve_call_name(node.func)
                if call_name and call_name in self._forbidden_calls:
                    violations.append(f"Forbidden call: '{call_name}()'")

            # Check attribute access: obj.attr
            elif isinstance(node, ast.Attribute):
                if node.attr in self._forbidden_attrs:
                    violations.append(f"Forbidden attribute access: '.{node.attr}'")

            # Check string literals for destructive SQL statements
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                str_val = node.value.upper()
                for sql_kw in FORBIDDEN_SQL_PATTERNS:
                    if sql_kw in str_val:
                        violations.append(f"Destructive SQL pattern detected: '{sql_kw.strip()}'")

        if not has_analyze_func:
            violations.append("Missing required entrypoint function 'async def analyze(db, query, context)'")

        return ScanResult(is_safe=len(violations) == 0, violations=violations)

    @staticmethod
    def _resolve_call_name(node: ast.AST) -> str | None:
        """Resolve the string identifier of a function call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


# ---------------------------------------------------------------------------
# Sandbox Execution Result
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""

    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    execution_time_ms: int = 0
    raw_stdout: str = ""
    raw_stderr: str = ""


# ---------------------------------------------------------------------------
# Sandboxed Subprocess Executor
# ---------------------------------------------------------------------------


class SandboxedExecutor:
    """Executes validated Python code in an isolated subprocess with resource limits."""

    def __init__(
        self,
        db_url_readonly: str,
        timeout_seconds: int = 10,
        max_memory_mb: int = 256,
        runner_path: str | Path | None = None,
    ) -> None:
        self._db_url = db_url_readonly
        self._timeout_seconds = timeout_seconds
        self._max_memory_mb = max_memory_mb
        self._runner_path = Path(runner_path) if runner_path else Path(__file__).parent / "sandbox_runner.py"

    async def execute(
        self,
        code: str,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute Python code in child process with stdin JSON payload."""
        t_start = time.monotonic()
        payload = {
            "code": code,
            "query": query,
            "context": context or {},
            "db_url": self._db_url,
            "max_memory_mb": self._max_memory_mb,
        }
        input_bytes = json.dumps(payload).encode("utf-8")

        proc = None
        try:
            # Spawn isolated python interpreter
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self._runner_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=input_bytes),
                timeout=float(self._timeout_seconds),
            )

            dur_ms = round((time.monotonic() - t_start) * 1000)
            stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                err_msg = stderr_str or f"Process exited with code {proc.returncode}"
                # If stdout has structured error JSON, prefer that
                with contextlib_suppress():
                    parsed_err = json.loads(stdout_str)
                    if isinstance(parsed_err, dict) and parsed_err.get("error"):
                        err_msg = parsed_err["error"]
                return SandboxResult(
                    success=False,
                    error=err_msg,
                    execution_time_ms=dur_ms,
                    raw_stdout=stdout_str,
                    raw_stderr=stderr_str,
                )

            # Parse JSON output from stdout
            if not stdout_str:
                return SandboxResult(
                    success=False,
                    error="Execution produced no output.",
                    execution_time_ms=dur_ms,
                    raw_stderr=stderr_str,
                )

            try:
                result_json = json.loads(stdout_str)
            except json.JSONDecodeError as jde:
                return SandboxResult(
                    success=False,
                    error=f"Failed to parse JSON result: {jde}. Raw: {stdout_str[:200]}",
                    execution_time_ms=dur_ms,
                    raw_stdout=stdout_str,
                    raw_stderr=stderr_str,
                )

            if not isinstance(result_json, dict):
                return SandboxResult(
                    success=False,
                    error=f"Expected dictionary result from analyze(), got {type(result_json).__name__}",
                    execution_time_ms=dur_ms,
                    raw_stdout=stdout_str,
                )

            if result_json.get("success") is False:
                return SandboxResult(
                    success=False,
                    error=result_json.get("error") or "Dynamic analysis failed.",
                    execution_time_ms=dur_ms,
                    raw_stdout=stdout_str,
                    raw_stderr=stderr_str,
                )

            return SandboxResult(
                success=True,
                output=result_json.get("output", result_json),
                execution_time_ms=dur_ms,
                raw_stdout=stdout_str,
                raw_stderr=stderr_str,
            )

        except asyncio.TimeoutError:
            dur_ms = round((time.monotonic() - t_start) * 1000)
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            logger.warning(f"Sandboxed code execution timed out after {self._timeout_seconds}s")
            return SandboxResult(
                success=False,
                error=f"Execution timed out after {self._timeout_seconds} seconds.",
                execution_time_ms=dur_ms,
            )
        except Exception as ex:
            dur_ms = round((time.monotonic() - t_start) * 1000)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
            logger.error(f"Unexpected sandbox execution error: {ex}")
            return SandboxResult(
                success=False,
                error=f"Internal execution error: {ex}",
                execution_time_ms=dur_ms,
            )

    async def execute_in_process(
        self,
        func: Any,
        db_session: Any,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Direct in-process execution hook for fast unit testing with mocks."""
        t_start = time.monotonic()
        try:
            coro = func(db_session, query, context or {})
            res = await asyncio.wait_for(coro, timeout=float(self._timeout_seconds))
            dur_ms = round((time.monotonic() - t_start) * 1000)
            return SandboxResult(success=True, output=res, execution_time_ms=dur_ms)
        except asyncio.TimeoutError:
            dur_ms = round((time.monotonic() - t_start) * 1000)
            return SandboxResult(success=False, error="Execution timed out.", execution_time_ms=dur_ms)
        except Exception as ex:
            dur_ms = round((time.monotonic() - t_start) * 1000)
            return SandboxResult(success=False, error=str(ex), execution_time_ms=dur_ms)


class contextlib_suppress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


__all__ = [
    "ASTSafetyScanner",
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_CALLS",
    "FORBIDDEN_MODULES",
    "FORBIDDEN_SQL_PATTERNS",
    "SandboxResult",
    "SandboxedExecutor",
    "ScanResult",
]
