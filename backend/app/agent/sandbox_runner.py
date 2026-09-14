"""Isolated subprocess runner for sandboxed dynamic tool execution.

Invoked as: python -m app.agent.sandbox_runner or python sandbox_runner.py
Receives JSON payload on stdin, executes the analyze() function, and writes JSON to stdout.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import datetime
import json
import math
import re
import resource
import statistics
import sys
import traceback
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sqlalchemy import text
except ImportError:
    text = None


# ---------------------------------------------------------------------------
# Safe Built-ins & Import Interceptor
# ---------------------------------------------------------------------------

try:
    from app.agent.sandbox import FORBIDDEN_MODULES as BLOCKED_MODULES
except ImportError:
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from app.agent.sandbox import FORBIDDEN_MODULES as BLOCKED_MODULES

_orig_import = builtins.__import__


def _restricted_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = name.split(".")[0]
    if root in BLOCKED_MODULES:
        raise ImportError(f"Import of forbidden module '{name}' is blocked by sandbox security policy.")
    return _orig_import(name, globals, locals, fromlist, level)


# Safe builtins dictionary
SAFE_BUILTINS = dict(builtins.__dict__)
SAFE_BUILTINS["__import__"] = _restricted_import
SAFE_BUILTINS.pop("open", None)
SAFE_BUILTINS.pop("eval", None)
SAFE_BUILTINS.pop("exec", None)
SAFE_BUILTINS.pop("compile", None)
SAFE_BUILTINS.pop("exit", None)
SAFE_BUILTINS.pop("quit", None)


async def main() -> None:
    # 1. Read input payload from stdin
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            print(json.dumps({"success": False, "error": "No input payload received on stdin."}))
            return
        payload = json.loads(raw_input)
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Failed to parse runner stdin: {e}"}))
        return

    code = payload.get("code", "")
    query = payload.get("query", "")
    context = payload.get("context", {})
    db_url = payload.get("db_url", "")
    max_memory_mb = payload.get("max_memory_mb", 256)

    # 2. Enforce memory limits if supported
    try:
        if hasattr(resource, "RLIMIT_AS"):
            mem_bytes = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass  # macOS may restrict RLIMIT_AS adjustments

    # 3. Compile and execute the user code in restricted scope
    exec_globals: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__dynamic_tool__",
        "re": re,
        "math": math,
        "statistics": statistics,
        "json": json,
        "datetime": datetime,
    }
    if np is not None:
        exec_globals["np"] = np
        exec_globals["numpy"] = np
    if pd is not None:
        exec_globals["pd"] = pd
        exec_globals["pandas"] = pd
    if text is not None:
        exec_globals["text"] = text

    try:
        compiled = compile(code, "<dynamic_tool>", "exec")
        # Use Python built-in exec in runner (which is restricted by SAFE_BUILTINS)
        builtins.exec(compiled, exec_globals)
    except Exception as e:
        err_out = {
            "success": False,
            "error": f"Compilation/Syntax Error: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(err_out))
        return

    analyze_fn = exec_globals.get("analyze")
    if not analyze_fn or not callable(analyze_fn):
        print(json.dumps({"success": False, "error": "Entrypoint function 'async def analyze(db, query, context)' was not found."}))
        return

    # 4. Manage database session
    db_session = None
    engine = None
    session_maker = None

    if db_url:
        try:
            from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
            engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
            session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        except Exception as e:
            # If DB engine fails to initialize, notify
            pass

    # 5. Execute analyze function
    try:
        if session_maker:
            async with session_maker() as session:
                if asyncio.iscoroutinefunction(analyze_fn):
                    result = await analyze_fn(session, query, context)
                else:
                    result = analyze_fn(session, query, context)
        else:
            # Execute without DB session (or with dummy session if DB was not configured)
            if asyncio.iscoroutinefunction(analyze_fn):
                result = await analyze_fn(None, query, context)
            else:
                result = analyze_fn(None, query, context)

        # 6. Normalize and serialize result
        if not isinstance(result, dict):
            result = {"summary": str(result), "data": [], "metadata": {}}

        print(json.dumps({"success": True, "output": result}, default=str))

    except Exception as e:
        err_out = {
            "success": False,
            "error": f"Runtime Error: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(err_out))
    finally:
        if engine:
            with contextlib.suppress(Exception):
                await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
