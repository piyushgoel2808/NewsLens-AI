"""Unit tests for AST safety scanner and sandboxed code execution."""

import pytest
from app.agent.sandbox import ASTSafetyScanner, SandboxedExecutor, ScanResult


def test_ast_scanner_blocks_os_import():
    scanner = ASTSafetyScanner()
    code = """
import os

async def analyze(db, query, context):
    return {"summary": os.uname()[0], "data": []}
"""
    res: ScanResult = scanner.scan(code)
    assert not res.is_safe
    assert any("Forbidden module import: 'os'" in v for v in res.violations)


def test_ast_scanner_blocks_subprocess_and_sys():
    scanner = ASTSafetyScanner()
    code = """
from subprocess import Popen
import sys

async def analyze(db, query, context):
    return {"summary": "bad", "data": []}
"""
    res = scanner.scan(code)
    assert not res.is_safe
    assert any("subprocess" in v for v in res.violations)
    assert any("sys" in v for v in res.violations)


def test_ast_scanner_blocks_open_and_eval():
    scanner = ASTSafetyScanner()
    code = """
async def analyze(db, query, context):
    with open("/etc/passwd") as f:
        data = eval(f.read())
    return {"summary": data, "data": []}
"""
    res = scanner.scan(code)
    assert not res.is_safe
    assert any("Forbidden call: 'open()'" in v for v in res.violations)
    assert any("Forbidden call: 'eval()'" in v for v in res.violations)


def test_ast_scanner_blocks_dunder_subclasses_attr():
    scanner = ASTSafetyScanner()
    code = """
async def analyze(db, query, context):
    cls = ().__class__.__bases__[0].__subclasses__()
    return {"summary": str(cls), "data": []}
"""
    res = scanner.scan(code)
    assert not res.is_safe
    assert any("Forbidden attribute access: '.__subclasses__'" in v for v in res.violations)


def test_ast_scanner_blocks_destructive_sql():
    scanner = ASTSafetyScanner()
    code = """
from sqlalchemy import text

async def analyze(db, query, context):
    await db.execute(text("DROP TABLE articles"))
    return {"summary": "dropped", "data": []}
"""
    res = scanner.scan(code)
    assert not res.is_safe
    assert any("Destructive SQL pattern detected: 'DROP TABLE'" in v for v in res.violations)


def test_ast_scanner_requires_analyze_entrypoint():
    scanner = ASTSafetyScanner()
    code = """
import math

def calculate_something(x):
    return math.sqrt(x)
"""
    res = scanner.scan(code)
    assert not res.is_safe
    assert any("Missing required entrypoint function 'async def analyze" in v for v in res.violations)


def test_ast_scanner_allows_safe_code():
    scanner = ASTSafetyScanner()
    code = """
import math
import statistics
import pandas as pd
import numpy as np
from sqlalchemy import text

async def analyze(db, query, context):
    vals = [10, 20, 30, 40]
    mean_val = statistics.mean(vals)
    var_val = statistics.variance(vals)
    return {
        "summary": f"Calculated mean: {mean_val}, variance: {var_val}",
        "data": [{"val": v} for v in vals],
        "metadata": {"mean": mean_val, "variance": var_val}
    }
"""
    res = scanner.scan(code)
    assert res.is_safe
    assert len(res.violations) == 0


@pytest.mark.asyncio
async def test_sandbox_executes_safe_computation():
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)
    code = """
import statistics

async def analyze(db, query, context):
    data = [12, 18, 25, 42, 60]
    m = statistics.mean(data)
    v = statistics.variance(data)
    return {
        "summary": f"Mean is {m}, Variance is {v}",
        "data": [{"val": x} for x in data],
        "metadata": {"mean": m, "variance": v}
    }
"""
    result = await sandbox.execute(code=code, query="Calculate mean and variance")
    assert result.success is True
    assert result.output is not None
    assert "Mean is 31.4" in result.output["summary"]
    assert result.output["metadata"]["mean"] == 31.4


@pytest.mark.asyncio
async def test_sandbox_handles_syntax_error():
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)
    code = """
async def analyze(db, query, context):
    this is invalid python syntax %%%
"""
    result = await sandbox.execute(code=code, query="test")
    assert result.success is False
    assert "Syntax Error" in result.error or "SyntaxError" in result.error


@pytest.mark.asyncio
async def test_sandbox_handles_runtime_exception():
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=5)
    code = """
async def analyze(db, query, context):
    x = 10 / 0
    return {"summary": str(x), "data": []}
"""
    result = await sandbox.execute(code=code, query="test")
    assert result.success is False
    assert "ZeroDivisionError" in result.error


@pytest.mark.asyncio
async def test_sandbox_handles_timeout():
    # Set short timeout of 1 second
    sandbox = SandboxedExecutor(db_url_readonly="", timeout_seconds=1)
    code = """
import asyncio

async def analyze(db, query, context):
    await asyncio.sleep(5)
    return {"summary": "slept", "data": []}
"""
    result = await sandbox.execute(code=code, query="test")
    assert result.success is False
    assert "timed out" in result.error.lower()
