"""Tests for parse_thought_and_answer across reasoning and non-reasoning models."""

import pytest
from app.agent.synthesizer import parse_thought_and_answer


def test_standard_think_tags():
    raw = "<think>Analyzing user query...</think>\n### ⚡ Executive Summary\nHere is the answer."
    th, ans = parse_thought_and_answer(raw)
    assert th == "Analyzing user query..."
    assert ans == "### ⚡ Executive Summary\nHere is the answer."


def test_nemotron_plain_text_reasoning():
    raw = (
        "Here's a thinking process:\n\n"
        "1. Analyze User Input:\n"
        "- Query: Demolition in Vasco\n\n"
        "### ⚡ Executive Summary\n"
        "The building in Vasco was demolished on Thursday."
    )
    th, ans = parse_thought_and_answer(raw)
    assert "Here's a thinking process:" in th
    assert "Query: Demolition in Vasco" in th
    assert ans == "### ⚡ Executive Summary\nThe building in Vasco was demolished on Thursday."


def test_nemotron_with_draft_marker():
    raw = (
        "Thinking Process:\n"
        "1. Check evidence\n"
        "Let's draft:\n\n"
        "### ⚡ Executive Summary\n"
        "Demolition completed."
    )
    th, ans = parse_thought_and_answer(raw)
    assert "Thinking Process:" in th
    assert ans == "### ⚡ Executive Summary\nDemolition completed."


def test_truncated_thought_no_answer():
    raw = (
        "Here's a thinking process:\n"
        "1. Check dates\n"
        "Check: '2026-08-01', '2026-08-01', '2026-08-01'"
    )
    th, ans = parse_thought_and_answer(raw)
    assert th == raw.strip()
    # Critical: answer must NOT contain raw thought
    assert ans == ""


def test_plain_answer_without_thoughts():
    raw = "### ⚡ Executive Summary\nDirect answer without any scratchpad."
    th, ans = parse_thought_and_answer(raw)
    assert th == ""
    assert ans == raw
