"""Tests.

Two halves:
- parsing, against a synthetic transcript shaped exactly like a real one
- checks, each pinned to the real failure that motivated it

The check tests matter more than they look. A heuristic with no test drifts into either crying
wolf or saying nothing, and both make it worthless.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agentrace.checks import analyse
from agentrace.parse import AgentRun, parse_session


def _run(result: str = "x" * 500, prompt: str = "y" * 500, **kw) -> AgentRun:
    defaults = dict(
        tool_use_id="toolu_test123456",
        description="test run",
        prompt=prompt,
        result=result,
        started_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 7, 16, 12, 1, tzinfo=timezone.utc),
    )
    defaults.update(kw)
    return AgentRun(**defaults)


def _codes(run: AgentRun) -> set[str]:
    return {f.check for f in analyse(run)}


# --------------------------------------------------------------------------- parsing


def test_parses_agent_use_and_result(tmp_path):
    """The real transcript shape: tool_use named Agent, later a tool_result with matching id."""
    p = tmp_path / "sess.jsonl"
    lines = [
        {
            "type": "assistant",
            "timestamp": "2026-07-16T12:00:00.000Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "Agent",
                        "input": {
                            "description": "Find contacts",
                            "prompt": "Find verified emails. Report as a table.",
                            "run_in_background": True,
                        },
                    }
                ]
            },
        },
        {
            "type": "user",
            "timestamp": "2026-07-16T12:02:30.000Z",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": [{"type": "text", "text": "Found 3 verified emails."}],
                    }
                ]
            },
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines))

    s = parse_session(p)
    assert len(s.runs) == 1
    r = s.runs[0]
    assert r.description == "Find contacts"
    assert r.result == "Found 3 verified emails."
    assert r.background is True
    assert r.duration_s == 150.0


def test_string_content_is_handled(tmp_path):
    """tool_result content is sometimes a bare string rather than typed blocks."""
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "t1", "name": "Agent", "input": {"prompt": "go"}}
                        ]
                    }
                },
                {"message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "plain"}]}},
            ]
        )
    )
    assert parse_session(p).runs[0].result == "plain"


def test_torn_line_is_skipped_not_fatal(tmp_path):
    """A live session being appended to can yield a half-written final line.

    Refusing to parse would mean you cannot analyse a run until it is over, which is exactly when
    you most want to look.
    """
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps(
            {"message": {"content": [{"type": "tool_use", "id": "t1", "name": "Agent", "input": {}}]}}
        )
        + "\n{ this is not valid json"
    )
    s = parse_session(p)
    assert len(s.runs) == 1


def test_run_without_result_still_parses(tmp_path):
    """An agent still running has a use but no result. It should appear, not vanish."""
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps(
            {"message": {"content": [{"type": "tool_use", "id": "t1", "name": "Agent", "input": {"prompt": "p"}}]}}
        )
    )
    s = parse_session(p)
    assert len(s.runs) == 1
    assert s.runs[0].result == ""
    assert s.runs[0].duration_s is None


def test_non_agent_tools_are_ignored(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps({"message": {"content": [{"type": "tool_use", "id": "b1", "name": "Bash", "input": {}}]}})
    )
    assert parse_session(p).runs == []


def test_non_string_timestamp_does_not_crash(tmp_path):
    """A numeric timestamp (malformed transcript) should not crash _ts."""
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps({
            "type": "assistant",
            "timestamp": 1718000000,
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Agent", "input": {"prompt": "go"}}
                ]
            },
        })
        + "\n"
        + json.dumps({
            "type": "user",
            "timestamp": 1718000600,
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
                ]
            },
        })
    )
    s = parse_session(p)
    assert len(s.runs) == 1
    assert s.runs[0].started_at is None
    assert s.runs[0].ended_at is None


def test_tool_use_without_id_is_skipped(tmp_path):
    """A tool_use block missing an id should be skipped rather than crashing with KeyError."""
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps(
            {
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Agent", "input": {"prompt": "p"}}
                    ]
                }
            }
        )
    )
    s = parse_session(p)
    assert s.runs == []


# --------------------------------------------------------------------------- checks


def test_empty_result_is_high_severity():
    assert "empty_result" in _codes(_run(result=""))


def test_error_result_is_high_severity():
    assert "error" in _codes(_run(is_error=True))


def test_gave_up_is_caught():
    """'I was unable to find...' reads like an answer if you skim. It is not one."""
    assert "gave_up" in _codes(_run(result="I was unable to find any contact information. " + "x" * 200))


def test_absence_as_evidence_is_caught():
    """The real one: an agent said a company was not hiring because an API returned empty.

    That API returns empty with HTTP 200 for accounts that do not exist.
    """
    r = _run(result="Checked their board: no open roles. Concluding they are not hiring. " + "x" * 200)
    assert "absence_as_evidence" in _codes(r)


def test_hedged_claim_is_caught_but_only_low():
    """Hedging is honest. The bug is the hedge getting flattened into fact downstream, so this is
    a note, not an alarm."""
    r = _run(result="The company appears to be hiring engineers. " + "x" * 200)
    findings = [f for f in analyse(r) if f.check == "hedged_claim"]
    assert findings and findings[0].severity == "low"


def test_repeated_hedged_claims_are_counted():
    r = _run(result="Probably hiring. Probably expanding. Probably remote-friendly. " + "x" * 200)
    findings = [f for f in analyse(r) if f.check == "hedged_claim"]
    assert findings
    assert findings[0].message.startswith("3 hedged claim(s).")


def test_many_urls_without_verification_is_flagged():
    urls = " ".join(f"https://example{i}.com/careers" for i in range(8))
    assert "unverified_urls" in _codes(_run(result=urls + " " + "x" * 200))


def test_urls_with_verification_are_not_flagged():
    """Do not cry wolf: an agent that says it checked should not be nagged."""
    urls = " ".join(f"https://example{i}.com" for i in range(8))
    r = _run(result=urls + " All verified: each returned HTTP 200. " + "x" * 200)
    assert "unverified_urls" not in _codes(r)


def test_thin_prompt_is_flagged_as_your_fault():
    """The failure that is the operator's, not the model's."""
    assert "thin_prompt" in _codes(_run(prompt="find some stuff"))


def test_prompt_without_output_contract_is_flagged():
    r = _run(prompt="Go and research the market for AI agent tooling companies in depth. " * 5)
    assert "no_output_contract" in _codes(r)


def test_prompt_with_output_contract_is_not_flagged():
    r = _run(prompt="Research AI tooling companies. Write results to out.md as a table. " * 5)
    assert "no_output_contract" not in _codes(r)


def test_slow_run_is_flagged():
    r = _run(
        started_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
    )
    assert "slow_run" in _codes(r)


def test_clean_run_produces_nothing():
    """The most important test here. A checker that flags everything gets ignored."""
    r = _run(
        prompt="Find verified emails for these 5 companies. Report as a table with source URLs. " * 4,
        result="Verified all 5 against their legal pages, each returned HTTP 200. " + "Details. " * 40,
    )
    assert _codes(r) == set()
