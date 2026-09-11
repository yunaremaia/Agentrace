"""Parse Claude Code session transcripts into subagent runs.

Claude Code writes every session to ~/.claude/projects/<slug>/<session-id>.jsonl, one JSON object
per line. When the main thread delegates work, it emits a `tool_use` block named "Agent" carrying
the subagent's prompt; the subagent's answer comes back later as a `tool_result` with a matching
tool_use_id.

That pairing is all we need, and it is the whole reason this tool can exist without any
instrumentation: the data is already on disk. You do not have to remember to turn tracing on
before the run you wish you had traced.

Nothing here is guessed. The shapes below were read off a real 34MB session containing 96 Agent
invocations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


@dataclass
class AgentRun:
    """One delegation: what was asked, what came back, how long it took."""

    tool_use_id: str
    description: str
    prompt: str
    result: str
    started_at: datetime | None
    ended_at: datetime | None
    is_error: bool = False
    background: bool = False
    session: str = ""

    @property
    def duration_s(self) -> float | None:
        if self.started_at and self.ended_at:
            duration = (self.ended_at - self.started_at).total_seconds()
            return duration if duration >= 0 else None
        return None

    @property
    def prompt_chars(self) -> int:
        return len(self.prompt)

    @property
    def result_chars(self) -> int:
        return len(self.result)


@dataclass
class Session:
    session_id: str
    path: Path
    runs: list[AgentRun] = field(default_factory=list)


def default_transcript_dir() -> Path:
    return Path.home() / ".claude" / "projects"


def find_sessions(root: Path | None = None) -> list[Path]:
    """Every session transcript under the Claude Code projects dir."""
    root = root or default_transcript_dir()
    if not root.exists():
        return []
    return sorted(root.glob("**/*.jsonl"))


def _ts(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _flatten(content) -> str:
    """tool_result content is sometimes a string, sometimes a list of typed blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text") or json.dumps(c))
            else:
                parts.append(str(c))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _records(path: Path) -> Iterator[dict]:
    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A transcript being appended to while we read it can yield a torn final line.
                # Skipping is correct: the alternative is refusing to analyse a live session.
                continue


def parse_session(path: Path) -> Session:
    """Pull every Agent delegation out of one transcript.

    Two passes over the file rather than one: results can appear before we have seen every use in
    weird orderings, and a 34MB file is cheap to scan twice compared to getting this subtly wrong.
    """
    uses: dict[str, dict] = {}
    results: dict[str, dict] = {}

    for rec in _records(path):
        msg = rec.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use" and block.get("name") in ("Agent", "Task"):
                bid = block.get("id")
                if bid:
                    uses[bid] = {"input": block.get("input") or {}, "ts": rec.get("timestamp")}
            elif btype == "tool_result":
                tid = block.get("tool_use_id")
                if tid:
                    results[tid] = {
                        "content": block.get("content"),
                        "ts": rec.get("timestamp"),
                        "is_error": bool(block.get("is_error")),
                    }

    session_id = path.stem
    runs: list[AgentRun] = []
    for tid, use in uses.items():
        res = results.get(tid, {})
        inp = use["input"]
        runs.append(
            AgentRun(
                tool_use_id=tid,
                description=str(inp.get("description") or ""),
                prompt=str(inp.get("prompt") or ""),
                result=_flatten(res.get("content")),
                started_at=_ts(use.get("ts")),
                ended_at=_ts(res.get("ts")),
                is_error=res.get("is_error", False),
                background=bool(inp.get("run_in_background")),
                session=session_id,
            )
        )

    runs.sort(key=lambda r: (r.started_at or datetime.min.replace(tzinfo=None), r.tool_use_id))
    return Session(session_id=session_id, path=path, runs=runs)


def parse_all(root: Path | None = None) -> list[Session]:
    return [parse_session(p) for p in find_sessions(root)]
