"""Wire schemas shared by the pipeline, the REST API and the WebSocket feed."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Lifecycle of one turn. Mirrors the status pills rendered in the UI.
TurnState = Literal[
    "uploaded",
    "transcribing",
    "thinking",
    "calling",
    "done",
    "no_speech",
    "unclear",
    "error",
]

# Non-error terminal states.
OK_STATES = frozenset({"done"})


class ToolCall(BaseModel):
    """One tool invocation, recorded for display in the UI trace."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    duration_ms: int | None = None


class Turn(BaseModel):
    id: str
    seq: int
    created_at: str
    updated_at: str
    state: TurnState = "uploaded"
    # A relative URL, never the server's filesystem path.
    audio_url: str | None = None
    duration_s: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    source: str | None = None
    transcript: str | None = None
    confidence: float | None = None
    answer: str | None = None
    error: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    transcript_ms: int | None = None
    llm_ms: int | None = None


class TurnCreated(BaseModel):
    """Envelope pushed over the WebSocket."""

    event: Literal["turn.created", "turn.updated"] = "turn.created"
    turn: Turn


class ManualTurnRequest(BaseModel):
    """Text-only turn, so the UI is developable without any audio."""

    text: str = Field(min_length=1, max_length=2000)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ingest: str
    stt: str
    llm: str
    device_connected: bool
    turns: int


class StatsResponse(BaseModel):
    total: int
    done: int
    errors: int
    avg_transcript_ms: int | None = None
    avg_llm_ms: int | None = None
