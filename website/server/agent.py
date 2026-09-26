"""Agent: a llama.cpp tool-calling loop over a local GGUF.

Kept deliberately small. The model is asked to answer a short spoken question,
calling at most a handful of tools. Everything it does is recorded so the UI
can show a trace and the tests can assert that a tool was genuinely called
rather than the answer being invented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .tools.registry import ToolRegistry, ToolResult

log = logging.getLogger("friday.agent")

SYSTEM_PROMPT = """You are FRIDAY, a voice assistant running on a local machine.

Rules:
- Your reply is read aloud. Keep it to one or two short sentences. No markdown, no lists, no emoji.
- For the current weather, ALWAYS call get_weather. Never guess a temperature.
- For arithmetic, ALWAYS call calculate. Never do the arithmetic yourself.
- If a tool fails, say so briefly instead of inventing an answer.
- If the request is unclear, ask one short clarifying question.
- Answer in the same language as the user."""

# Swapped in for the final round. The planning prompt above tells the model to
# reach for a tool, and a 3B model will happily print ``functions.getweather:``
# as plain text when the tool is taken away. This prompt closes that door.
SYNTHESIS_PROMPT = """You are FRIDAY, a voice assistant.

The tool results you need are already above. Now answer the user out loud.

Rules:
- Reply with plain spoken words only. One or two short sentences.
- Never write a tool name, function name, JSON, braces, or a colon.
- Use the exact numbers from the tool results; do not re-round them.
- Do not say you called a tool or that you are checking anything."""


@dataclass(slots=True)
class AgentReply:
    text: str
    tool_calls: list[ToolResult] = field(default_factory=list)
    elapsed_ms: int = 0
    rounds: int = 0
    error: str | None = None


class Agent:
    """Wraps a ``llama_cpp.Llama`` instance and drives the tool loop.

    Blocking by design: the pipeline calls it from a dedicated worker thread.
    Tool handlers are async, so the agent owns a small event loop used only
    from that thread.
    """

    def __init__(self, cfg: Config, registry: ToolRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self._llm = None
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.close()

    def _run_coro(self, coro: Any) -> Any:
        """Run a tool coroutine on the agent's private loop."""
        return self._loop.run_until_complete(coro)

    # ------------------------------------------------------------------ load
    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        path = self.cfg.llm_model_file
        if not path.exists():
            raise FileNotFoundError(
                f"LLM weights not found at {path}. Run: python scripts/fetch_models.py"
            )
        log.info("loading LLM %s (n_gpu_layers=%d)", path.name, self.cfg.llm.n_gpu_layers)
        started = time.monotonic()
        self._llm = Llama(
            model_path=str(path),
            n_ctx=self.cfg.llm.n_ctx,
            n_gpu_layers=self.cfg.llm.n_gpu_layers,
            n_threads=max(1, (os.cpu_count() or 4) - 2),
            verbose=False,
            # Without this the model ignores the tools entirely and does the
            # arithmetic itself (incorrectly).
            chat_format=self.cfg.llm.chat_format,
            # Bound generation so a runaway response cannot stall the pipeline.
            max_tokens=self.cfg.llm.max_tokens,
        )
        log.info(
            "LLM ready in %.1fs (chat_format=%s)", time.monotonic() - started, self.cfg.llm.chat_format
        )

    # ---------------------------------------------------------------- answer
    def answer(self, question: str) -> AgentReply:
        """Answer ``question``, calling tools as needed."""
        if not question.strip():
            return AgentReply(text="", error="empty question")

        try:
            self.load()
        except Exception as exc:  # noqa: BLE001
            log.error("LLM load failed: %s", exc)
            return AgentReply(text="", error=f"model unavailable: {exc}")

        started = time.monotonic()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        calls: list[ToolResult] = []
        tools = self.registry.schemas()
        answer = ""

        try:
            seen_calls: set[str] = set()
            for _round in range(1, self.cfg.llm.max_tool_rounds + 1):
                with self._lock:
                    completion = self._complete(messages, tools)
                message = completion["choices"][0]["message"]
                tool_calls = _extract_tool_calls(message)

                # A model that keeps re-issuing the same call is stuck, not
                # thinking. Stop planning and make it speak.
                fresh = [tc for tc in tool_calls if _signature(tc) not in seen_calls]
                if tool_calls and not fresh:
                    log.info("repeated tool call detected; forcing a spoken answer")
                    answer = self._speak(messages)
                    break

                if not tool_calls:
                    text = _clean_text(message.get("content") or "")
                    if not text or _is_call_stub(text):
                        # The model started a call it could not finish, or
                        # answered with punctuation. Ask for prose.
                        answer = self._speak(messages)
                    else:
                        answer = text
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": [
                            {
                                "id": tc.get("id") or f"call_{len(calls) + 1}",
                                "type": "function",
                                "function": {
                                    "name": tc.get("name") or "",
                                    "arguments": json.dumps(tc.get("arguments") or {}),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    name = tc.get("name") or ""
                    args = tc.get("arguments") or {}
                    if isinstance(args, str):
                        args = _safe_json(args)
                    seen_calls.add(_signature(tc))
                    log.info("tool call: %s(%s)", name, args)
                    result = self._run_coro(self.registry.invoke(name, args))
                    calls.append(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id") or f"call_{len(calls)}",
                            "name": name,
                            "content": json.dumps(
                                result.to_dict().get("result")
                                if not result.error
                                else {"error": result.error},
                                default=str,
                            ),
                        }
                    )
            else:
                # Ran out of tool rounds; ask for a plain answer anyway.
                answer = self._speak(messages)
        except Exception as exc:  # noqa: BLE001
            log.exception("agent failed")
            return AgentReply(
                text=answer,
                tool_calls=calls,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

        return AgentReply(
            text=answer,
            tool_calls=calls,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            rounds=len(calls),
        )

    def _speak(self, messages: list[dict[str, Any]]) -> str:
        """Force a prose answer with tool calling switched off.

        Qwen2.5 happily re-enters "call a tool" mode even when it already has
        every result it needs, and emits a bare ``functions.calculate:`` stub.
        Removing the tools from the request is what actually stops it, and
        swapping the system prompt stops it apologising about having no tools.
        """
        spoken = [
            {**m, "content": SYNTHESIS_PROMPT} if m["role"] == "system" else m
            for m in messages
        ]
        with self._lock:
            completion = self._complete(spoken, [], force_text=True)
        text = _clean_text(completion["choices"][0]["message"].get("content") or "")
        if _is_call_stub(text):
            # Last resort: strip any leaked call syntax out of the prose.
            text = _strip_call_syntax(text)
        return text

    def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_text: bool = False,
    ) -> dict[str, Any]:
        assert self._llm is not None
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": self.cfg.llm.temperature,
            "max_tokens": self.cfg.llm.max_tokens,
        }
        if tools and not force_text:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = "none"
        return self._llm.create_chat_completion(**kwargs)


# ------------------------------------------------------------------ helpers


def _signature(call: dict[str, Any]) -> str:
    """Identity of a tool call, used to detect the model going in circles."""
    args = call.get("arguments")
    if isinstance(args, dict):
        args = json.dumps(args, sort_keys=True)
    return f"{call.get('name') or ''}({args or ''})"


# Fragments that mean "the model tried to emit a call but gave up mid-way".
_STUB_RE = re.compile(
    r"^\s*(?:functions?\.[\w.]*\s*:|<\s*/?\s*tool|\{\s*\"|[\s:.,()\[\]{}<>\-]*)\s*$",
    re.IGNORECASE,
)


def _is_call_stub(text: str) -> bool:
    """True when ``text`` is a function-call remnant rather than an answer."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith(("functions.", "function.", "<tool", "</tool", "<|tool")):
        return True
    # Nothing speakable: punctuation and control characters only.
    return bool(_STUB_RE.match(stripped)) or not re.search(r"[A-Za-z0-9]", stripped)


# The call syntax a model leaks when it is mid-emission and gets cut off.
_CALL_LEAK_RE = re.compile(
    r"(?:functions?\.[\w.]*\s*:?\s*)|(?:<\s*/?\s*tool[^>]*>)|(?:\{\s*[\"']?[\w]*[\"']?\s*:?)|"
    r"(?:</?tool_call>)|(?:&3[49];)",
    re.IGNORECASE,
)
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _strip_call_syntax(text: str) -> str:
    """Salvage prose from a response that leaked tool-call formatting."""
    out = _CALL_LEAK_RE.sub(" ", text)
    out = _JSON_BLOCK_RE.sub(" ", out)
    out = out.replace("`", " ")
    out = re.sub(r"\s+", " ", out).strip(" :,.-")
    return out


def _extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise llama-cpp's tool_calls into a plain list of dicts."""
    raw = message.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for tc in raw:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _safe_json(args)
        out.append(
            {
                "id": tc.get("id"),
                "name": fn.get("name") or "",
                "arguments": args if isinstance(args, dict) else {},
            }
        )
    return out


def _safe_json(text: str) -> dict[str, Any]:
    # A dict is what most callers already have; round-tripping it through
    # json.loads would raise TypeError and silently discard the arguments.
    if isinstance(text, dict):
        return text
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, ValueError):
        return {}


_MARKDOWN = re.compile(r"[*_`#]+")
_LEAD_LABEL = re.compile(
    r"^\s*(?:answer|response|reply|friday)\s*[:\-]\s*", re.IGNORECASE
)


def _clean_text(text: str) -> str:
    """Make a model response safe to render and to read aloud."""
    s = (text or "").strip()
    s = _LEAD_LABEL.sub("", s)
    s = _MARKDOWN.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    # Strip a single trailing period duplication from quoted speech.
    return s
