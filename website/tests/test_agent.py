"""The agent's tool-calling loop.

Two layers. The loop itself is driven by a fake ``Llama`` so the control flow
(single call, repeated call, leaked call syntax, tool failure, runaway) is tested
deterministically. On top of that, a slow suite runs the real GGUF to prove the
model actually picks the right tool and does not hallucinate the answer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from server.agent import (
    Agent,
    AgentReply,
    _clean_text,
    _extract_tool_calls,
    _is_call_stub,
    _safe_json,
    _signature,
    _strip_call_syntax,
)
from server.config import Config
from server.tools import build_registry
from server.tools.registry import ToolRegistry
from server.tools.weather import WeatherClient

from .conftest import needs_llm

# ------------------------------------------------------------------ fakes


class FakeLlama:
    """A scripted ``create_chat_completion``.

    ``script`` is a list of responses returned in order; the last entry repeats.
    Each entry is either a plain string (prose) or a list of ``(name, args)``
    pairs (tool calls).
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.requests: list[dict[str, Any]] = []
        self._index = 0

    def _next(self) -> Any:
        entry = self.script[min(self._index, len(self.script) - 1)]
        self._index += 1
        return entry

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        entry = self._next()
        if isinstance(entry, str):
            message: dict[str, Any] = {"role": "assistant", "content": entry}
        else:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{self._index}_{i}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        },
                    }
                    for i, (name, args) in enumerate(entry)
                ],
            }
        return {"choices": [{"message": message, "finish_reason": "stop"}]}


class FakeWeather:
    """Stands in for the network so no test depends on Open-Meteo."""

    def __init__(self, temperature: float = 28.4, fail: bool = False) -> None:
        self.temperature = temperature
        self.fail = fail
        self.calls = 0

    async def current(self, city: str, unit: str = "celsius") -> dict[str, Any]:
        self.calls += 1
        from server.tools.weather import WeatherError

        if self.fail:
            raise WeatherError("the weather service is unreachable")
        return {
            "location": {"name": city, "country": "India", "timezone": "Asia/Kolkata"},
            "current": {
                "temperature": self.temperature,
                "condition": "partly cloudy",
                "humidity": 62,
                "feels_like": 30.0,
                "wind_speed": 11.0,
            },
            "today": {"temp_max": 31.0, "temp_min": 20.0, "precipitation_probability": 30},
            "unit": unit,
            "cached": False,
        }

    async def aclose(self) -> None:
        pass


def make_agent(cfg: Config, script: list[Any], weather: FakeWeather | None = None):
    weather = weather or FakeWeather()
    registry = build_registry(cfg, weather)  # type: ignore[arg-type]
    agent = Agent(cfg, registry)
    llama = FakeLlama(script)
    agent._llm = llama  # injected so load() is a no-op
    try:
        yield agent, llama, registry
    finally:
        agent.close()


def tool_names(reply: AgentReply) -> list[str]:
    return [c.name for c in reply.tool_calls]


# ----------------------------------------------------------------- helpers


class TestCallStubs:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "functions.get_weather:",
            "functions.calculate:",
            "function.get_weather",
            "<tool_call>",
            "</tool_call>",
            "<|tool_call|>",
            ":::",
            "...",
            "{}",
            "-",
        ],
    )
    def test_detected(self, text: str) -> None:
        assert _is_call_stub(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The weather in Pune is 28 degrees.",
            "161",
            "I could not find that city.",
            "",
        ][:-1],
    )
    def test_not_detected(self, text: str) -> None:
        assert _is_call_stub(text) is False


class TestStripCallSyntax:
    def test_removes_a_leaked_function_prefix(self) -> None:
        assert "161" in _strip_call_syntax("functions.calculate: 161")

    def test_removes_a_leaked_json_block(self) -> None:
        out = _strip_call_syntax('The answer is {"expression": "7 * 23", "result": "161"}.')
        assert "expression" not in out
        assert "161" in out

    def test_removes_markdown_backticks(self) -> None:
        assert "`161`" not in _strip_call_syntax("`161`")

    def test_leaves_plain_prose_alone(self) -> None:
        assert _strip_call_syntax("The weather is nice").startswith("The weather is nice")

    def test_a_bare_stub_yields_nothing(self) -> None:
        """A stub with no prose in it has nothing to salvage, and says so."""
        assert _strip_call_syntax("functions.get_weather:") == ""

    def test_prose_around_a_stub_is_kept(self) -> None:
        out = _strip_call_syntax("functions.get_weather: It is 28 degrees in Pune.")
        assert "It is 28 degrees in Pune" in out
        assert "functions" not in out


class TestCleanText:
    def test_strips_markdown(self) -> None:
        assert _clean_text("**161**") == "161"
        assert _clean_text("`code`") == "code"

    def test_strips_a_leading_label(self) -> None:
        assert _clean_text("Answer: 161") == "161"
        assert _clean_text("FRIDAY: hello") == "hello"

    def test_collapses_whitespace(self) -> None:
        assert _clean_text("a   b\n\n\nc") == "a b\nc"

    def test_empty_input(self) -> None:
        assert _clean_text(None) == ""  # type: ignore[arg-type]


class TestToolCallParsing:
    def test_extracts_and_parses_string_arguments(self) -> None:
        message = {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "calculate", "arguments": '{"expression": "7 * 23"}'},
                }
            ]
        }
        got = _extract_tool_calls(message)
        assert got == [{"id": "c1", "name": "calculate", "arguments": {"expression": "7 * 23"}}]

    def test_malformed_arguments_become_empty(self) -> None:
        message = {"tool_calls": [{"id": "c1", "function": {"name": "x", "arguments": "{oops"}}]}
        assert _extract_tool_calls(message)[0]["arguments"] == {}

    def test_missing_name_does_not_raise(self) -> None:
        assert _extract_tool_calls({"tool_calls": [{}]})[0]["name"] == ""

    def test_no_tool_calls(self) -> None:
        assert _extract_tool_calls({}) == []
        assert _extract_tool_calls({"tool_calls": None}) == []

    def test_signature_is_order_independent_for_dicts(self) -> None:
        a = {"name": "f", "arguments": {"a": 1, "b": 2}}
        b = {"name": "f", "arguments": {"b": 2, "a": 1}}
        assert _signature(a) == _signature(b)

    def test_signature_differs_by_arguments(self) -> None:
        assert _signature({"name": "f", "arguments": {"x": 1}}) != _signature(
            {"name": "f", "arguments": {"x": 2}}
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"a":1}', {"a": 1}),
            ({"a": 1}, {"a": 1}),
            (None, {}),
            ("not json", {}),
        ],
    )
    def test_safe_json(self, raw, expected) -> None:
        assert _safe_json(raw) == expected

    def test_safe_json_on_a_bare_list(self) -> None:
        assert _safe_json("[1, 2]") == {"value": [1, 2]}


# ------------------------------------------------------------ the loop


class TestLoop:
    def test_a_single_tool_call_then_an_answer(self, cfg: Config) -> None:
        script = [
            [("calculate", {"expression": "7 * 23"})],
            "The answer is 161.",
        ]
        for agent, _, _ in make_agent(cfg, script):
            reply = agent.answer("what is 7 times 23?")
            assert tool_names(reply) == ["calculate"]
            assert reply.text == "The answer is 161."
            assert reply.rounds == 1
            assert not reply.error

    def test_the_tool_really_ran(self, cfg: Config) -> None:
        script = [[("calculate", {"expression": "144 / 12"})], "12."]
        for agent, _, _ in make_agent(cfg, script):
            reply = agent.answer("144 divided by 12")
            call = reply.tool_calls[0]
            assert call.result == {"expression": "144 / 12", "result": "12"}
            assert call.error is None
            assert call.duration_ms is not None

    def test_a_direct_answer_needs_no_tool(self, cfg: Config) -> None:
        for agent, _, _ in make_agent(cfg, ["Hello there, how can I help?"]):
            reply = agent.answer("hello")
            assert reply.tool_calls == []
            assert "help" in reply.text
            assert reply.rounds == 0

    def test_two_tools_in_one_round(self, cfg: Config) -> None:
        script = [
            [("get_datetime", {}), ("calculate", {"expression": "1 + 1"})],
            "It is noon and two.",
        ]
        for agent, _, _ in make_agent(cfg, script):
            reply = agent.answer("what time is it and what is 1 plus 1?")
            assert tool_names(reply) == ["get_datetime", "calculate"]

    def test_tool_result_is_fed_back_to_the_model(self, cfg: Config) -> None:
        """The second completion must contain the first round's result."""
        script = [[("calculate", {"expression": "7 * 23"})], "161."]
        for agent, llama, _ in make_agent(cfg, script):
            agent.answer("what is 7 times 23?")
            second = llama.requests[1]["messages"]
            tool_messages = [m for m in second if m["role"] == "tool"]
            assert len(tool_messages) == 1
            assert "161" in tool_messages[0]["content"]


class TestLoopGuards:
    def test_repeated_identical_call_is_cut_off(self, cfg: Config) -> None:
        """A model that keeps re-issuing the same call must still get spoken to."""
        script = [
            [("calculate", {"expression": "7 * 23"})],
            [("calculate", {"expression": "7 * 23"})],
            "The answer is 161.",
        ]
        for agent, _, _ in make_agent(cfg, script):
            reply = agent.answer("what is 7 times 23?")
            assert tool_names(reply) == ["calculate"]  # not two
            assert reply.text

    def test_a_leaked_call_stub_forces_a_spoken_answer(self, cfg: Config) -> None:
        script = ["functions.get_weather:", "It is 28 degrees in Pune."]
        for agent, llama, _ in make_agent(cfg, script):
            reply = agent.answer("what is the weather in Pune?")
            assert reply.text == "It is 28 degrees in Pune."
            # The synthesis round must not offer tools.
            assert "tools" not in llama.requests[-1]

    def test_the_synthesis_round_swaps_the_system_prompt(self, cfg: Config) -> None:
        """Keeping the tool-eager prompt is what makes the model print call syntax."""
        from server.agent import SYNTHESIS_PROMPT, SYSTEM_PROMPT

        script = ["functions.calculate:", "161."]
        for agent, llama, _ in make_agent(cfg, script):
            agent.answer("what is 7 times 23?")
            systems = [m["content"] for m in llama.requests[-1]["messages"] if m["role"] == "system"]
            assert systems == [SYNTHESIS_PROMPT]
            assert systems[0] != SYSTEM_PROMPT

    def test_punctuation_only_answer_forces_a_spoken_answer(self, cfg: Config) -> None:
        script = ["...", "Sorry, could you say that again?"]
        for agent, _, _ in make_agent(cfg, script):
            reply = agent.answer("mumble")
            assert reply.text == "Sorry, could you say that again?"

    def test_tool_round_budget_is_respected(self, cfg: Config) -> None:
        cfg.llm.max_tool_rounds = 2
        script = [[("calculate", {"expression": "1 + 1"})], "done"]
        for agent, _, _ in make_agent(cfg, script):
            reply = agent.answer("what is 1 plus 1")
            assert reply.text

    def test_an_unknown_tool_becomes_a_visible_error(self, cfg: Config) -> None:
        script = [[("teleport", {"to": "mars"})], "I cannot do that."]
        for agent, _, _ in make_agent(cfg, script):
            reply = agent.answer("teleport me to mars")
            assert reply.tool_calls[0].error is not None
            assert "no such tool" in reply.tool_calls[0].error
            assert reply.text == "I cannot do that."

    def test_a_failing_tool_still_yields_a_spoken_answer(self, cfg: Config) -> None:
        weather = FakeWeather(fail=True)
        script = [
            [("get_weather", {"city": "Atlantis"})],
            "I could not reach the weather service.",
        ]
        for agent, _, _ in make_agent(cfg, script, weather):
            reply = agent.answer("what is the weather in Atlantis?")
            assert reply.tool_calls[0].error
            assert "could not reach" in reply.text

    def test_the_weather_tool_really_fetches(self, cfg: Config) -> None:
        weather = FakeWeather(temperature=31.0)
        script = [[("get_weather", {"city": "Pune"})], "It is 31 degrees in Pune."]
        for agent, _, _ in make_agent(cfg, script, weather):
            reply = agent.answer("what is the weather in Pune?")
            assert weather.calls == 1
            assert reply.tool_calls[0].result["current"]["temperature"] == 31.0

    def test_an_empty_question_is_refused_without_a_model_call(self, cfg: Config) -> None:
        for agent, llama, _ in make_agent(cfg, ["unused"]):
            reply = agent.answer("   ")
            assert reply.error == "empty question"
            assert llama.requests == []


class TestLoopCrashes:
    def test_a_generation_crash_becomes_an_error_reply(self, cfg: Config) -> None:
        class Exploding(FakeLlama):
            def create_chat_completion(self, **kwargs):
                raise RuntimeError("CUDA out of memory")

        weather = FakeWeather()
        agent = Agent(cfg, build_registry(cfg, weather))
        agent._llm = Exploding(["x"])
        try:
            reply = agent.answer("hello")
            assert reply.error
            assert "CUDA out of memory" in reply.error
        finally:
            agent.close()

    def test_a_load_failure_is_reported_not_raised(self, cfg: Config, monkeypatch) -> None:
        agent = Agent(cfg, ToolRegistry())
        try:
            def boom() -> None:
                raise FileNotFoundError("weights not found")

            monkeypatch.setattr(agent, "load", boom)
            reply = agent.answer("hello")
            assert reply.text == ""
            assert "model unavailable" in (reply.error or "")
        finally:
            agent.close()


# ---------------------------------------------------------------- registry


class TestRegistryIntegration:
    def test_every_tool_is_exposed_with_a_valid_schema(self, cfg: Config) -> None:
        """A malformed schema makes llama-cpp silently ignore the whole tool list."""
        registry = build_registry(cfg, FakeWeather())  # type: ignore[arg-type]
        names = registry.names()
        assert names == ["calculate", "get_datetime", "get_weather", "system_status"]
        for schema in registry.schemas():
            fn = schema["function"]
            assert schema["type"] == "function"
            assert fn["name"] and fn["description"]
            params = fn["parameters"]
            assert params["type"] == "object"
            assert isinstance(params.get("properties"), dict)

    def test_invoke_returns_a_result_not_an_exception(self, cfg: Config) -> None:
        import asyncio

        registry = build_registry(cfg, FakeWeather())  # type: ignore[arg-type]

        async def run() -> None:
            ok = await registry.invoke("calculate", {"expression": "7 * 23"})
            assert ok.result == {"expression": "7 * 23", "result": "161"}
            bad = await registry.invoke("calculate", {"expression": "import os"})
            assert bad.error
            missing = await registry.invoke("nope", {})
            assert "no such tool" in (missing.error or "")

        asyncio.run(run())

    def test_the_calculate_result_hides_the_float(self, cfg: Config) -> None:
        """If a raw 12.0 leaks out, the model reads "12.0" out loud."""
        import asyncio

        registry = build_registry(cfg, FakeWeather())  # type: ignore[arg-type]
        got = asyncio.run(registry.invoke("calculate", {"expression": "144 / 12"}))
        assert got.result == {"expression": "144 / 12", "result": "12"}
        assert "value" not in got.result


# ------------------------------------------------------------- the real model


@pytest.mark.slow
@pytest.mark.timeout(300)  # first call loads the 3B model onto the GPU
@needs_llm
class TestRealModel:
    """Proves the 3B model picks the right tool and does not do the maths itself."""

    @pytest.fixture(scope="class")
    def agent(self, session_cfg: Config):
        weather = WeatherClient()
        a = Agent(session_cfg, build_registry(session_cfg, weather))
        a.load()
        try:
            yield a
        finally:
            a.close()

    def test_arithmetic_goes_through_the_tool(self, agent: Agent) -> None:
        reply = agent.answer("what is 7 times 23?")
        assert tool_names(reply) == ["calculate"]
        assert reply.tool_calls[0].result == {"expression": "7 * 23", "result": "161"}
        assert "161" in reply.text

    def test_weather_goes_through_the_tool(self, agent: Agent) -> None:
        reply = agent.answer("what is the weather in Pune right now?")
        assert "get_weather" in tool_names(reply)
        assert reply.tool_calls[0].error is None
        assert reply.text  # a real sentence, not a number dump

    def test_the_answer_never_leaks_call_syntax(self, agent: Agent) -> None:
        """This is the bug the two-round loop exists to prevent."""
        for question in (
            "what is 7 times 23?",
            "what is 144 divided by 12?",
            "what is the weather in Delhi?",
            "what time is it right now?",
        ):
            reply = agent.answer(question)
            assert not _is_call_stub(reply.text), f"{question!r} -> {reply.text!r}"
            assert "functions." not in reply.text
            assert "{" not in reply.text and "}" not in reply.text

    def test_gibberish_asks_instead_of_inventing(self, agent: Agent) -> None:
        reply = agent.answer("blah blah wubble frobnicate")
        assert reply.tool_calls == []
        assert reply.text

    def test_the_answer_is_short_enough_to_speak(self, agent: Agent) -> None:
        reply = agent.answer("what is the weather in Mumbai right now?")
        assert len(reply.text) < 400

    def test_every_reply_is_recorded_for_the_ui(self, agent: Agent) -> None:
        reply = agent.answer("what is 12 divided by 4?")
        assert reply.elapsed_ms > 0
        assert all(c.to_dict()["name"] for c in reply.tool_calls)
