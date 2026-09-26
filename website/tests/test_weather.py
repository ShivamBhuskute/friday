"""The weather tool.

No network here: httpx's MockTransport stands in for Open-Meteo so the tests are
deterministic and instant. The point is not that Pune is 28 degrees, it is that
a tool failure becomes a sentence the agent can say out loud.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from server.tools.weather import WeatherClient, WeatherError

GEO_BODY = {
    "results": [
        {
            "name": "Pune",
            "admin1": "Maharashtra",
            "country": "India",
            "country_code": "IN",
            "latitude": 18.52,
            "longitude": 73.85,
            "timezone": "Asia/Kolkata",
        },
        {
            "name": "Pune",
            "admin1": "Idaho",
            "country": "United States",
            "country_code": "US",
            "latitude": 44.2,
            "longitude": -116.2,
            "timezone": "America/Boise",
        },
    ]
}

FORECAST_BODY = {
    "timezone": "Asia/Kolkata",
    "current": {
        "time": "2026-01-01T10:00",
        "temperature_2m": 28.4,
        "apparent_temperature": 30.1,
        "relative_humidity_2m": 62,
        "precipitation": 0.0,
        "weather_code": 2,
        "wind_speed_10m": 11.5,
        "is_day": 1,
    },
    "daily": {
        "temperature_2m_max": [31.2],
        "temperature_2m_min": [19.8],
        "precipitation_probability_max": [30],
        "sunrise": ["2026-01-01T06:59"],
        "sunset": ["2026-01-01T18:24"],
    },
}


def make_transport(
    *,
    geo=None,
    forecast=None,
    geo_status: int = 200,
    forecast_status: int = 200,
    raise_on: set[str] | None = None,
):
    """A MockTransport that answers the two Open-Meteo endpoints."""
    raise_on = raise_on or set()
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        name = "geo" if "geocoding" in str(request.url) else "forecast"
        seen.append((name, dict(request.url.params)))
        if name in raise_on:
            raise httpx.ConnectError("simulated network failure", request=request)
        if name == "geo":
            return httpx.Response(geo_status, json=geo if geo is not None else GEO_BODY)
        return httpx.Response(
            forecast_status, json=forecast if forecast is not None else FORECAST_BODY
        )

    return httpx.MockTransport(handler), seen


@pytest.fixture
async def client():
    """A client wired to the mock transport, recording every request it makes."""
    transport, seen = make_transport()
    c = WeatherClient(transport=transport, ttl=600.0)
    c.seen = seen  # type: ignore[attr-defined]
    try:
        yield c
    finally:
        await c.aclose()


class TestCurrent:
    async def test_returns_a_useful_payload(self, client: WeatherClient) -> None:
        data = await client.current("Pune")
        assert data["location"]["name"] == "Pune"
        assert data["location"]["country"] == "India"
        assert data["current"]["temperature"] == pytest.approx(28.4)
        assert data["current"]["condition"] == "partly cloudy"
        assert data["current"]["humidity"] == 62
        assert data["today"]["temp_max"] == pytest.approx(31.2)
        assert data["unit"] == "celsius"
        assert data["cached"] is False

    async def test_ambiguous_name_reports_alternatives(self, client: WeatherClient) -> None:
        data = await client.current("Pune")
        assert data["location"]["ambiguous"] is True
        assert any("Idaho" in a for a in data["location"]["alternatives"])

    async def test_fahrenheit_conversion(self, client: WeatherClient) -> None:
        data = await client.current("Pune", unit="fahrenheit")
        assert data["unit"] == "fahrenheit"
        assert data["current"]["temperature"] == pytest.approx(83.1, abs=0.1)
        assert data["today"]["temp_max"] == pytest.approx(88.2, abs=0.1)

    async def test_bangalore_resolves_to_bengaluru(self, client: WeatherClient) -> None:
        await client.current("bangalore")
        _, params = client.seen[0]  # type: ignore[attr-defined]
        assert params["name"] == "Bengaluru"

    async def test_leading_filler_is_stripped(self, client: WeatherClient) -> None:
        """A transcript can arrive as "the weather in Pune right now"."""
        await client.current("the weather in Pune right now")
        _, params = client.seen[0]  # type: ignore[attr-defined]
        assert params["name"] == "Pune"

    async def test_forecast_params_are_sane(self, client: WeatherClient) -> None:
        await client.current("Pune")
        name, params = client.seen[1]  # type: ignore[attr-defined]
        assert name == "forecast"
        assert float(params["latitude"]) == pytest.approx(18.52)
        assert float(params["longitude"]) == pytest.approx(73.85)
        assert "temperature_2m" in params["current"]


class TestCaching:
    async def test_second_lookup_is_cached(self, client: WeatherClient) -> None:
        first = await client.current("Pune")
        second = await client.current("Pune")
        assert first["cached"] is False
        assert second["cached"] is True
        assert len(client.seen) == 2  # type: ignore[attr-defined]  (geo + forecast only)

    async def test_unit_change_is_a_separate_cache_key(self, client: WeatherClient) -> None:
        await client.current("Pune", unit="celsius")
        await client.current("Pune", unit="fahrenheit")
        assert len(client.seen) == 4  # type: ignore[attr-defined]

    async def test_city_is_case_insensitive_in_the_cache(self, client: WeatherClient) -> None:
        await client.current("Pune")
        second = await client.current("  pune  ")
        assert second["cached"] is True

    async def test_expired_cache_refetches(self) -> None:
        transport, seen = make_transport()
        c = WeatherClient(transport=transport, ttl=0.0)
        await c.current("Pune")
        await c.current("Pune")
        assert len(seen) == 4


class TestFailures:
    """Every failure has to surface as WeatherError, never as a traceback."""

    async def test_unknown_city(self) -> None:
        transport, _ = make_transport(geo={"results": []})
        c = WeatherClient(transport=transport)
        with pytest.raises(WeatherError, match="could not find a place"):
            await c.current("Atlantis")

    async def test_empty_city(self, client: WeatherClient) -> None:
        with pytest.raises(WeatherError, match="no city"):
            await client.current("   ")

    async def test_geocoding_http_error(self) -> None:
        transport, _ = make_transport(geo_status=500)
        c = WeatherClient(transport=transport)
        with pytest.raises(WeatherError, match="HTTP 500"):
            await c.current("Pune")

    async def test_forecast_http_error(self) -> None:
        transport, _ = make_transport(forecast_status=502)
        c = WeatherClient(transport=transport)
        with pytest.raises(WeatherError, match="HTTP 502"):
            await c.current("Pune")

    async def test_network_failure_mentions_the_service(self) -> None:
        transport, _ = make_transport(raise_on={"geo"})
        c = WeatherClient(transport=transport)
        with pytest.raises(WeatherError, match="geocoding service"):
            await c.current("Pune")

    async def test_error_message_names_the_exception_when_it_has_none(self) -> None:
        """httpx's timeout errors stringify to nothing, which told nobody anything.

        The real failure this came from surfaced as
        `could not reach the geocoding service: ` -- a colon, a space, and no
        explanation of what had actually gone wrong.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            # No message, exactly like the real httpx.ReadTimeout.
            raise httpx.ReadTimeout("", request=request)

        c = WeatherClient(transport=httpx.MockTransport(handler))
        with pytest.raises(WeatherError) as excinfo:
            await c.current("Pune")
        message = str(excinfo.value)
        assert not message.endswith(":"), f"error message says nothing: {message!r}"
        assert "ReadTimeout" in message
        assert "geocoding service" in message

    async def test_a_slow_but_successful_request_still_succeeds(self) -> None:
        """A single request gets the whole configured timeout.

        Slicing the budget three ways to fit the retries in looks tidier and is
        actively harmful: this network answers Open-Meteo in well under a second
        or takes six or seven, and a third of an 8 s budget turns the slow-but-
        fine case into a failure the user sees.
        """
        timeouts: list[float | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            spec = request.extensions.get("timeout") or {}
            timeouts.append(spec.get("read"))
            if "geocoding" in str(request.url):
                return httpx.Response(200, json=GEO_BODY)
            return httpx.Response(200, json=FORECAST_BODY)

        c = WeatherClient(transport=httpx.MockTransport(handler), timeout=8.0)
        await c.current("Pune")
        assert timeouts
        assert all(t == pytest.approx(8.0) for t in timeouts), timeouts

    async def test_a_retry_cannot_push_the_lookup_past_its_budget(self) -> None:
        """The retry is what needs bounding, not the request.

        Two requests plus a retry each must not become a twenty-second stall in
        front of the user and in front of the LLM.
        """
        attempts: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            # Burn more than the per-request timeout, then time out.
            time.sleep(0.30)
            attempts.append(time.monotonic())
            raise httpx.ReadTimeout("", request=request)

        budget = 0.3
        c = WeatherClient(transport=httpx.MockTransport(handler), timeout=budget)
        t0 = time.monotonic()
        with pytest.raises(WeatherError):
            await c.current("Pune")
        elapsed = time.monotonic() - t0

        span = budget * 1.5
        assert elapsed <= span + 0.15, f"took {elapsed:.2f}s, budget was {span:.2f}s"
        assert len(attempts) == 1, f"retried with no time left: {len(attempts)} attempts"

    async def test_a_read_timeout_is_not_retried_forever(self) -> None:
        """A read timeout may mean the request landed, so it gets one extra go
        and no more."""
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            raise httpx.ReadTimeout("", request=request)

        c = WeatherClient(transport=httpx.MockTransport(handler), timeout=8.0)
        with pytest.raises(WeatherError):
            await c.current("Pune")
        assert len(attempts) == 2, attempts

    async def test_a_404_is_not_retried(self) -> None:
        """A 404 is an answer, not a hiccup. Retrying it only adds latency in
        front of the same failure."""
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(404, json={})

        c = WeatherClient(transport=httpx.MockTransport(handler), timeout=8.0)
        with pytest.raises(WeatherError, match="HTTP 404"):
            await c.current("Pune")
        assert len(attempts) == 1, attempts

    async def test_missing_fields_default_safely(self) -> None:
        transport, _ = make_transport(forecast={"current": {}, "daily": {}})
        c = WeatherClient(transport=transport)
        data = await c.current("Pune")
        assert data["current"]["temperature"] == 0.0
        assert data["current"]["condition"] == "unknown conditions"
        assert data["today"]["temp_max"] == 0.0

    async def test_unknown_wmo_code_does_not_crash(self) -> None:
        body = json.loads(json.dumps(FORECAST_BODY))
        body["current"]["weather_code"] = 999
        transport, _ = make_transport(forecast=body)
        c = WeatherClient(transport=transport)
        assert (await c.current("Pune"))["current"]["condition"] == "unknown conditions"


def sequence_transport(statuses: list[int] | None = None, *, fail_first: int = 0):
    """A transport that walks a script, so a retry can be observed.

    ``statuses`` is consumed one entry per request; the last entry repeats, so a
    script shorter than the number of attempts still terminates.
    """
    script = list(statuses or [])
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        name = "geo" if "geocoding" in str(request.url) else "forecast"
        seen.append(name)
        step = min(len(seen) - 1, len(script) - 1)
        if len(seen) <= fail_first:
            raise httpx.ConnectError("simulated network failure", request=request)
        status = script[step] if script else 200
        if status == 0:
            raise httpx.ConnectError("simulated network failure", request=request)
        body = GEO_BODY if name == "geo" else FORECAST_BODY
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), seen


@pytest.fixture
def instant_backoff(monkeypatch):
    """Remove the real sleep so a retry test costs milliseconds, not a second."""
    from server.tools import weather as weather_module

    monkeypatch.setattr(weather_module, "_BACKOFF_S", (0.0, 0.0))


class TestRetry:
    """A transient blip must not become "the weather is unavailable"."""

    async def test_a_429_is_retried_and_then_succeeds(self, instant_backoff) -> None:
        transport, seen = sequence_transport([429, 200])
        c = WeatherClient(transport=transport)
        try:
            data = await c.current("Pune")
        finally:
            await c.aclose()
        assert data["location"]["name"] == "Pune"
        # Two geocoding attempts, then one forecast.
        assert seen == ["geo", "geo", "forecast"]

    async def test_a_503_is_retried(self, instant_backoff) -> None:
        transport, seen = sequence_transport([503, 503, 200])
        c = WeatherClient(transport=transport)
        try:
            assert (await c.current("Pune"))["unit"] == "celsius"
        finally:
            await c.aclose()
        assert seen.count("geo") == 3

    async def test_a_dropped_connection_is_retried_once(self, instant_backoff) -> None:
        transport, seen = sequence_transport(fail_first=1)
        c = WeatherClient(transport=transport)
        try:
            assert (await c.current("Pune"))["location"]["name"] == "Pune"
        finally:
            await c.aclose()
        assert seen == ["geo", "geo", "forecast"]

    async def test_it_gives_up_after_a_bounded_number_of_attempts(
        self, instant_backoff
    ) -> None:
        transport, seen = sequence_transport([429])
        c = WeatherClient(transport=transport)
        try:
            with pytest.raises(WeatherError, match="HTTP 429"):
                await c.current("Pune")
        finally:
            await c.aclose()
        from server.tools.weather import _ATTEMPTS

        assert len(seen) == _ATTEMPTS

    async def test_a_404_is_not_retried(self, instant_backoff) -> None:
        """A real "no" is an answer; retrying it only delays the same failure."""
        transport, seen = sequence_transport([404])
        c = WeatherClient(transport=transport)
        try:
            with pytest.raises(WeatherError, match="HTTP 404"):
                await c.current("Pune")
        finally:
            await c.aclose()
        assert len(seen) == 1

    async def test_the_error_names_the_service_that_failed(self, instant_backoff) -> None:
        transport, _ = sequence_transport([500])
        c = WeatherClient(transport=transport)
        try:
            with pytest.raises(WeatherError, match="geocoding service"):
                await c.current("Pune")
        finally:
            await c.aclose()

    async def test_a_retried_forecast_still_succeeds(self, instant_backoff) -> None:
        """The second call in the chain needs the same treatment as the first."""
        calls = {"geo": 0, "forecast": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            name = "geo" if "geocoding" in str(request.url) else "forecast"
            calls[name] += 1
            if name == "forecast" and calls[name] == 1:
                return httpx.Response(504, json={})
            return httpx.Response(200, json=GEO_BODY if name == "geo" else FORECAST_BODY)

        c = WeatherClient(transport=httpx.MockTransport(handler))
        try:
            # The value proves the *second* forecast response is the one used.
            assert (await c.current("Pune"))["current"]["temperature"] == 28.4
        finally:
            await c.aclose()
        assert calls == {"geo": 1, "forecast": 2}
