"""Live weather via Open-Meteo.

Keyless and JSON-first, which suits an LLM tool far better than the HTML-ish
``wttr.in`` output. Geocoding and forecast are two calls; both are cached so a
repeated question ("and now?", "in Pune?") does not re-hit the network.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes, condensed to the ones that actually matter
# for a spoken answer.
WMO: dict[int, str] = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}

# Regional aliases, so "Pune" resolves but so do the names people improvise.
_ALIASES: dict[str, str] = {
    "pune": "Pune", "bangalore": "Bengaluru", "bengaluru": "Bengaluru",
    "mumbai": "Mumbai", "bombay": "Mumbai", "delhi": "Delhi",
    "new delhi": "New Delhi", "chennai": "Chennai", "madras": "Chennai",
    "hyderabad": "Hyderabad", "kolkata": "Kolkata", "calcutta": "Kolkata",
    "ahmedabad": "Ahmedabad", "jaipur": "Jaipur", "lucknow": "Lucknow",
    "punjab": "Punjab", "noida": "Noida", "gurgaon": "Gurugram",
    "gurugram": "Gurugram", "kochi": "Kochi", "coimbatore": "Coimbatore",
    "nagpur": "Nagpur", "indore": "Indore", "bhopal": "Bhopal",
    "patna": "Patna", "surat": "Surat",
    "visakhapatnam": "Visakhapatnam", "thiruvananthapuram": "Thiruvananthapuram",
    "mysore": "Mysuru", "mysuru": "Mysuru", "varanasi": "Varanasi",
    "amritsar": "Amritsar", "chandigarh": "Chandigarh", "dehradun": "Dehradun",
}


class WeatherError(RuntimeError):
    """Weather lookup failed in a way the agent can report to the user."""


# Statuses worth trying again: the service is busy or briefly broken, not
# telling us no. A 404 (or any other 4xx) is a real answer, so retrying it would
# only add latency in front of the same failure.
_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
# Attempts after the first, and the pause before each. Open-Meteo rate-limits by
# per-minute budget, so one short backoff clears a burst without stalling speech
# for seconds on a genuine outage.
_ATTEMPTS = 3
_BACKOFF_S = (0.25, 0.75)
# How much longer than one request the whole lookup may take, so that a retry
# is affordable without the worst case becoming a stall that looks like a hang.
_DEADLINE_FACTOR = 1.5


# A raw transcript is a plausible argument, so "the weather in Pune right now"
# has to reduce to "Pune" before it goes to the geocoder.
_LEADING = ("what is the weather in ", "whats the weather in ", "the weather in ",
            "weather in ", "how is the weather in ", "in ")
_TRAILING = (
    " right now", " at the moment", " at this moment", " currently", " today",
    " now", " please", " rn", " these days", " presently", " tonight",
)


def _clean_query(query: str) -> str:
    lowered = query.lower().strip()
    for prefix in _LEADING:
        if lowered.startswith(prefix):
            query = query[len(prefix) :].strip()
            break
    lowered = query.lower()
    for suffix in _TRAILING:
        if lowered.endswith(suffix):
            query = query[: -len(suffix)].strip()
            break
    return query.strip(" ,.?!")


class WeatherClient:
    """Cached Open-Meteo client."""

    def __init__(
        self,
        *,
        ttl: float = 600.0,
        timeout: float = 8.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.ttl = ttl
        self._budget = timeout
        # A whole lookup is two requests plus at most one retry each. Allow some
        # slack over the per-request timeout so a retry is possible, but not so
        # much that a dead network turns into a twenty-second stall.
        self._deadline = timeout * _DEADLINE_FACTOR
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "friday-voice/0.1"},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ main
    async def current(self, city: str, unit: str = "celsius") -> dict[str, Any]:
        key = f"{city.strip().lower()}|{unit}"
        hit = self._cache.get(key)
        if hit and (time.monotonic() - hit[0]) < self.ttl:
            return {**hit[1], "cached": True}

        # One budget for the whole lookup. A weather question costs two HTTP
        # calls (geocode, then forecast) and either can be retried, so a
        # per-request timeout multiplied by attempts and by calls is a wait
        # several times longer than the number the operator actually set.
        deadline = time.monotonic() + self._deadline
        place = await self._geocode(city, deadline)
        data = await self._fetch(place, deadline)

        if unit.lower().startswith("f"):
            data["current"]["temperature"] = round(
                data["current"]["temperature"] * 9 / 5 + 32, 1
            )
            data["current"]["feels_like"] = round(
                data["current"]["feels_like"] * 9 / 5 + 32, 1
            )
            data["today"]["temp_max"] = round(data["today"]["temp_max"] * 9 / 5 + 32, 1)
            data["today"]["temp_min"] = round(data["today"]["temp_min"] * 9 / 5 + 32, 1)
            data["unit"] = "fahrenheit"
        else:
            data["unit"] = "celsius"

        self._cache[key] = (time.monotonic(), data)
        return {**data, "cached": False}

    # -------------------------------------------------------------- internals
    async def _get(
        self,
        url: str,
        params: dict[str, Any],
        what: str,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """GET with a bounded retry, so a transient blip is not a failed answer.

        The two call sites both want "raise WeatherError unless I got a 200 with
        JSON", so the retry lives here rather than being duplicated -- a
        half-retried client is worse than either.

        A single request gets the full configured ``timeout``. Slicing the
        budget three ways to fit the retry attempts in looks tidier and is wrong:
        a congested network answers these in well under a second *or* takes six
        or seven, and a third of the budget turns the slow-but-fine case into a
        failure. What needs bounding is the retry, not the request.

        ``deadline`` is a ``time.monotonic()`` value shared across the whole
        lookup, so the second request and any retries all draw from one budget
        rather than each getting a fresh full-length timeout.
        """
        if deadline is None:
            deadline = time.monotonic() + self._deadline
        # The loop always runs at least once, so this is only ever a placeholder.
        last = "the request was never attempted"
        for attempt in range(_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0.1:
                break
            per_attempt = max(0.5, min(self._budget, remaining))
            try:
                resp = await self._client.get(url, params=params, timeout=per_attempt)
            except httpx.HTTPError as exc:
                # httpx's timeout and network errors frequently stringify to
                # nothing at all, which used to surface as
                # "could not reach the geocoding service: " and tell nobody
                # anything. Name the type so the message is actionable.
                detail = str(exc).strip() or type(exc).__name__
                last = f"could not reach the {what} service ({detail})"
                # A connect error is worth another go; a read timeout may mean
                # the request landed, so only the first attempt is repeated.
                retryable = attempt == 0
            else:
                if resp.status_code == 200:
                    return resp.json() or {}
                last = f"{what} service returned HTTP {resp.status_code}"
                retryable = resp.status_code in _RETRY_STATUSES
            if not retryable or attempt == _ATTEMPTS - 1:
                break
            backoff = _BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)]
            # Never sleep past the deadline, and never start a retry we have no
            # meaningful time for.
            if time.monotonic() + backoff >= deadline:
                break
            await asyncio.sleep(backoff)
        raise WeatherError(last)

    async def _geocode(self, city: str, deadline: float | None = None) -> dict[str, Any]:
        query = (city or "").strip()
        if not query:
            raise WeatherError("no city given")
        query = _clean_query(query)
        if not query:
            raise WeatherError("no city given")
        query = _ALIASES.get(query.lower(), query)

        body = await self._get(
            GEOCODE_URL,
            {"name": query, "count": 5, "language": "en", "format": "json"},
            "geocoding",
            deadline,
        )

        results = (body or {}).get("results") or []
        if not results:
            raise WeatherError(f"I could not find a place called {query!r}")

        best = results[0]
        return {
            "name": best.get("name", query),
            "admin": best.get("admin1"),
            "country": best.get("country"),
            "country_code": best.get("country_code"),
            "latitude": best["latitude"],
            "longitude": best["longitude"],
            "timezone": best.get("timezone", "auto"),
            "ambiguous": len(results) > 1,
            "alternatives": [
                f"{r.get('name')}, {r.get('admin1')}, {r.get('country')}"
                for r in results[1:4]
            ],
        }

    async def _fetch(self, place: dict[str, Any], deadline: float | None = None) -> dict[str, Any]:
        params = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,weather_code,wind_speed_10m,is_day",
            "daily": "temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,sunrise,sunset",
            "timezone": "auto",
            "forecast_days": 1,
        }
        body = await self._get(FORECAST_URL, params, "weather", deadline)
        cur = body.get("current") or {}
        day = body.get("daily") or {}

        code = int(cur.get("weather_code", -1))
        return {
            "location": {
                "name": place["name"],
                "admin": place["admin"],
                "country": place["country"],
                "timezone": body.get("timezone", place["timezone"]),
                "ambiguous": place["ambiguous"],
                "alternatives": place["alternatives"],
            },
            "current": {
                "temperature": float(cur.get("temperature_2m", 0.0)),
                "feels_like": float(cur.get("apparent_temperature", 0.0)),
                "humidity": int(cur.get("relative_humidity_2m", 0)),
                "precipitation_mm": float(cur.get("precipitation", 0.0)),
                "wind_kph": float(cur.get("wind_speed_10m", 0.0)),
                "condition": WMO.get(code, "unknown conditions"),
                "is_day": bool(cur.get("is_day", 1)),
                "observed_at": cur.get("time"),
            },
            "today": {
                "temp_max": float((day.get("temperature_2m_max") or [0.0])[0]),
                "temp_min": float((day.get("temperature_2m_min") or [0.0])[0]),
                "precip_probability": int(
                    (day.get("precipitation_probability_max") or [0])[0] or 0
                ),
                "sunrise": (day.get("sunrise") or [None])[0],
                "sunset": (day.get("sunset") or [None])[0],
            },
        }
