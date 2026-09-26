"""Tool wiring: build the registry the agent and the pipeline share."""

from __future__ import annotations

import json
import os
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import Config
from .calc import CalcError, evaluate
from .registry import Tool, ToolRegistry
from .weather import WeatherClient, WeatherError

__all__ = ["build_registry", "ToolRegistry", "CalcError"]


def build_registry(cfg: Config, weather: WeatherClient | None = None) -> ToolRegistry:
    reg = ToolRegistry()

    async def get_weather(args: dict[str, Any]) -> dict[str, Any]:
        city = str(args.get("city") or args.get("location") or "").strip()
        if not city:
            raise WeatherError("a city is required")
        unit = str(args.get("unit") or "celsius")
        client = weather or _shared_weather(cfg)
        return await client.current(city, unit)

    reg.register(
        Tool(
            name="get_weather",
            description=(
                "Get the current live weather and today's high/low for a city. "
                "Always use this for any question about current or live weather; "
                "never guess the temperature."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Pune' or 'New York'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default celsius.",
                    },
                },
                "required": ["city"],
            },
            handler=get_weather,
        )
    )

    async def calculate(args: dict[str, Any]) -> dict[str, Any]:
        expr = args.get("expression") or args.get("query") or ""
        result = evaluate(str(expr))
        # Only the pre-formatted string is exposed. Handing the model the raw
        # float as well just invites it to answer "12.0" out loud.
        return {
            "expression": result.expression,
            "result": result.display,
        }

    reg.register(
        Tool(
            name="calculate",
            description=(
                "Evaluate an arithmetic expression exactly. Use this for any "
                "multiplication, division, powers or percentages instead of doing "
                "the arithmetic yourself. Spoken words are accepted, e.g. "
                "'7 times 23'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression, e.g. '7 * 23'.",
                    }
                },
                "required": ["expression"],
            },
            handler=calculate,
        )
    )

    async def get_datetime(args: dict[str, Any]) -> dict[str, Any]:
        tz_name = str(args.get("timezone") or "local")
        if tz_name.lower() in {"local", "", "here", "now"}:
            local = datetime.now().astimezone()
            return {
                "timezone": str(local.tzinfo),
                "iso": local.isoformat(timespec="seconds"),
                "date": local.date().isoformat(),
                "time": local.strftime("%H:%M:%S"),
                "weekday": local.strftime("%A"),
            }
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {tz_name}") from exc
        now = datetime.now(UTC).astimezone(tz)
        return {
            "timezone": tz_name,
            "iso": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
        }

    reg.register(
        Tool(
            name="get_datetime",
            description="Get the current date and time, optionally in a named IANA timezone.",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone such as 'Asia/Kolkata'. Default local.",
                    }
                },
            },
            handler=get_datetime,
        )
    )

    async def system_status(_args: dict[str, Any]) -> dict[str, Any]:
        return _system_status(cfg)

    reg.register(
        Tool(
            name="system_status",
            description=(
                "Report the health of this machine and the connected FRIDAY device: "
                "CPU load, memory, disk and the device's free heap / Wi-Fi signal."
            ),
            parameters={"type": "object", "properties": {}},
            handler=system_status,
        )
    )

    return reg


# --------------------------------------------------------------------- shared

_weather_singleton: WeatherClient | None = None


def _shared_weather(cfg: Config) -> WeatherClient:
    global _weather_singleton
    if _weather_singleton is None:
        _weather_singleton = WeatherClient(
            ttl=cfg.tools.weather_cache_ttl,
            timeout=cfg.tools.weather_timeout_s,
        )
    return _weather_singleton


def _load_sysmon(cfg: Config) -> dict[str, Any] | None:
    """Read the device's system-monitor snapshot if it has written one."""
    path = cfg.sysmon_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _system_status(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {"host": _host_status(), "device": _load_sysmon(cfg)}
    if out["device"] is None:
        out["device_note"] = "no sysmon snapshot pushed by the device yet"
    return out


def _host_status() -> dict[str, Any]:
    load = None
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass

    mem_used = mem_total = None
    try:
        meminfo = Path("/proc/meminfo").read_text()
        vals = {}
        for line in meminfo.splitlines():
            key, _, rest = line.partition(":")
            vals[key] = int(rest.strip().split()[0]) * 1024
        mem_total = vals.get("MemTotal")
        mem_used = mem_total - vals.get("MemAvailable", 0)
    except (OSError, ValueError, IndexError):
        pass

    disk_free = None
    try:
        disk_free = shutil.disk_usage("/home").free
    except OSError:
        pass

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "load_1m": round(load, 2) if load is not None else None,
        "cpu_count": _cpu_count(),
        "memory_used_gb": round(mem_used / 1e9, 1) if mem_used else None,
        "memory_total_gb": round(mem_total / 1e9, 1) if mem_total else None,
        "disk_free_gb": round(disk_free / 1e9, 1) if disk_free else None,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 0
