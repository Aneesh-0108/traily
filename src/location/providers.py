"""Structured location readings from a local source.

`Location` is the single contract between acquisition and the voice
pipeline: latitude/longitude plus source metadata. Acquisition stays in
this module so a future GPS device can be added without touching the
pipeline; filtering stays in `src/places/geo.py`.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any

from src.places.pack import is_valid_lat, is_valid_lon


@dataclass(frozen=True)
class Location:
    lat: float
    lon: float
    source: str  # e.g. "manual", "gpsd" — never "live" unless truly live
    observed_at: float  # time.time() epoch seconds
    accuracy_m: float | None = None  # None == unknown, not zero


class LocationError(RuntimeError):
    """Unavailable source, no fix, invalid/stale coordinates."""


def validate_coords(lat: Any, lon: Any) -> tuple[float, float]:
    """Return (lat, lon) as floats or raise LocationError. Never fakes data."""
    if isinstance(lat, bool) or isinstance(lon, bool):
        raise LocationError(f"invalid coordinates: lat={lat!r} lon={lon!r}")
    try:
        flat, flon = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise LocationError(f"invalid coordinates: lat={lat!r} lon={lon!r}") from exc
    if flat != flat or flon != flon:  # NaN
        raise LocationError(f"invalid coordinates: lat={lat!r} lon={lon!r}")
    if not is_valid_lat(flat) or not is_valid_lon(flon):
        raise LocationError(f"coordinates out of range: lat={flat} lon={flon}")
    return flat, flon


def is_stale(loc: Location, max_age_s: float = 30.0, now: float | None = None) -> bool:
    """True when the reading is older than max_age_s (or from the future)."""
    current = time.time() if now is None else now
    age = current - loc.observed_at
    return age < 0 or age > max_age_s


def check_fresh(loc: Location, max_age_s: float = 30.0) -> Location:
    """Return loc or raise LocationError when stale."""
    if is_stale(loc, max_age_s):
        raise LocationError(
            f"stale {loc.source} reading ({loc.observed_at}); "
            "refusing to answer from an old position"
        )
    return loc


def manual_location(lat: Any, lon: Any) -> Location:
    """Explicit development/testing coordinates.

    ALWAYS labelled source="manual". Never described as live location.
    """
    flat, flon = validate_coords(lat, lon)
    return Location(lat=flat, lon=flon, source="manual", observed_at=time.time())


# --- Local GPSD client (localhost only) ------------------------------------
# Talks to a GPSD daemon on 127.0.0.1:2947 using the GPSD JSON protocol.
# Loopback only: no remote hosts, no positioning APIs, no network
# dependency beyond the local daemon socket. Requires a GPSD daemon plus
# a receiver, NEITHER of which is installed or started by this code —
# see README for setup. When absent, every call raises LocationError with
# a clear diagnostic instead of faking coordinates.

GPSD_HOST = "127.0.0.1"
GPSD_PORT = 2947


def gpsd_location(
    host: str = GPSD_HOST,
    port: int = GPSD_PORT,
    timeout_s: float = 5.0,
    max_age_s: float = 30.0,
) -> Location:
    """Read one TPV fix from local GPSD. Raises LocationError when:

    - no daemon answers on host:port (refuses anything but loopback);
    - no fix arrives before timeout_s (no fix / no sky view);
    - the fix has no valid lat/lon or is stale.
    """
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise LocationError(f"refusing non-local GPSD host: {host!r}")
    import json as _json

    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except OSError as exc:
        raise LocationError(
            f"no GPSD daemon on {host}:{port} ({exc}); "
            "live location unavailable — use --lat/--lon test coordinates"
        ) from exc
    sock.settimeout(timeout_s)
    deadline = time.time() + timeout_s
    try:
        sock.sendall(b'?WATCH={"enable":true,"json":true};')
        buf = b""
        while time.time() < deadline:
            try:
                chunk = sock.recv(4096)
            except socket.timeout as exc:
                raise LocationError("GPSD gave no fix before timeout (no fix?)") from exc
            if not chunk:
                raise LocationError("GPSD closed the connection without a fix")
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                try:
                    msg = _json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if not isinstance(msg, dict) or msg.get("class") != "TPV":
                    continue
                if msg.get("mode", 1) < 2:  # 0/1 == no fix
                    continue
                try:
                    flat, flon = validate_coords(msg.get("lat"), msg.get("lon"))
                except LocationError:
                    continue
                accuracy = msg.get("eph")
                if not isinstance(accuracy, (int, float)):
                    accuracy = None
                observed = msg.get("time")
                loc = Location(
                    lat=flat,
                    lon=flon,
                    source="gpsd",
                    observed_at=time.time(),
                    accuracy_m=accuracy,
                )
                _ = observed
                return check_fresh(loc, max_age_s)
        raise LocationError("GPSD gave no fix before timeout (no fix?)")
    finally:
        sock.close()
