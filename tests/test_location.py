"""Tests for location providers: manual coords, validation, GPSD (mocked).

Deterministic: no hardware, no daemons, no network beyond mocked sockets.
"""

import socket
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.location import providers as lp  # noqa: E402


class TestManualLocation(unittest.TestCase):
    def test_manual_labelled(self):
        loc = lp.manual_location(12.95, 77.55)
        self.assertEqual(loc.source, "manual")
        self.assertIsNone(loc.accuracy_m)
        self.assertAlmostEqual(loc.lat, 12.95)

    def test_manual_invalid(self):
        for lat, lon in [(91.0, 0.0), (0.0, 181.0), (float("nan"), 0.0), ("x", 0.0)]:
            with self.subTest(lat=lat, lon=lon), self.assertRaises(lp.LocationError):
                lp.manual_location(lat, lon)  # type: ignore[arg-type]


class TestStaleness(unittest.TestCase):
    def test_fresh_vs_stale(self):
        fresh = lp.Location(lat=0.0, lon=0.0, source="manual", observed_at=time.time())
        stale = lp.Location(lat=0.0, lon=0.0, source="gpsd", observed_at=time.time() - 120)
        self.assertFalse(lp.is_stale(fresh, 30.0))
        self.assertTrue(lp.is_stale(stale, 30.0))
        self.assertEqual(lp.check_fresh(fresh, 30.0), fresh)
        with self.assertRaises(lp.LocationError):
            lp.check_fresh(stale, 30.0)

    def test_missing_accuracy_ok(self):
        loc = lp.Location(lat=0.0, lon=0.0, source="manual", observed_at=time.time())
        self.assertIsNone(loc.accuracy_m)  # unknown, not an error


class TestGpsd(unittest.TestCase):
    def _tpv_socket(self, lines: list[bytes]):
        sock = mock.MagicMock()
        sock.recv.side_effect = lines + [b""]
        return sock

    def test_gpsd_fix(self):
        tpv = b'{"class":"TPV","mode":3,"lat":12.95,"lon":77.55,"eph":4.5}\n'
        sock = self._tpv_socket([tpv])
        with mock.patch("socket.create_connection", return_value=sock):
            loc = lp.gpsd_location()
        self.assertEqual(loc.source, "gpsd")
        self.assertAlmostEqual(loc.lat, 12.95)
        self.assertAlmostEqual(loc.accuracy_m, 4.5)

    def test_gpsd_daemon_missing(self):
        with mock.patch("socket.create_connection", side_effect=OSError("refused")):
            with self.assertRaises(lp.LocationError):
                lp.gpsd_location()

    def test_gpsd_no_fix(self):
        nofix = b'{"class":"TPV","mode":1}\n'
        sock = self._tpv_socket([nofix])
        with mock.patch("socket.create_connection", return_value=sock):
            with self.assertRaises(lp.LocationError):
                lp.gpsd_location(timeout_s=0.2)

    def test_gpsd_refuses_remote_host(self):
        with self.assertRaises(lp.LocationError):
            lp.gpsd_location(host="example.com")

    def test_gpsd_loopback_only(self):
        # Real socket class is untouched for non-loopback: provider must
        # never open remote connections (verified: raises before connect).
        with mock.patch("socket.create_connection") as conn:
            with self.assertRaises(lp.LocationError):
                lp.gpsd_location(host="192.0.2.1")
            conn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
