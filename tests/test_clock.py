"""Tests for the app clock (always India time, even on a UTC cloud server)."""

from datetime import datetime, timedelta, timezone

from src.clock import from_timestamp, now


def test_now_is_india_time():
    """now() is UTC + 5:30, whatever time zone the computer or server uses."""
    expected = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=5, minutes=30)
    assert abs(now() - expected) < timedelta(seconds=5)


def test_from_timestamp_is_india_time():
    """Unix time 0 (1 Jan 1970, 00:00 UTC) is 05:30 in India."""
    assert from_timestamp(0) == datetime(1970, 1, 1, 5, 30)
