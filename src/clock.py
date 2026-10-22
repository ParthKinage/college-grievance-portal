"""
One clock for the whole app.

Cloud servers usually run on UTC time, which is 5 hours 30 minutes behind
India. If we used datetime.now() directly, ticket times, deadlines and the
"Overdue" count would all be wrong on the deployed app. Every timestamp in
the app comes from now() below, which always uses the college's time zone.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

# The college's time zone. Change this for a college in another country.
APP_TIMEZONE = ZoneInfo("Asia/Kolkata")


def now():
    """Current date and time in the college's time zone (without time zone info, like stored values)."""
    return datetime.now(APP_TIMEZONE).replace(tzinfo=None)


def from_timestamp(seconds):
    """Convert a file timestamp (seconds since 1970) to the college's local date and time."""
    return datetime.fromtimestamp(seconds, APP_TIMEZONE).replace(tzinfo=None)
