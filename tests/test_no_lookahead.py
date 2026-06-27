"""Guard test: the point-in-time feed must never leak future data."""

import pandas as pd
import pytest

from src.data.feed import PointInTimeFeed

SYMBOL = "BTC/USDT"


def test_history_excludes_as_of_and_future():
    feed = PointInTimeFeed(SYMBOL)
    # Pick a date roughly in the middle of available history
    mid_date = feed.dates[len(feed.dates) // 2]

    hist = feed.history_before(mid_date)

    # Every row returned must be strictly before the as_of date
    assert (hist.index < mid_date).all(), "Feed leaked the as_of date or future!"
    # And it must actually return the past (non-empty)
    assert len(hist) > 0


def test_history_grows_monotonically():
    """As the as-of date advances, the agent should see more, never less."""
    feed = PointInTimeFeed(SYMBOL)
    early = feed.dates[100]
    later = feed.dates[200]
    assert len(feed.history_before(later)) > len(feed.history_before(early))
