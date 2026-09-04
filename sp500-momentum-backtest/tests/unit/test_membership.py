"""Isolated unit tests for src/membership.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.membership import MembershipData, constant_membership


def test_constant_membership_same_set_every_year():
    tickers = {"A", "B", "C"}
    mem = constant_membership(tickers, range(2000, 2005))
    for y in range(2000, 2005):
        assert mem.members_as_of_year_end(y) == tickers
    # returned sets must be independent copies, not aliases of the input
    got = mem.members_as_of_year_end(2001)
    got.add("Z")
    assert "Z" not in mem.members_as_of_year_end(2002)


def test_members_as_of_year_end_raises_for_missing_year():
    mem = constant_membership({"A"}, range(2000, 2003))
    with pytest.raises(ValueError):
        mem.members_as_of_year_end(1999)


def test_members_as_of_nearest_year_end_falls_back_to_earlier_snapshot():
    rows = [
        {"as_of_year_end": 2005, "snapshot_date": None, "tickers": {"A", "B"}},
        {"as_of_year_end": 2010, "snapshot_date": None, "tickers": {"C", "D"}},
    ]
    mem = MembershipData(yearend=pd.DataFrame(rows), ticker_ranges=None, source="unit-test")
    # exact match
    assert mem.members_as_of_nearest_year_end(2010) == {"C", "D"}
    # falls back to the nearest earlier snapshot
    assert mem.members_as_of_nearest_year_end(2012) == {"C", "D"}
    assert mem.members_as_of_nearest_year_end(2007) == {"A", "B"}


def test_members_as_of_nearest_year_end_raises_when_nothing_early_enough():
    rows = [{"as_of_year_end": 2010, "snapshot_date": None, "tickers": {"A"}}]
    mem = MembershipData(yearend=pd.DataFrame(rows), ticker_ranges=None, source="unit-test")
    with pytest.raises(ValueError):
        mem.members_as_of_nearest_year_end(2005)


def test_is_active_on_defaults_true_without_range_data():
    mem = constant_membership({"A"}, range(2000, 2002))
    assert mem.is_active_on("A", pd.Timestamp("1970-01-01")) is True
    assert mem.is_active_on("NOT_EVEN_IN_UNIVERSE", pd.Timestamp("2020-01-01")) is True


def test_is_active_on_respects_documented_ranges():
    ranges = pd.DataFrame(
        [
            {"ticker": "AAL", "start_date": pd.Timestamp("1996-01-02"), "end_date": pd.Timestamp("1997-01-15")},
            {"ticker": "AAL", "start_date": pd.Timestamp("2015-03-23"), "end_date": pd.NaT},
        ]
    )
    mem = MembershipData(yearend=pd.DataFrame(), ticker_ranges=ranges, source="unit-test")
    assert mem.is_active_on("AAL", pd.Timestamp("1996-06-01")) is True
    assert mem.is_active_on("AAL", pd.Timestamp("2000-01-01")) is False  # gap between the two listings
    assert mem.is_active_on("AAL", pd.Timestamp("2020-01-01")) is True  # open-ended (NaT) second listing
    assert mem.is_active_on("UNLISTED_TICKER", pd.Timestamp("2020-01-01")) is True  # no data -> can't disprove
