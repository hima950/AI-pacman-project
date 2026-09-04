"""Isolated unit tests for src/ranking.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src import ranking
from src.membership import MembershipData


def _yearend(years_and_tickers: dict[int, set[str]]) -> MembershipData:
    rows = [{"as_of_year_end": y, "snapshot_date": None, "tickers": tks} for y, tks in years_and_tickers.items()]
    return MembershipData(yearend=pd.DataFrame(rows), ticker_ranges=None, source="unit-test")


def _flat_series(start: str, end: str, price: float) -> pd.Series:
    idx = pd.bdate_range(start, end)
    return pd.Series([price] * len(idx), index=idx)


def _linear_growth_series(start: str, end: str, start_price: float, end_price: float) -> pd.Series:
    idx = pd.bdate_range(start, end)
    n = len(idx)
    return pd.Series([start_price + (end_price - start_price) * i / (n - 1) for i in range(n)], index=idx)


def test_rank_year_orders_by_return_descending():
    mem = _yearend({2019: {"A", "B", "C"}})
    prices = {
        "A": _linear_growth_series("2019-01-02", "2019-12-31", 100, 110),   # +10%
        "B": _linear_growth_series("2019-01-02", "2019-12-31", 100, 150),   # +50%
        "C": _linear_growth_series("2019-01-02", "2019-12-31", 100, 90),    # -10%
    }
    result = ranking.rank_year(2020, mem, prices, top_ns=[2])
    assert list(result.ranked["ticker"]) == ["B", "A", "C"]
    assert result.basket[2] == ["B", "A"]
    assert result.ranking_year == 2019
    assert result.n_membership == 3


def test_rank_year_drops_partial_year_history():
    mem = _yearend({2019: {"A", "B"}})
    prices = {
        "A": _linear_growth_series("2019-01-02", "2019-12-31", 100, 120),
        "B": _linear_growth_series("2019-08-01", "2019-12-31", 100, 200),  # mid-year IPO, huge "return"
    }
    result = ranking.rank_year(2020, mem, prices, top_ns=[2])
    assert "B" not in result.ranked["ticker"].tolist()
    assert result.n_dropped_partial_year == 1
    assert result.dropped_partial_year_tickers == ["B"]
    # the surviving name should still be ranked and basketed
    assert result.basket[2] == ["A"]


def test_rank_year_counts_missing_price_data_without_crashing():
    mem = _yearend({2019: {"A", "B"}})
    prices = {"A": _linear_growth_series("2019-01-02", "2019-12-31", 100, 110)}
    result = ranking.rank_year(2020, mem, prices, top_ns=[5])
    assert result.n_missing_price_data == 1
    assert result.ranked["ticker"].tolist() == ["A"]


def test_rank_year_raises_for_year_without_snapshot():
    mem = _yearend({2019: {"A"}})
    prices = {"A": _linear_growth_series("2019-01-02", "2019-12-31", 100, 110)}
    with pytest.raises(ValueError):
        ranking.rank_year(2025, mem, prices, top_ns=[1])  # needs 2024 snapshot, not present


def test_rank_year_no_lookahead_beyond_ranking_year():
    # A rallies hard AFTER the ranking year ends -- must not affect the 2019 ranking.
    mem = _yearend({2019: {"A", "B"}})
    idx_2019 = pd.bdate_range("2019-01-02", "2019-12-31")
    idx_2020 = pd.bdate_range("2020-01-02", "2020-06-30")
    a_2019 = pd.Series([100 + 2 * i / len(idx_2019) for i in range(len(idx_2019))], index=idx_2019)  # +2%
    a_2020_spike = pd.Series([1000.0] * len(idx_2020), index=idx_2020)  # huge, but in the future
    a = pd.concat([a_2019, a_2020_spike])
    b = _linear_growth_series("2019-01-02", "2019-12-31", 100, 120)  # +20%, no future spike
    result = ranking.rank_year(2020, mem, {"A": a, "B": b}, top_ns=[2])
    assert result.ranked.set_index("ticker").loc["A", "return"] < 0.1  # nowhere near the 900% spike
    assert result.basket[2][0] == "B"


def test_rank_trailing_period_excludes_insufficient_history():
    as_of = pd.Timestamp("2010-01-01")
    long_history = _linear_growth_series("1998-01-01", "2010-01-01", 100, 300)   # ~12y, strong return
    short_history = _linear_growth_series("2009-06-01", "2010-01-01", 100, 500)  # <1y, huge % but not enough history
    prices = {"LONG": long_history, "SHORT": short_history}
    result = ranking.rank_trailing_period(
        as_of, {"LONG", "SHORT"}, prices, top_n=5, lookback_years=10, min_lookback_years=3,
    )
    assert result.winners == ["LONG"]
    assert result.n_dropped_insufficient_history == 1


def test_rank_trailing_period_shortens_lookback_when_needed():
    as_of = pd.Timestamp("2010-01-01")
    # Nobody has 10 years; everybody has ~5.
    prices = {
        "X": _linear_growth_series("2005-01-01", "2010-01-01", 100, 200),
        "Y": _linear_growth_series("2005-01-01", "2010-01-01", 100, 150),
    }
    result = ranking.rank_trailing_period(
        as_of, {"X", "Y"}, prices, top_n=5, lookback_years=10, min_lookback_years=3,
    )
    assert result.lookback_years < result.requested_lookback_years
    assert result.lookback_years >= 3
    assert result.winners[0] == "X"


def test_rank_trailing_period_empty_when_nobody_qualifies():
    as_of = pd.Timestamp("2010-01-01")
    prices = {"Z": _linear_growth_series("2009-11-01", "2010-01-01", 100, 110)}  # ~2 months only
    result = ranking.rank_trailing_period(
        as_of, {"Z"}, prices, top_n=5, lookback_years=10, min_lookback_years=3,
    )
    assert result.winners == []
    assert result.lookback_years == 0
    assert result.n_dropped_insufficient_history == 1
