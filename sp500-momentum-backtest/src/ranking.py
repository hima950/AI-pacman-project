"""Prior calendar-year momentum ranking.

For selecting year Y's basket: rank S&P 500 constituents (as of 31 Dec of
year Y-1) by total return over calendar year Y-1, using only price data
dated on or before 31 Dec Y-1 (no look-ahead). A name needs a full year of
price history in Y-1 to be rankable, which naturally excludes mid-year IPOs
(they'd otherwise show an inflated "return since IPO" and get selected on a
partial, non-comparable window).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import config
from .membership import MembershipData

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    year: int                       # the holding year Y (basket applies to this year)
    ranking_year: int               # Y - 1, the year returns are computed over
    ranked: pd.DataFrame            # columns: ticker, return_y-1, rank
    basket: dict[int, list[str]]    # {n: [top-n tickers]} for each requested N
    n_membership: int                # size of point-in-time membership as of 31-Dec Y-1
    n_missing_price_data: int        # eligible by membership but we have no price series at all
    n_dropped_partial_year: int      # had *some* price data but not a full year (e.g. mid-year IPO)
    dropped_partial_year_tickers: list[str]


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31)


def rank_year(
    year: int,
    membership: MembershipData,
    prices: dict[str, pd.Series],
    top_ns: list[int],
    min_trading_days: int = config.MIN_TRADING_DAYS_FOR_FULL_YEAR,
) -> RankingResult:
    ranking_year = year - 1
    members = membership.members_as_of_year_end(ranking_year)
    ry_start, ry_end = _year_bounds(ranking_year)

    rows = []
    missing_price_data = 0
    dropped_partial: list[str] = []

    for ticker in sorted(members):
        series = prices.get(ticker)
        if series is None or series.empty:
            missing_price_data += 1
            continue

        # No look-ahead: only use data dated on/before 31-Dec of ranking_year.
        series = series[series.index <= ry_end]
        year_slice = series[(series.index >= ry_start) & (series.index <= ry_end)]

        if len(year_slice) < min_trading_days:
            dropped_partial.append(ticker)
            continue

        # Require the series to start at/near the beginning of the year too
        # (a ticker with a mid-year gap could still clear the day-count bar).
        first_obs = year_slice.index.min()
        if (first_obs - ry_start).days > 21:
            dropped_partial.append(ticker)
            continue

        # Reuse guard: make sure this symbol is documented as the *same*
        # listing across the ranking window, not a recycled ticker.
        if not membership.is_active_on(ticker, year_slice.index.min()) or not membership.is_active_on(
            ticker, year_slice.index.max()
        ):
            logger.info("Skipping %s for %d: ticker-reuse guard flagged discontinuity.", ticker, ranking_year)
            continue

        start_price = float(year_slice.iloc[0])
        end_price = float(year_slice.iloc[-1])
        if start_price <= 0:
            continue
        total_return = end_price / start_price - 1.0
        rows.append({"ticker": ticker, "return": total_return, "n_obs": len(year_slice)})

    ranked = pd.DataFrame(rows).sort_values("return", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    basket = {n: ranked.head(n)["ticker"].tolist() for n in top_ns}

    return RankingResult(
        year=year,
        ranking_year=ranking_year,
        ranked=ranked,
        basket=basket,
        n_membership=len(members),
        n_missing_price_data=missing_price_data,
        n_dropped_partial_year=len(dropped_partial),
        dropped_partial_year_tickers=dropped_partial,
    )


def rank_all_years(
    years: range,
    membership: MembershipData,
    prices: dict[str, pd.Series],
    top_ns: list[int],
) -> dict[int, RankingResult]:
    results = {}
    for y in years:
        try:
            results[y] = rank_year(y, membership, prices, top_ns)
        except ValueError as e:
            logger.warning("Skipping year %d: %s", y, e)
    return results


@dataclass
class TrailingRankingResult:
    as_of_date: pd.Timestamp
    lookback_years: int
    requested_lookback_years: int
    ranked: pd.DataFrame            # columns: ticker, return, n_obs
    winners: list[str]
    n_eligible_universe: int
    n_dropped_insufficient_history: int


def rank_trailing_period(
    as_of_date: pd.Timestamp,
    eligible_tickers: set[str],
    prices: dict[str, pd.Series],
    top_n: int,
    lookback_years: int = 10,
    min_lookback_years: int = 3,
    start_tolerance_days: int = 45,
) -> TrailingRankingResult:
    """Rank `eligible_tickers` by total return over the trailing
    `lookback_years` ending on `as_of_date`, using only data on/before
    `as_of_date` (no look-ahead). Used for the crash-deployment rule
    ("winners of the previous 10 years"), which can fire on any trading
    day rather than a calendar year boundary.

    If not enough tickers have a full `lookback_years` of history (e.g. a
    crash early in the backtest, before most names have a decade of
    data), the window is shortened one year at a time down to
    `min_lookback_years` rather than returning an empty ranking -- this is
    logged on the result via `lookback_years` vs `requested_lookback_years`.
    """
    requested = lookback_years
    for candidate_years in range(lookback_years, min_lookback_years - 1, -1):
        window_start = as_of_date - pd.DateOffset(years=candidate_years)
        rows = []
        dropped = 0
        for ticker in sorted(eligible_tickers):
            series = prices.get(ticker)
            if series is None or series.empty:
                continue
            series = series[series.index <= as_of_date]
            window = series[series.index >= window_start]
            if window.empty:
                dropped += 1
                continue
            first_obs = window.index.min()
            if (first_obs - window_start).days > start_tolerance_days:
                dropped += 1
                continue
            start_price = float(window.iloc[0])
            end_price = float(window.iloc[-1])
            if start_price <= 0:
                continue
            rows.append(
                {"ticker": ticker, "return": end_price / start_price - 1.0, "n_obs": len(window)}
            )
        if rows:
            ranked = pd.DataFrame(rows).sort_values("return", ascending=False).reset_index(drop=True)
            ranked["rank"] = ranked.index + 1
            winners = ranked.head(top_n)["ticker"].tolist()
            if candidate_years < requested:
                logger.warning(
                    "Trailing-return ranking at %s: only %d years of history broadly "
                    "available (requested %d); using a shortened lookback.",
                    as_of_date.date(), candidate_years, requested,
                )
            return TrailingRankingResult(
                as_of_date=as_of_date,
                lookback_years=candidate_years,
                requested_lookback_years=requested,
                ranked=ranked,
                winners=winners,
                n_eligible_universe=len(eligible_tickers),
                n_dropped_insufficient_history=dropped,
            )
    logger.warning(
        "Trailing-return ranking at %s: no ticker had even %d years of history -- "
        "returning an empty ranking.", as_of_date.date(), min_lookback_years,
    )
    return TrailingRankingResult(
        as_of_date=as_of_date,
        lookback_years=0,
        requested_lookback_years=requested,
        ranked=pd.DataFrame(columns=["ticker", "return", "n_obs", "rank"]),
        winners=[],
        n_eligible_universe=len(eligible_tickers),
        n_dropped_insufficient_history=len(eligible_tickers),
    )
