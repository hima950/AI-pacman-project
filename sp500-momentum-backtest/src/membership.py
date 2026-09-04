"""Point-in-time S&P 500 membership reconstruction.

Screening past winners against *today's* S&P 500 roster is the classic
survivorship-bias bug in this kind of backtest: it silently deletes every
name that was later removed (acquired, delisted, relegated to a smaller
index) and inflates momentum-style strategies. This module avoids that by
using a dated, point-in-time membership history instead of the current
constituent list.

Data source (see DATA_SOURCES.md for full provenance / limitations):
    fja05680/sp500 on GitHub (MIT licensed). A community-maintained
    reconstruction built by tracking Wikipedia's "Selected changes to the
    list of S&P 500 components" table over time, cross-checked against
    Wikipedia's current list and originally seeded from a dataset used in
    Andreas Clenow's "Trading Evolved". This is approach (2) from the task
    brief ("reconstruct membership backwards ... from the documented
    history of index additions and removals"), pre-built by a third party
    rather than parsed by this code directly from Wikipedia -- it is NOT an
    official vendor point-in-time feed (S&P/CRSP), so treat it as good but
    not authoritative. Known soft spots (per the maintainer): the first
    ~5 years (1996-2000) may be missing a handful of symbols, and ticker
    symbols get reused by unrelated companies over multi-decade histories.

This module tries to fetch the latest version of that dataset over HTTPS
at runtime, and falls back to a bundled snapshot (data/membership/) taken
at development time if the network call fails -- which keeps the pipeline
runnable in network-restricted environments, at the cost of the fallback
being frozen as of whenever it was bundled (see snapshot file itself).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)


@dataclass
class MembershipData:
    """Point-in-time membership snapshots plus ticker active-date ranges."""

    yearend: pd.DataFrame          # columns: as_of_year_end, snapshot_date, tickers (set)
    ticker_ranges: pd.DataFrame | None  # columns: ticker, start_date, end_date
    source: str                    # "live" or "bundled-fallback"

    def members_as_of_year_end(self, year: int) -> set[str]:
        """S&P 500 constituents as of the last available snapshot on/before
        31 Dec `year`. Raises if no snapshot is available that early."""
        row = self.yearend[self.yearend["as_of_year_end"] == year]
        if row.empty:
            raise ValueError(
                f"No point-in-time membership snapshot available for year-end {year}."
            )
        return set(row.iloc[0]["tickers"])

    def members_as_of_nearest_year_end(self, year: int) -> set[str]:
        """Like members_as_of_year_end, but falls back to the nearest earlier
        year with a snapshot instead of raising (used by the crash-deployment
        rule, which can fire at an arbitrary date rather than a calendar
        year boundary)."""
        available = sorted(self.yearend["as_of_year_end"].tolist())
        candidates = [y for y in available if y <= year]
        if not candidates:
            raise ValueError(f"No point-in-time membership snapshot available at or before {year}.")
        return self.members_as_of_year_end(max(candidates))

    def is_active_on(self, ticker: str, date: pd.Timestamp) -> bool:
        """Guard against ticker-symbol reuse: True if `ticker` is documented
        to have been an actively-listed S&P 500 symbol covering `date`.
        If we have no range data for the ticker, default to True (can't
        disprove it) but this is logged upstream as a coverage gap."""
        if self.ticker_ranges is None:
            return True
        rows = self.ticker_ranges[self.ticker_ranges["ticker"] == ticker]
        if rows.empty:
            return True
        for _, r in rows.iterrows():
            start = r["start_date"]
            end = r["end_date"] if pd.notna(r["end_date"]) else pd.Timestamp.max
            if start <= date <= end:
                return True
        return False


def _parse_yearend_csv(text_or_path) -> pd.DataFrame:
    df = pd.read_csv(text_or_path)
    df["tickers"] = df["tickers"].apply(lambda s: set(t.strip() for t in s.split(",") if t.strip()))
    return df


def _build_yearend_from_full_history(full_history_csv_text: str) -> pd.DataFrame:
    """Collapse the full daily(ish) history file into one row per calendar
    year-end (nearest snapshot date on/before 31 Dec of each year)."""
    from io import StringIO

    df = pd.read_csv(StringIO(full_history_csv_text), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    rows = []
    for year in range(1997, config.END_DATE.year + 2):
        cutoff = pd.Timestamp(year=year, month=12, day=31)
        sub = df[df["date"] <= cutoff]
        if sub.empty:
            continue
        last = sub.iloc[-1]
        rows.append(
            {
                "as_of_year_end": year,
                "snapshot_date": last["date"].date().isoformat(),
                "tickers": set(t.strip() for t in last["tickers"].split(",") if t.strip()),
            }
        )
    return pd.DataFrame(rows)


def load_membership(timeout: float = 20.0) -> MembershipData:
    """Load point-in-time membership, preferring a live fetch of the
    up-to-date source dataset and falling back to the bundled snapshot."""
    ticker_ranges = None
    try:
        resp = requests.get(config.MEMBERSHIP_SOURCE_URL, timeout=timeout)
        resp.raise_for_status()
        yearend = _build_yearend_from_full_history(resp.text)
        source = "live"
        logger.info("Fetched live point-in-time membership history from %s", config.MEMBERSHIP_SOURCE_URL)
        try:
            r2 = requests.get(config.MEMBERSHIP_TICKER_RANGE_URL, timeout=timeout)
            r2.raise_for_status()
            from io import StringIO

            ticker_ranges = pd.read_csv(StringIO(r2.text), parse_dates=["start_date", "end_date"])
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not fetch ticker start/end ranges live (%s); reuse-guard disabled unless bundled copy exists.", e)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Live membership fetch failed (%s). Falling back to bundled snapshot at %s "
            "(dated data -- see file for its as-of date).",
            e,
            config.BUNDLED_YEAREND_MEMBERSHIP,
        )
        if not config.BUNDLED_YEAREND_MEMBERSHIP.exists():
            raise RuntimeError(
                "No live membership data and no bundled fallback present. "
                "Cannot proceed without point-in-time membership data "
                "(refusing to silently fall back to today's constituent list)."
            ) from e
        yearend = _parse_yearend_csv(config.BUNDLED_YEAREND_MEMBERSHIP)
        source = "bundled-fallback"

    if ticker_ranges is None and config.BUNDLED_TICKER_RANGES.exists():
        ticker_ranges = pd.read_csv(
            config.BUNDLED_TICKER_RANGES, parse_dates=["start_date", "end_date"]
        )

    return MembershipData(yearend=yearend, ticker_ranges=ticker_ranges, source=source)


def constant_membership(tickers: set[str], years: range, source_label: str = "fixed-basket") -> MembershipData:
    """Build a MembershipData whose roster is the same fixed ticker set for
    every year in `years` -- used for baskets that aren't a real
    point-in-time index (e.g. the Asia sleeve's hand-picked stock list),
    so they can reuse the same rank_year()/rank_trailing_period() machinery
    as the real S&P 500 sleeve. NOT a substitute for real point-in-time
    membership -- see config.ASIA_UNIVERSE's docstring/comment."""
    rows = [{"as_of_year_end": y, "snapshot_date": None, "tickers": set(tickers)} for y in years]
    return MembershipData(yearend=pd.DataFrame(rows), ticker_ranges=None, source=source_label)


def universe_of_all_tickers_needed(membership: MembershipData, years: range) -> set[str]:
    """Union of every ticker that appears in any year-end snapshot relevant
    to the backtest (year-1 snapshots for each contribution year), i.e. the
    full set of names we need price history for."""
    universe: set[str] = set()
    for y in years:
        try:
            universe |= membership.members_as_of_year_end(y)
        except ValueError:
            continue
    return universe
