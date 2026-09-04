"""Adjusted-close price downloader with local parquet caching.

Uses yfinance with auto_adjust=True, which returns prices adjusted for both
splits and dividends (i.e. a total-return-style series) -- using the raw
"Close" instead would understate every strategy's return, since none of the
capital gain from reinvested dividends would show up. Every ticker that
fails to fetch is logged (not silently dropped): see `FailedFetch` records
and `output/failed_tickers.csv` written by main.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


@dataclass
class FailedFetch:
    ticker: str
    reason: str


@dataclass
class FetchResult:
    prices: dict[str, pd.Series] = field(default_factory=dict)  # ticker -> adj close Series
    failures: list[FailedFetch] = field(default_factory=list)


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return config.CACHE_DIR / f"{safe}.parquet"


def _load_cache(ticker: str) -> pd.Series | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        s = df["adj_close"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:  # noqa: BLE001
        logger.warning("Cache read failed for %s (%s); will re-fetch.", ticker, e)
        return None


def _save_cache(ticker: str, series: pd.Series) -> None:
    df = series.rename("adj_close").to_frame()
    df.to_parquet(_cache_path(ticker))


def _download_one(
    yf_symbol: str, start: str, end: str, retries: int, backoff: float
) -> pd.Series | None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                yf_symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError("empty response")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):  # multiindex edge case
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                raise ValueError("all-NaN close series")
            close.index = pd.to_datetime(close.index)
            return close.sort_index()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    logger.warning("Failed to fetch %s after %d attempts: %s", yf_symbol, retries, last_err)
    return None


def fetch_adjusted_close(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
    retries: int = 3,
    backoff: float = 1.5,
) -> FetchResult:
    """Fetch adjusted close series for each ticker, using and refreshing the
    on-disk parquet cache. Returns both the successfully fetched series and
    a list of every ticker that failed (never silently skipped)."""
    if yf is None:
        raise RuntimeError("yfinance is not installed. `pip install yfinance`.")

    result = FetchResult()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    for raw_ticker in tickers:
        yf_symbol = config.to_yfinance_symbol(raw_ticker)
        cached = _load_cache(raw_ticker) if use_cache else None

        needs_fetch = True
        if cached is not None and not cached.empty:
            covers_start = cached.index.min() <= start_ts + pd.Timedelta(days=7)
            covers_end = cached.index.max() >= min(end_ts, pd.Timestamp.today()) - pd.Timedelta(days=7)
            if covers_start and covers_end:
                needs_fetch = False

        if not needs_fetch:
            result.prices[raw_ticker] = cached[(cached.index >= start_ts) & (cached.index <= end_ts)]
            continue

        series = _download_one(yf_symbol, start, end, retries, backoff)
        if series is None:
            if cached is not None and not cached.empty:
                logger.warning("Using stale cache for %s after live fetch failure.", raw_ticker)
                result.prices[raw_ticker] = cached[(cached.index >= start_ts) & (cached.index <= end_ts)]
            else:
                result.failures.append(FailedFetch(raw_ticker, "download failed after retries"))
            continue

        if cached is not None and not cached.empty:
            merged = pd.concat([cached, series])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = series

        _save_cache(raw_ticker, merged)
        result.prices[raw_ticker] = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]

    return result
