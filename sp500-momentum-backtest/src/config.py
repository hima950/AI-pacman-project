"""Central configuration for the SP500 momentum backtest."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

# --- paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
MEMBERSHIP_DIR = DATA_DIR / "membership"
OUTPUT_DIR = ROOT_DIR / "output"

for _d in (CACHE_DIR, MEMBERSHIP_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- backtest window -----------------------------------------------------
START_DATE = _dt.date(2000, 1, 1)


def most_recent_complete_month_end(today: _dt.date | None = None) -> _dt.date:
    """Last calendar day of the most recently *completed* month before today."""
    today = today or _dt.date.today()
    first_of_this_month = today.replace(day=1)
    return first_of_this_month - _dt.timedelta(days=1)


END_DATE = most_recent_complete_month_end()

# --- strategy parameters --------------------------------------------------
MONTHLY_CONTRIBUTION = 1000.0
TOP_N_STRATEGIES = {"top5": 5, "top10": 10}
INDEX_TICKER = "SPY"
RISK_FREE_TICKER = "^IRX"          # 13-week T-bill discount rate, annualized %
RISK_FREE_FLAT_FALLBACK = 0.02     # used only if ^IRX cannot be fetched

SLIPPAGE_BPS = 5.0                 # 5 bp = 0.05% on every buy and every sell
CAPITAL_GAINS_TAX_RATE = 0.15      # applied only under the annual-rebalance variant

# A ticker must have at least this many trading days of price history
# ending on/before 31-Dec of year Y-1 AND beginning on/before the first
# trading day of Y-1 to be considered "a full year of price history".
MIN_TRADING_DAYS_FOR_FULL_YEAR = 200

# --- membership data source ----------------------------------------------
# Community-maintained, MIT-licensed reconstruction of point-in-time S&P 500
# membership (sourced originally from Wikipedia's "Selected changes" table,
# cross-checked and extended by the maintainer). This is NOT an official
# vendor point-in-time feed (e.g. S&P/CRSP) -- see README.md / DATA_SOURCES.md
# for the full provenance and known limitations discussion.
MEMBERSHIP_SOURCE_REPO = "fja05680/sp500"
MEMBERSHIP_SOURCE_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
MEMBERSHIP_SOURCE_URL = (
    f"https://raw.githubusercontent.com/{MEMBERSHIP_SOURCE_REPO}/master/"
    + MEMBERSHIP_SOURCE_FILE.replace("&", "%26").replace(" ", "%20")
)
MEMBERSHIP_TICKER_RANGE_URL = (
    f"https://raw.githubusercontent.com/{MEMBERSHIP_SOURCE_REPO}/master/"
    "sp500_ticker_start_end.csv"
)

# Bundled fallback snapshots (extracted at development time from the same
# upstream source -- see data/membership/UPSTREAM_LICENSE_fja05680_sp500.txt)
BUNDLED_YEAREND_MEMBERSHIP = MEMBERSHIP_DIR / "sp500_membership_yearend.csv"
BUNDLED_TICKER_RANGES = MEMBERSHIP_DIR / "sp500_ticker_start_end.csv"

# Ticker symbol cleanup: yfinance uses '-' where these datasets use '.'
# (e.g. BRK.B -> BRK-B, BF.B -> BF-B).
def to_yfinance_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")
