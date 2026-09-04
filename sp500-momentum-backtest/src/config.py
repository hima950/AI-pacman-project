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


# ============================================================================
# Strategy D: "All-Weather Diversified" -- see README.md for the full sleeve
# breakdown and the assumptions made to resolve the ambiguous parts of the
# request (Asian-stock universe, the crash-deployment rule's second clause).
# ============================================================================

# Monthly $1,000 contribution split by sleeve (must sum to 1.0).
DIVERSIFIED_WEIGHTS = {
    "metals_bonds": 0.10,   # split 50/50 GLD / AGG within this sleeve
    "sp_top10": 0.20,       # reuses the existing point-in-time S&P 500 top-10 momentum basket
    "asia_top10": 0.20,     # top 10 of ASIA_UNIVERSE by trailing 1-year return, reselected annually
    "index_fund": 0.20,     # SPY (see README: used as a VOO stand-in, VOO only exists from 2010)
    "cash": 0.20,           # accrues at the risk-free rate; deployed in full on a >25% SPY crash
    "high_risk": 0.10,      # BTC-USD (see README: sits idle, uninvested, until BTC-USD price history begins in Sep 2014)
}
assert abs(sum(DIVERSIFIED_WEIGHTS.values()) - 1.0) < 1e-9

METALS_TICKER = "GLD"
BONDS_TICKER = "AGG"
HIGH_RISK_TICKER = "BTC-USD"

# Fixed, hand-picked basket of large, liquid Asia-domiciled companies
# tradeable via yfinance (mostly US-listed ADRs, plus a few major native
# listings for names without a liquid ADR). This is NOT a real point-in-time
# index -- there is no free equivalent of the S&P 500 membership dataset for
# a broad Asian index, so this list is today's well-known large-caps
# projected across the whole backtest window. Ranked by trailing 1-year
# return each year exactly like the S&P sleeve; results should be read as
# more survivorship-biased than the S&P sleeves. See DATA_SOURCES.md.
ASIA_UNIVERSE = {
    "TSM",       # Taiwan Semiconductor (ADR)
    "BABA",      # Alibaba (ADR)
    "TM",        # Toyota Motor (ADR)
    "SONY",      # Sony Group (ADR)
    "INFY",      # Infosys (ADR)
    "HDB",       # HDFC Bank (ADR)
    "IBN",       # ICICI Bank (ADR)
    "WIT",       # Wipro (ADR)
    "JD",        # JD.com (ADR)
    "BIDU",      # Baidu (ADR)
    "NTES",      # NetEase (ADR)
    "PDD",       # PDD Holdings / Pinduoduo (ADR)
    "TCEHY",     # Tencent (OTC ADR)
    "MUFG",      # Mitsubishi UFJ Financial (ADR)
    "SMFG",      # Sumitomo Mitsui Financial (ADR)
    "NMR",       # Nomura Holdings (ADR)
    "CHT",       # Chunghwa Telecom (ADR)
    "SE",        # Sea Limited (ADR, Singapore)
    "TTM",       # Tata Motors (ADR)
    "GDS",       # GDS Holdings (ADR)
    "005930.KS",  # Samsung Electronics (Korea Exchange)
    "RELIANCE.NS",  # Reliance Industries (NSE)
    "TCS.NS",    # Tata Consultancy Services (NSE)
}

CRASH_DRAWDOWN_THRESHOLD = 0.25   # SPY drawdown-from-running-peak that triggers deployment
CRASH_LOOKBACK_YEARS = 10         # "winners of the previous 10 years"
CRASH_TOP_N = 10
