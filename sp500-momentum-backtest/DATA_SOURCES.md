# Data sources and provenance

## Prices

`yfinance`, `auto_adjust=True` -> a total-return-style adjusted close (splits
*and* dividends reinvested). Raw "Close" is intentionally not used: it would
understate every strategy, since none of them would show the return
contribution of reinvested dividends. Every ticker is cached to
`data/cache/<TICKER>.parquet` after first fetch so re-runs are fast and
don't re-hit Yahoo. Every ticker that fails to fetch (after retries) is
recorded in `output/failed_tickers.csv` and logged -- never silently
dropped from the run without a trace.

## Point-in-time S&P 500 membership

**The problem this solves:** ranking "2003's best-performing S&P 500 stocks"
against *today's* constituent list silently deletes every company that was
later removed from the index (acquired, delisted, relegated for falling
market cap). That inflates momentum-style strategies, sometimes a lot,
because the losers that would have dragged down the basket have been
edited out of the eligible universe after the fact.

**What this project uses:** [`fja05680/sp500`](https://github.com/fja05680/sp500)
on GitHub, MIT licensed (full license text bundled at
`data/membership/UPSTREAM_LICENSE_fja05680_sp500.txt`). It's a
community-maintained, dated reconstruction of S&P 500 membership from
1996 to the present, built by:

1. Starting from a base historical file spanning 1996-2019 that the
   maintainer sourced from a dataset distributed with Andreas Clenow's
   *Trading Evolved* (used there for the same survivorship-bias problem).
2. Extending it forward by tracking Wikipedia's "Selected changes to the
   list of S&P 500 components" table since 2019, cross-checked against
   Wikipedia's current constituent list, with gaps filled in by the
   maintainer researching individual corporate-action histories where
   Wikipedia's "selected" (not exhaustive) changes table was incomplete.

This is functionally very close to approach **(2)** from the task brief --
"reconstruct membership backwards from the current list plus the
documented history of index additions and removals" -- except pre-built by
a third party rather than assembled by this code parsing Wikipedia
directly. **It is not an official vendor point-in-time feed** (e.g.
S&P Dow Jones Indices' own data, or CRSP) -- approach **(1)** from the
brief. Treat it as good-quality but not authoritative.

### Known limitations (per the upstream maintainer, and independently worth flagging)

- **Early years (1996-2000) may be missing a handful of symbols.** The
  first snapshot has 487-488 names instead of ~500; coverage stabilizes
  above 494 from 2001 onward. This project's backtest starts in January
  2000, so the January-2000 basket (ranked on 1999 returns) is the vintage
  most exposed to this; every year from 2001 onward is on firmer ground.
- **Wikipedia's changes table is "selected," not exhaustive.** The
  maintainer fills gaps by manual research, which is inherently
  best-effort rather than a guaranteed complete audit trail.
- **Ticker symbol reuse.** The same ticker can be recycled by an unrelated
  company years apart (e.g. a delisted company's old symbol reassigned to
  a new listing). This project cross-references
  `data/membership/sp500_ticker_start_end.csv` (also from the same
  upstream source) to guard against treating two different companies'
  price histories as one continuous series when ranking -- see
  `MembershipData.is_active_on()` in `src/membership.py`. This guard is
  necessarily heuristic, not a certainty.
- **Price histories truncated at delisting.** `yfinance`/Yahoo Finance
  generally stops returning data once a ticker is delisted or the company
  is acquired, which is exactly the signal this project uses to detect
  "sell at the last available price" events (see `README.md` ->
  Portfolio mechanics). It means we cannot distinguish "delisted the day
  after the last observed price" from "delisted weeks later"; the
  approximation is to act on the last observed price, per the brief.

### How the code uses this

`src/membership.py` tries a **live** fetch of the upstream CSV
(`https://raw.githubusercontent.com/fja05680/sp500/master/...`) on every
run, so the membership history stays current. If that fetch fails (e.g. no
network, as in the sandbox this project was developed in -- see
`README.md`), it falls back to a **bundled snapshot**
(`data/membership/sp500_membership_yearend.csv`, one row per calendar
year-end from 1997 to whenever it was extracted) taken at development
time. The code refuses to run at all rather than silently substituting
today's S&P 500 roster if neither is available -- see the
`RuntimeError` in `load_membership()`.

The bundled snapshot only stores one row **per year-end** (not the full
daily history) because that's all this backtest actually needs: eligibility
for year Y's basket is decided once, from the 31-Dec-(Y-1) snapshot.

## Strategy D's additional tickers

`GLD`, `AGG`, and `BTC-USD` are standard `yfinance` tickers (SPDR Gold
Shares, iShares Core U.S. Aggregate Bond ETF, and Coinbase-sourced
BTC-USD spot) fetched the same way as every other price series -- no
special provenance concerns beyond the usual adjusted-close caveat.

The Asia sleeve's `config.ASIA_UNIVERSE` is **not** sourced from any
index-membership dataset at all -- there is no free equivalent of the
S&P 500 point-in-time dataset above for a broad Asian index. It's a
hand-picked, fixed list of large, liquid Asia-domiciled companies
tradeable via `yfinance`, chosen for name recognition and data
availability, not for representativeness of any particular index. See
`README.md` -> "Strategy D" -> "The Asia sleeve, honestly" for the full
caveat.
