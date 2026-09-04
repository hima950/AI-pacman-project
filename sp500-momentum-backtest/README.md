# S&P 500 momentum backtest: Index vs. Top-5 vs. Top-10 vs. Diversified

Compares four $1,000/month dollar-cost-averaging strategies, January 2000
through the most recently completed month:

- **A -- Index**: buy SPY every month.
- **B -- Top 5**: each January, take the 5 best-performing S&P 500
  constituents of the *prior* calendar year (as it actually looked back
  then, not today's roster) and split that year's monthly contributions
  equally among them.
- **C -- Top 10**: same, with 10 names.
- **D -- Diversified**: a six-sleeve all-weather allocation (10% metals &
  bonds / 20% S&P top-10 / 20% Asia top-10 / 20% index fund / 20% cash
  deployed on a >25% crash / 10% BTC) -- see "Strategy D" below for the
  full breakdown. Run via `python main_diversified.py` (separate entry
  point / separate outputs from A-C's `main.py`).

Strategies A-C are each run under two portfolio mechanics (see below):
buy-and-hold and annual-rebalance-with-tax. Strategy D is buy-and-hold
only (no annual-rebalance/tax variant was requested for it).

## /!\ Read this before trusting any numbers from this checkout

**This code was developed inside a network-restricted sandbox that cannot
reach Yahoo Finance or Wikipedia** (outbound HTTPS is limited to GitHub and
package registries there). That means:

- Neither `python main.py` nor `python main_diversified.py` has **been run
  against real market data** in this environment -- it cannot be, here.
- Every mechanic (ranking, slippage, tax, delisting/redistribution,
  IPO-exclusion, XIRR/TWR/drawdown math, and for Strategy D specifically:
  crash detection/re-arming, trailing-10-year ranking, cash accrual, the
  6-sleeve combiner's totals) **has** been validated end-to-end against
  synthetic, clearly-fake data (`tests/test_synthetic.py`,
  `tests/test_diversified_synthetic.py`) that exercises the exact same
  code path, including a full smoke test of every plot/CSV export. That
  proves the *mechanics* are correct; it says nothing about what the real
  numbers are.
- The one real (non-price) dataset used, point-in-time S&P 500 membership,
  was fetched successfully from GitHub during development (see
  `DATA_SOURCES.md`) since GitHub *is* reachable there, and a snapshot is
  bundled in `data/membership/` as a fallback.

**To get real results:** run `python main.py` and/or `python
main_diversified.py` on any machine with normal internet access (your
laptop, a CI runner, etc.). Both need outbound HTTPS to Yahoo Finance (via
`yfinance`) and, ideally, to `raw.githubusercontent.com` (for a live
membership-history refresh; falls back to the bundled snapshot if that's
unreachable too). A full first run of `main.py` fetches price history for
every ticker that has ever been in the S&P 500 since 1999 (~1,000+
symbols); `main_diversified.py` fetches that same universe plus the Asia
basket, GLD, AGG, and BTC-USD, over a wider 1990-present window (the
crash rule needs up to 10 years of *trailing* history as of an arbitrary
crash date) -- expect either to take a while on a first run; every result
is cached to `data/cache/*.parquet` so subsequent runs are fast.

```bash
pip install -r requirements.txt
python main.py                              # Strategies A/B/C against real data
python main_diversified.py                  # Strategy D vs. A/B/C, real data + comparison plots

pip install -r requirements-dev.txt
pytest tests/                                # everything below, one command, no network needed
python tests/test_synthetic.py              # (equivalently, standalone) A/B/C mechanics self-test
python tests/test_diversified_synthetic.py  # (equivalently, standalone) Strategy D mechanics self-test
```

## Testing

Two layers, both network-free (synthetic/fabricated data only) and both
run by a plain `pytest tests/`:

- **`tests/unit/`** -- isolated, hand-computable unit tests for individual
  functions: XIRR against a closed-form compounding example, TWR/vol/
  Sharpe/Sortino/drawdown against manually-derived expected values,
  ranking's no-look-ahead and partial-year-exclusion rules, membership
  lookup/fallback/ticker-reuse logic, the `_Book` ledger's slippage and
  tax arithmetic in isolation, cash-sleeve interest accrual against an
  exact compounding formula, and the concentration diagnostic. 45 tests.
- **`tests/test_synthetic.py`** and **`tests/test_diversified_synthetic.py`**
  -- whole-pipeline scenarios against fabricated tickers with engineered
  edge cases (an IPO mid-year, a delisting mid-year, a name that delists
  *before* it's ever bought, an asset that starts trading partway through
  the backtest, an engineered >25% crash with re-arming). These predate
  pytest adoption and use a `check()`/print pattern for a readable
  standalone transcript (`python tests/test_synthetic.py`); a small
  `pytest_checked` decorator makes a failed `check()` also fail the test
  under plain `pytest` (verified by deliberately breaking a check and
  confirming pytest reports the failure, then reverting -- not just
  assumed). 6 tests, covering everything the unit tests don't: end-to-end
  wiring, the delisting-redistribution mechanic, the 6-sleeve combiner's
  totals reconciling, and a full smoke test of every plot/CSV export.

All 51 tests currently pass. Two of them print a benign
`PytestReturnNotNoneWarning` (they double as pytest tests and as
data-preparing helpers called by a later test, so they legitimately
return a value) -- not a failure, just pytest noting the pattern.

This layer proves the *mechanics* are correct; it cannot validate the
real S&P 500 numbers, since this sandbox can't reach Yahoo Finance or
Wikipedia (see the warning above).

## Methodology

### Ranking rules

For year Y's basket: rank every stock that was an S&P 500 constituent **as
of 31 Dec Y-1** by total return over calendar year Y-1, computed only from
price data dated on or before 31 Dec Y-1 (no look-ahead). A stock needs a
full year of price history in Y-1 to be ranked -- this naturally excludes
mid-year IPOs, whose "return since IPO" would be measured over a
shorter, non-comparable window. Dropped-name counts are logged per year and
exported to `output/annual_baskets.csv` (`n_dropped_partial_year`,
`n_missing_price_data`).

### Survivorship bias

Handled by using dated, point-in-time membership snapshots (not today's
roster) to decide eligibility for each year's ranking -- see
`DATA_SOURCES.md` for the full data-provenance writeup, including the
known limitations of the underlying dataset and why it's labeled as a
community reconstruction rather than an official point-in-time feed.

### Portfolio mechanics

**Variant 1 -- buy and hold.** Each month's $1,000 buys that calendar
year's basket, split equally by dollar amount. Nothing is ever sold except
a forced sale when a holding is delisted/acquired mid-year, at its last
available price; the (slippage-adjusted) proceeds are immediately
redistributed equally across the surviving members of the basket(s) that
name belonged to (falling back to the whole live portfolio if none
survive). No tax is modeled in this variant -- there's no year in which
gains are systematically realized, so nothing triggers a tax event.

**Variant 2 -- annual rebalance.** Every January (first trading day),
every current holding is fully liquidated -- 5bp slippage on the sale, and
15% tax on the *realized gain* only (average-cost basis; realized losses
are not credited/rebated) -- and the after-tax proceeds plus that month's
$1,000 are redeployed equally across the new basket. Forced delisting
sales during the year are taxed the same way, since this variant's whole
premise is realizing gains annually.

**Strategy A (SPY) is identical under both variants** -- there is no
"new basket" to rebalance into (it's always just SPY), so forcing an
annual sell/rebuy of the same single position would only add slippage
drag for no reason. The code special-cases this (`single_asset_no_rebalance`
in `src/portfolio.py`) rather than mechanically applying the rebalance
step; both columns are reported and will match exactly.

**Both variants:** fractional shares, zero commissions, 5bp slippage on
every buy and every sell (modeled as a price shift against the trader:
`fill_price = quote * (1 +/- 0.0005)`, not a separate fee line).

### Metrics

- **XIRR (money-weighted)** is the headline return number -- correct here
  because contributions are spread over 26 years; a naive CAGR on
  final-vs-first value would be badly wrong.
- **Time-weighted return (TWR)** is computed from monthly sub-period
  returns `r_m = V_m / (V_{m-1} + C_m) - 1` (exact, not approximate, since
  every contribution lands on the first trading day of its period).
  Annualized vol, Sharpe, Sortino, drawdown, and best/worst calendar year
  are all computed from this TWR series (i.e. they measure the *strategy's*
  return behavior, not the investor's contribution-timing luck).
- **Max drawdown** peak/trough dates are read off the TWR growth-of-$1
  index, at month-end resolution.
- **Sharpe/Sortino** use `^IRX` (13-week T-bill) as the risk-free rate,
  monthly rate = `(annual yield / 100) / 12`; falls back to a flat 2%/yr if
  `^IRX` can't be fetched.
- **Turnover** is reported per strategy/variant as average annual
  (sell-dollar-volume / average portfolio value that year).

## Strategy D: "All-Weather Diversified"

Same $1,000/month contribution as A-C, split across six sleeves every
month (`src/config.py::DIVERSIFIED_WEIGHTS`):

| Sleeve | Weight | What it buys |
|---|---|---|
| Metals & bonds | 10% | 50/50 GLD (gold) / AGG (aggregate bond ETF) |
| S&P top 10 | 20% | Same point-in-time momentum basket as Strategy C |
| Asia top 10 | 20% | Top 10 of a fixed Asia stock/ADR basket by trailing 1-year return (see below -- **not** a real point-in-time index) |
| Index fund | 20% | SPY, standing in for Vanguard's VOO (VOO only exists from Sep 2010; SPY tracks the same index for the full 2000-present window) |
| Cash | 20% | Accrues at the risk-free rate (`^IRX`); deployed in full the first time SPY closes >25% below its running all-time high (see below) |
| High risk / high reward | 10% | BTC-USD; sits idle, uninvested, until BTC-USD price history begins (~Sep 2014), then lump-sum invests the backlog and continues normally |

Every sleeve is buy-and-hold: nothing is ever sold except the same forced
delisting/acquisition handling used in Strategies B/C, and (for the S&P
and Asia sleeves) the same annual reselection of that year's basket. Five
of the six sleeves are literally the same `portfolio.simulate()` engine
used for Strategies A-C, just fed a different monthly dollar amount and
basket -- see `src/diversified.py`. Only the cash/crash sleeve needed new
mechanics (`src/crash_sleeve.py`), since "accrue interest until a
market-wide trigger fires, then lump-sum deploy" doesn't fit the
single-basket-per-year shape at all.

### The crash-deployment rule

On the first trading day SPY's closing price is more than 25% below its
running all-time high, the *entire* accumulated cash sleeve is deployed in
one shot into the top 10 S&P 500 names by trailing 10-year total return as
of that date (`ranking.rank_trailing_period`), split equally and bought
with the same 5bp slippage used everywhere else. The positions bought are
then held forever (this sleeve doesn't rebalance itself further). The
trigger then **disarms** until SPY makes a new all-time high (a full
recovery), after which a fresh >25% drawdown from that new peak fires it
again -- so a single prolonged bear market only triggers one deployment,
not one per day, but a multi-decade backtest can see multiple independent
deployments (e.g. 2008-09, 2020).

If fewer than 10 S&P names have a full 10-year trailing history as of the
trigger date (only possible for a crash early in the backtest), the
lookback window is shortened year-by-year (down to a 3-year floor) rather
than deploying into an artificially tiny or empty basket; this is logged
and recorded in `output/diversified_crash_events.csv`.

**On the ambiguous part of the original request:** "deploy cash into
winners of the previous 10 years and the [garbled] afar crash" was
interpreted, per your direction, as trailing-10-year winners only (the
second clause dropped rather than guessed at).

### The Asia sleeve, honestly

There is no free, point-in-time constituent dataset for a broad Asian
index the way `fja05680/sp500` covers the S&P 500 (see `DATA_SOURCES.md`).
Per your direction, this sleeve instead uses a fixed, hand-picked basket
of ~23 large, liquid Asia-domiciled companies tradeable via `yfinance`
(`src/config.py::ASIA_UNIVERSE` -- mostly US-listed ADRs: TSM, BABA, TM,
SONY, INFY, JD, BIDU, TCEHY, etc., plus a few major native listings like
Samsung and Reliance), ranked by trailing 1-year return each year exactly
like the S&P sleeve. **This is today's well-known large-caps projected
across the whole backtest window, not a real point-in-time index** -- it
is more survivorship-biased than the S&P sleeves, and the exact
membership is a judgment call, not a sourced fact. Treat the Asia sleeve's
numbers as illustrative, not authoritative.

### Cash accrual and the pre-2014 crypto sleeve, as assumptions

- Idle cash (both the dedicated cash sleeve pre-crash, and any sleeve's
  temporarily-uninvestable contributions, e.g. BTC-USD before it existed)
  earns **0%** while genuinely un-investable (no asset to buy yet) but the
  *dedicated* cash sleeve specifically accrues at the risk-free rate
  month over month while waiting for a crash trigger -- these are
  different cases handled differently on purpose: one is "nothing to buy
  yet" (crypto), the other is "deliberately holding cash as an asset
  class" (the cash sleeve). Neither was fully specified in the brief;
  both are reasonable, clearly-logged defaults.
- BTC-USD's entire pre-Sep-2014 contribution backlog invests as one
  lump sum the first month price data exists, which shows up as a visible
  step in `diversified_sleeve_composition.png`. A real investor could
  instead choose to skip that sleeve entirely pre-2014; this project
  keeps the money "reserved for BTC" rather than silently redirecting it,
  since redirecting it would be a bigger, unrequested behavior change.

## Outputs (`output/`)

### From `main.py` (Strategies A/B/C)

| File | Contents |
|---|---|
| `summary_metrics.csv` | One row per strategy x variant, every metric above |
| `portfolio_value_timeseries.csv` | Month-end value, all strategies/variants |
| `annual_baskets.csv` | Every basket pick, every year, with its ranking return (eyeball-check the picks) |
| `transactions.csv` | Full transaction log (buys/sells, slippage, tax, reason) |
| `failed_tickers.csv` | Every ticker that failed to fetch, and why |
| `portfolio_value_log.png` | Value over time, log scale, all strategies + cumulative-contributions line |
| `drawdowns.png` | Drawdown chart, all strategies overlaid |
| `interpretation.md` | Auto-generated, numbers-driven read of the results (see below) |
| `run.log` | Full run log, including every failed-fetch warning |

### From `main_diversified.py` (Strategy D vs. A/B/C)

Same `summary_metrics.csv` / `portfolio_value_timeseries.csv` /
`annual_baskets.csv` / `transactions.csv` / `failed_tickers.csv` shape as
above (this run overwrites those with the 4-strategy comparison), plus:

| File | Contents |
|---|---|
| `diversified_value_log.png` | Value over time, log scale -- Index/Top5/Top10/Diversified + cumulative contributions |
| `diversified_drawdowns.png` | Drawdown chart, all four overlaid |
| `diversified_sleeve_composition.png` | Stacked-area chart of Strategy D's six sleeves over time, with crash-deployment dates marked |
| `diversified_allocation_pie.png` | Target monthly-contribution allocation (the 10/20/20/20/20/10 split) |
| `diversified_metric_comparison.png` | Grouped bar chart: XIRR, Sharpe, max drawdown across all four strategies |
| `diversified_sleeve_breakdown.csv` | Final value and total contributed, per sleeve |
| `diversified_crash_events.csv` | Every crash-deployment event: date, SPY drawdown, cash deployed, lookback years used, winners bought |
| `interpretation_diversified.md` | Auto-generated read of the 4-strategy comparison |
| `run_diversified.log` | Full run log for this script |

`interpretation.md` / `interpretation_diversified.md` are generated *from*
the computed metrics/transactions each run -- including a concentration
check (`src/interpretation.py::concentration_summary`) that flags when a
momentum basket's result is actually driven by one or two names, per the
original brief's request to check for this rather than assume it away.

## Repo layout

```
main.py                  orchestration: fetch -> rank -> simulate -> report
src/config.py             all tunable parameters in one place
src/membership.py         point-in-time S&P 500 membership (see DATA_SOURCES.md)
src/data_fetch.py         yfinance + parquet cache, failure logging
src/ranking.py            prior-calendar-year AND trailing-N-year momentum ranking
src/portfolio.py          the single-basket simulation engine (both variants)
src/crash_sleeve.py       cash-accrual + crash-triggered deployment sleeve (Strategy D only)
src/diversified.py        Strategy D's 6-sleeve composer (reuses src/portfolio.py x5 + src/crash_sleeve.py)
src/metrics.py            XIRR, TWR, vol, Sharpe, Sortino, drawdown, ...
src/reporting.py          tables, plots, CSV export
src/interpretation.py     auto-generated written read of the results
tests/unit/                          isolated, hand-computable unit tests (metrics, ranking, membership, _Book, crash accrual, interpretation)
tests/test_synthetic.py              A/B/C whole-pipeline scenario tests against fabricated data (no network)
tests/test_diversified_synthetic.py  Strategy D whole-pipeline scenario tests (crash detection, re-arming, sleeve totals)
tests/conftest.py                    pytest sys.path setup shared by every test file
data/membership/          bundled point-in-time S&P 500 membership snapshot + license
data/cache/                parquet price cache (gitignored, populated at runtime)
output/                    all generated reports (gitignored; regenerate via main.py / main_diversified.py)
```

## Assumptions worth double-checking against your own use case

- Average-cost basis for realized-gain/tax calculations (not FIFO/LIFO --
  the brief didn't specify a lot-accounting method).
- Sortino's minimum acceptable return is 0% monthly (not the risk-free
  rate) -- a common simplification, adjustable in `src/metrics.py`.
- Delisting detection is "ticker has no more price data after date X" --
  approximates the true delisting date but can't distinguish a stock that
  delisted the next day from one that delisted weeks later, since Yahoo
  Finance itself stops returning history at that point.
- "Most recent complete month" is computed dynamically from the run date
  (`src/config.py::most_recent_complete_month_end`), not hardcoded.
- Strategy D's Asia sleeve and crash-deployment rule carry their own,
  separately-documented assumptions -- see "Strategy D" above.
