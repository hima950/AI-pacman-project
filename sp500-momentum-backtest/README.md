# S&P 500 momentum backtest: Index vs. Top-5 vs. Top-10

Compares three $1,000/month dollar-cost-averaging strategies, January 2000
through the most recently completed month:

- **A -- Index**: buy SPY every month.
- **B -- Top 5**: each January, take the 5 best-performing S&P 500
  constituents of the *prior* calendar year (as it actually looked back
  then, not today's roster) and split that year's monthly contributions
  equally among them.
- **C -- Top 10**: same, with 10 names.

Each strategy is run under two portfolio mechanics (see below): buy-and-hold
and annual-rebalance-with-tax.

## /!\ Read this before trusting any numbers from this checkout

**This code was developed inside a network-restricted sandbox that cannot
reach Yahoo Finance or Wikipedia** (outbound HTTPS is limited to GitHub and
package registries there). That means:

- `python main.py` has **not been run against real market data** in this
  environment -- it cannot be, here.
- Every mechanic (ranking, slippage, tax, delisting/redistribution,
  IPO-exclusion, XIRR/TWR/drawdown math) **has** been validated end-to-end
  against synthetic, clearly-fake data (`tests/test_synthetic.py`) that
  exercises the exact same code path. That proves the *mechanics* are
  correct; it says nothing about what the real S&P 500 numbers are.
- The one real (non-price) dataset used, point-in-time S&P 500 membership,
  was fetched successfully from GitHub during development (see
  `DATA_SOURCES.md`) since GitHub *is* reachable there, and a snapshot is
  bundled in `data/membership/` as a fallback.

**To get real results:** run `python main.py` on any machine with normal
internet access (your laptop, a CI runner, etc.). It needs outbound HTTPS
to Yahoo Finance (via `yfinance`) and, ideally, to
`raw.githubusercontent.com` (for a live membership-history refresh; falls
back to the bundled snapshot if that's unreachable too). A full first run
fetches price history for every ticker that has ever been in the S&P 500
since 1999 (~1,000+ symbols) -- expect it to take a while; every result is
cached to `data/cache/*.parquet` so subsequent runs are fast.

```bash
pip install -r requirements.txt
python main.py                      # full backtest against real data
python tests/test_synthetic.py      # mechanics self-test, no network needed
```

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

## Outputs (`output/`)

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

`interpretation.md` (also printed at the end of `main.py`) is generated
*from* the computed metrics/transactions each run -- including a
concentration check (`src/interpretation.py::concentration_summary`) that
flags when a momentum basket's result is actually driven by one or two
names, per the brief's request to check for this rather than assume it
away.

## Repo layout

```
main.py                  orchestration: fetch -> rank -> simulate -> report
src/config.py             all tunable parameters in one place
src/membership.py         point-in-time S&P 500 membership (see DATA_SOURCES.md)
src/data_fetch.py         yfinance + parquet cache, failure logging
src/ranking.py            prior-calendar-year momentum ranking
src/portfolio.py          the simulation engine (both variants)
src/metrics.py            XIRR, TWR, vol, Sharpe, Sortino, drawdown, ...
src/reporting.py          tables, plots, CSV export
src/interpretation.py     auto-generated written read of the results
tests/test_synthetic.py   mechanics self-test against fabricated data (no network)
data/membership/          bundled point-in-time membership snapshot + license
data/cache/                parquet price cache (gitignored, populated at runtime)
output/                    all generated reports (gitignored; regenerate via main.py)
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
