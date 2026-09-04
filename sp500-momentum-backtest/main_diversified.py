#!/usr/bin/env python3
"""Strategy D ("All-Weather Diversified") vs. the three original strategies.

Adds a fourth, six-sleeve strategy on top of main.py's Index/Top5/Top10:
    10% precious metals & bonds, 20% S&P top-10 momentum, 20% Asia top-10
    momentum (fixed basket, see README/DATA_SOURCES), 20% index fund (SPY
    standing in for VOO), 20% cash deployed on a >25% SPY crash into
    trailing-10-year winners, 10% BTC-USD.

Run with: python main_diversified.py
Same network requirements and caveats as main.py -- see README.md.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

from src import config, data_fetch, diversified, interpretation, membership, metrics as metrics_mod, portfolio, ranking, reporting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(config.OUTPUT_DIR / "run_diversified.log", mode="w")],
)
logger = logging.getLogger("main_diversified")


def build_risk_free_monthly(irx_series: pd.Series | None, month_end_dates) -> float | pd.Series:
    if irx_series is None or irx_series.empty:
        logger.warning("^IRX unavailable; falling back to flat %.1f%% risk-free rate.", config.RISK_FREE_FLAT_FALLBACK * 100)
        return config.RISK_FREE_FLAT_FALLBACK / 12.0
    vals = {}
    for d in month_end_dates:
        sub = irx_series[irx_series.index <= d]
        if sub.empty:
            continue
        vals[d] = (float(sub.iloc[-1]) / 100.0) / 12.0
    return pd.Series(vals)


def main() -> None:
    start_ts = pd.Timestamp(config.START_DATE)
    end_ts = pd.Timestamp(config.END_DATE)
    logger.info("Backtest window: %s to %s", start_ts.date(), end_ts.date())

    logger.info("Loading point-in-time S&P 500 membership...")
    sp_mem = membership.load_membership()
    logger.info("S&P membership source: %s", sp_mem.source)

    ranking_years = range(1999, end_ts.year)
    holding_years = range(2000, end_ts.year + 1)

    sp_universe = membership.universe_of_all_tickers_needed(sp_mem, ranking_years)
    asia_mem = membership.constant_membership(config.ASIA_UNIVERSE, range(ranking_years.start, holding_years.stop))
    asia_universe = set(config.ASIA_UNIVERSE)

    extra_tickers = {config.INDEX_TICKER, config.METALS_TICKER, config.BONDS_TICKER, config.HIGH_RISK_TICKER}
    full_universe = sorted(sp_universe | asia_universe | extra_tickers)
    logger.info(
        "Fetching %d tickers (%d S&P, %d Asia, %d metals/bonds/index/crypto)...",
        len(full_universe), len(sp_universe), len(asia_universe), len(extra_tickers),
    )

    # Wider fetch window than main.py: the crash-deployment rule needs up
    # to 10 years of *trailing* price history as of an arbitrary crash
    # date, so an early crash (e.g. 2008) needs data reaching back further
    # than the 2000-basket-selection buffer main.py uses.
    fetch_start = "1990-01-01"
    fetch_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    fetch_result = data_fetch.fetch_adjusted_close(full_universe, fetch_start, fetch_end)
    logger.info("Fetched %d/%d tickers; %d failed.", len(fetch_result.prices), len(full_universe), len(fetch_result.failures))
    for f in fetch_result.failures:
        logger.warning("FAILED FETCH: %s (%s)", f.ticker, f.reason)

    all_prices = dict(fetch_result.prices)
    if config.INDEX_TICKER not in all_prices:
        raise RuntimeError("Could not fetch SPY -- cannot proceed (calendar reference + index sleeve).")
    spy_prices = all_prices[config.INDEX_TICKER]

    irx_result = data_fetch.fetch_adjusted_close([config.RISK_FREE_TICKER], fetch_start, fetch_end)
    irx_series = irx_result.prices.get(config.RISK_FREE_TICKER)

    logger.info("Ranking S&P and Asia top-10/top-5 baskets for %s-%s...", holding_years.start, holding_years.stop - 1)
    sp_rankings = ranking.rank_all_years(holding_years, sp_mem, all_prices, [5, 10])
    asia_rankings = ranking.rank_all_years(holding_years, asia_mem, all_prices, [10])

    basket_index = {y: [config.INDEX_TICKER] for y in holding_years}
    basket_top5 = {y: sp_rankings[y].basket[5] for y in sp_rankings if 5 in sp_rankings[y].basket}
    basket_top10 = {y: sp_rankings[y].basket[10] for y in sp_rankings if 10 in sp_rankings[y].basket}

    logger.info("Simulating Index / Top5 / Top10 (buy & hold)...")
    results: dict[str, portfolio.SimulationResult] = {}
    results["Index (SPY)"] = portfolio.simulate(
        "index", "buy_and_hold", basket_index, all_prices, spy_prices,
        start=start_ts, end=end_ts, single_asset_no_rebalance=True,
    )
    results["Top 5"] = portfolio.simulate(
        "top5", "buy_and_hold", basket_top5, all_prices, spy_prices, start=start_ts, end=end_ts,
    )
    results["Top 10"] = portfolio.simulate(
        "top10", "buy_and_hold", basket_top10, all_prices, spy_prices, start=start_ts, end=end_ts,
    )

    month_end_dates = results["Index (SPY)"].monthly_value.index
    rf_monthly = build_risk_free_monthly(irx_series, month_end_dates)

    logger.info("Simulating Diversified (6 sleeves)...")
    div = diversified.build_diversified_strategy(
        holding_years=holding_years,
        membership=sp_mem,
        all_prices=all_prices,
        spy_prices=spy_prices,
        sp_top10_basket_by_year=basket_top10,
        asia_rankings=asia_rankings,
        start=start_ts,
        end=end_ts,
        risk_free_monthly=rf_monthly,
    )
    results["Diversified"] = div.combined

    for e in div.crash_events:
        logger.info(
            "CRASH EVENT %s: SPY drawdown %.1f%%, deployed $%.0f into %s",
            e.date.date(), e.spy_drawdown * 100, e.cash_deployed, e.winners,
        )

    logger.info("Computing metrics...")
    all_metrics = {label: metrics_mod.compute_metrics(res, rf_monthly) for label, res in results.items()}
    summary_df = reporting.summary_table(list(all_metrics.values()))

    basket_df = reporting.per_year_basket_table({"top5": sp_rankings, "top10": sp_rankings, "asia_top10": asia_rankings})

    idx_res = results["Index (SPY)"]
    cum_contrib = pd.Series(
        [config.MONTHLY_CONTRIBUTION * (i + 1) for i in range(len(idx_res.monthly_value))],
        index=idx_res.monthly_value.index,
    )

    logger.info("Generating plots...")
    reporting.plot_value_over_time(results, cum_contrib, config.OUTPUT_DIR / "diversified_value_log.png")
    reporting.plot_drawdowns(results, config.OUTPUT_DIR / "diversified_drawdowns.png")
    reporting.plot_sleeve_composition(
        div.sleeve_monthly_value, [e.date for e in div.crash_events], config.OUTPUT_DIR / "diversified_sleeve_composition.png"
    )
    reporting.plot_allocation_pie(config.DIVERSIFIED_WEIGHTS, config.OUTPUT_DIR / "diversified_allocation_pie.png")
    reporting.plot_metric_bars(list(all_metrics.values()), list(results.keys()), config.OUTPUT_DIR / "diversified_metric_comparison.png")

    failed_df = pd.DataFrame([{"ticker": f.ticker, "reason": f.reason} for f in fetch_result.failures])
    reporting.export_all_csv(config.OUTPUT_DIR, summary_df, basket_df, results, failed_df)

    sleeve_df = pd.DataFrame(
        [
            {"strategy": "Diversified", "sleeve": name, "final_value": r.final_value, "total_contributed": r.total_contributed}
            for name, r in div.sleeve_results.items()
        ]
    )
    sleeve_df.to_csv(config.OUTPUT_DIR / "diversified_sleeve_breakdown.csv", index=False)

    crash_df = pd.DataFrame(
        [
            {"date": e.date, "spy_drawdown": e.spy_drawdown, "cash_deployed": e.cash_deployed,
             "lookback_years_used": e.lookback_years_used, "winners": ", ".join(e.winners)}
            for e in div.crash_events
        ]
    )
    crash_df.to_csv(config.OUTPUT_DIR / "diversified_crash_events.csv", index=False)

    report_text = interpretation.build_report(
        all_metrics, results, sp_mem.source, len(fetch_result.failures),
        sum(r.n_dropped_partial_year for r in sp_rankings.values()) + sum(r.n_dropped_partial_year for r in asia_rankings.values()),
    )
    (config.OUTPUT_DIR / "interpretation_diversified.md").write_text(report_text)

    print("\n" + "=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(report_text)
    logger.info("Done. Outputs written to %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
