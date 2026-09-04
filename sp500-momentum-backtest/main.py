#!/usr/bin/env python3
"""SP500 momentum backtest: index vs top-5 vs top-10 prior-year-winner baskets.

Run with: python main.py
See README.md for the full methodology, assumptions and network requirements.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

from src import config, data_fetch, interpretation, membership, metrics as metrics_mod, portfolio, ranking, reporting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(config.OUTPUT_DIR / "run.log", mode="w")],
)
logger = logging.getLogger("main")


def build_risk_free_monthly(irx_series: pd.Series | None, month_end_dates) -> float | pd.Series:
    if irx_series is None or irx_series.empty:
        logger.warning("^IRX unavailable; falling back to flat %.1f%% risk-free rate.", config.RISK_FREE_FLAT_FALLBACK * 100)
        return config.RISK_FREE_FLAT_FALLBACK / 12.0
    vals = {}
    for d in month_end_dates:
        sub = irx_series[irx_series.index <= d]
        if sub.empty:
            continue
        annual_pct = float(sub.iloc[-1])
        vals[d] = (annual_pct / 100.0) / 12.0
    return pd.Series(vals)


def main() -> None:
    start_ts = pd.Timestamp(config.START_DATE)
    end_ts = pd.Timestamp(config.END_DATE)
    logger.info("Backtest window: %s to %s", start_ts.date(), end_ts.date())

    logger.info("Loading point-in-time S&P 500 membership...")
    mem = membership.load_membership()
    logger.info("Membership source: %s", mem.source)

    ranking_years = range(1999, end_ts.year)          # Y-1 snapshots needed
    holding_years = range(2000, end_ts.year + 1)       # basket years to simulate
    universe = membership.universe_of_all_tickers_needed(mem, ranking_years)
    logger.info("Universe of tickers needing price history: %d", len(universe))

    fetch_start = "1998-06-01"  # buffer before 2000 so year-2000 rankings have full prior-year data
    fetch_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Fetching SPY (index strategy + trading calendar reference)...")
    spy_result = data_fetch.fetch_adjusted_close([config.INDEX_TICKER], fetch_start, fetch_end)
    if config.INDEX_TICKER not in spy_result.prices:
        raise RuntimeError("Could not fetch SPY -- cannot proceed (calendar reference + Strategy A).")
    spy_prices = spy_result.prices[config.INDEX_TICKER]

    logger.info("Fetching risk-free proxy %s...", config.RISK_FREE_TICKER)
    irx_result = data_fetch.fetch_adjusted_close([config.RISK_FREE_TICKER], fetch_start, fetch_end)
    irx_series = irx_result.prices.get(config.RISK_FREE_TICKER)

    logger.info("Fetching adjusted-close price history for %d constituents (cached to parquet)...", len(universe))
    universe_list = sorted(universe)
    fetch_result = data_fetch.fetch_adjusted_close(universe_list, fetch_start, fetch_end)
    logger.info(
        "Fetched %d/%d tickers successfully; %d failed.",
        len(fetch_result.prices), len(universe_list), len(fetch_result.failures),
    )
    for f in fetch_result.failures:
        logger.warning("FAILED FETCH: %s (%s)", f.ticker, f.reason)

    all_prices = dict(fetch_result.prices)
    all_prices[config.INDEX_TICKER] = spy_prices

    logger.info("Ranking prior-year winners for holding years %s-%s...", holding_years.start, holding_years.stop - 1)
    top_ns = list(config.TOP_N_STRATEGIES.values())
    rankings = ranking.rank_all_years(holding_years, mem, all_prices, top_ns)

    n_dropped_ipo_total = sum(r.n_dropped_partial_year for r in rankings.values())
    for y, r in sorted(rankings.items()):
        logger.info(
            "Year %d (ranked on %d): %d eligible members, %d missing price data, %d dropped (partial-year/IPO)",
            y, r.ranking_year, r.n_membership, r.n_missing_price_data, r.n_dropped_partial_year,
        )

    basket_index = {y: [config.INDEX_TICKER] for y in holding_years}
    basket_top5 = {y: rankings[y].basket[5] for y in rankings if 5 in rankings[y].basket}
    basket_top10 = {y: rankings[y].basket[10] for y in rankings if 10 in rankings[y].basket}

    strategies = {
        "index": ("Index (SPY)", basket_index, True),
        "top5": ("Top 5", basket_top5, False),
        "top10": ("Top 10", basket_top10, False),
    }
    variants = ["buy_and_hold", "annual_rebalance"]

    results: dict[str, portfolio.SimulationResult] = {}
    for strat_key, (strat_label, basket, single_asset) in strategies.items():
        for variant in variants:
            logger.info("Simulating %s / %s...", strat_label, variant)
            res = portfolio.simulate(
                strategy_name=strat_key,
                variant=variant,
                basket_by_year=basket,
                prices=all_prices,
                calendar_reference=spy_prices,
                start=start_ts,
                end=end_ts,
                single_asset_no_rebalance=single_asset,
            )
            label = f"{strat_label} - {'Buy & Hold' if variant == 'buy_and_hold' else 'Annual Rebalance'}"
            results[label] = res

    month_end_dates = results[list(results.keys())[0]].monthly_value.index
    rf_monthly = build_risk_free_monthly(irx_series, month_end_dates)

    all_metrics = {}
    for label, res in results.items():
        all_metrics[label] = metrics_mod.compute_metrics(res, rf_monthly)

    summary_df = reporting.summary_table(list(all_metrics.values()))
    rankings_by_strategy = {"top5": rankings, "top10": rankings}
    basket_df = reporting.per_year_basket_table(rankings_by_strategy)

    idx_res = results[[l for l in results if "Index" in l][0]]
    cum_contrib = pd.Series(
        [config.MONTHLY_CONTRIBUTION * (i + 1) for i in range(len(idx_res.monthly_value))],
        index=idx_res.monthly_value.index,
    )

    reporting.plot_value_over_time(results, cum_contrib, config.OUTPUT_DIR / "portfolio_value_log.png")
    reporting.plot_drawdowns(results, config.OUTPUT_DIR / "drawdowns.png")

    failed_df = pd.DataFrame([{"ticker": f.ticker, "reason": f.reason} for f in fetch_result.failures])
    reporting.export_all_csv(config.OUTPUT_DIR, summary_df, basket_df, results, failed_df)

    report_text = interpretation.build_report(
        all_metrics, results, mem.source, len(fetch_result.failures), n_dropped_ipo_total
    )
    (config.OUTPUT_DIR / "interpretation.md").write_text(report_text)

    print("\n" + "=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(report_text)
    logger.info("Done. Outputs written to %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
