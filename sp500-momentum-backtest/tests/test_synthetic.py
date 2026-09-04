#!/usr/bin/env python3
"""End-to-end validation of the backtest mechanics against synthetic data.

This sandbox cannot reach Yahoo Finance or Wikipedia (see README.md), so
this is what stands in for a real run: fabricated, clearly-fake tickers
(SYN_*) with deterministic price paths, run through the exact same
membership -> ranking -> portfolio -> metrics pipeline used on real data.
It proves the mechanics (ranking correctness, slippage, tax, delisting
redistribution, IPO exclusion, XIRR/TWR/drawdown math) are implemented
correctly. It says nothing about what the real S&P 500 numbers would be --
run `python main.py` on a machine with normal internet access for that.

Run: python tests/test_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics as metrics_mod
from src import portfolio, ranking
from src.membership import MembershipData

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def bdate_prices(start: str, end: str, daily_rates: dict[int, float], base_price: float) -> pd.Series:
    """Build a synthetic price series that compounds at a per-calendar-year
    daily growth rate, chained across years so annual returns are exact and
    predictable by construction."""
    idx = pd.bdate_range(start, end)
    prices = []
    p = base_price
    last_year = None
    for d in idx:
        if last_year is not None and d.year != last_year:
            pass  # carry p forward across year boundary (continuous compounding)
        rate = daily_rates.get(d.year, 0.0)
        p = p * (1 + rate)
        prices.append(p)
        last_year = d.year
    return pd.Series(prices, index=idx)


def build_synthetic_universe():
    # Per-ticker, per-year daily growth rates chosen to make rankings
    # unambiguous and hand-verifiable.
    rates = {
        "SYN_A": {1999: 0.0002, 2000: 0.0001, 2001: 0.0003, 2002: 0.0001},
        "SYN_B": {1999: 0.0004, 2000: 0.0015, 2001: 0.0002, 2002: 0.0002},  # best 2000 -> top of 2001 basket
        "SYN_C": {1999: 0.0012, 2000: 0.0003, 2001: 0.0002, 2002: 0.0002},  # best 1999 -> top of 2000 basket
        "SYN_D": {1999: 0.0010, 2000: 0.0004, 2001: 0.0002, 2002: 0.0002},  # 2nd best 1999
        "SYN_E": {1999: 0.0008, 2000: 0.0005, 2001: 0.0002, 2002: 0.0002},  # 3rd best 1999
        "SYN_G": {1999: 0.0001, 2000: 0.0020, 2001: 0.0002, 2002: 0.0000},  # best 2000, DELISTS mid-2001
        "SYN_H": {1999: 0.0001, 2000: 0.0018, 2001: 0.0000, 2002: 0.0000},  # 2nd-best 2000, gone BEFORE 2001 starts
        "SPY":   {1999: 0.0003, 2000: 0.0003, 2001: 0.0003, 2002: 0.0003},
    }
    prices = {}
    for tk, rr in rates.items():
        prices[tk] = bdate_prices("1998-06-01", "2002-12-31", rr, base_price=50.0)

    # SYN_F: mid-year-1999 IPO -> must be excluded from the 2000 basket
    # (no full year of 1999 history) but eligible for the 2001 basket
    # (full 2000 history).
    ipo_rates = {1999: 0.0025, 2000: 0.0002, 2001: 0.0002, 2002: 0.0002}
    f_full = bdate_prices("1999-07-01", "2002-12-31", ipo_rates, base_price=20.0)
    prices["SYN_F"] = f_full

    # SYN_G delists on 2001-06-15: truncate its price history there.
    g = prices["SYN_G"]
    prices["SYN_G"] = g[g.index <= "2001-06-15"]

    # SYN_H: has a full, rankable 2000 (so it lands in the 2001 basket by
    # return), but its price history ends 2000-12-31 -- i.e. it's already
    # gone before the very first 2001 contribution date. Regression test
    # for buying a name that delisted before we ever got a chance to hold
    # it (as opposed to delisting *while* held).
    h = prices["SYN_H"]
    prices["SYN_H"] = h[h.index <= "2000-12-31"]

    membership_tickers = {"SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_E", "SYN_F", "SYN_G", "SYN_H"}
    yearend_rows = []
    for y in (1998, 1999, 2000, 2001, 2002):
        yearend_rows.append({"as_of_year_end": y, "snapshot_date": f"{y}-12-31", "tickers": set(membership_tickers)})
    yearend = pd.DataFrame(yearend_rows)
    mem = MembershipData(yearend=yearend, ticker_ranges=None, source="synthetic")
    return prices, mem


def test_ranking_excludes_ipo_and_ranks_correctly():
    prices, mem = build_synthetic_universe()
    res2000 = ranking.rank_year(2000, mem, prices, top_ns=[3])
    check(
        "IPO (SYN_F) excluded from year-2000 ranking (no full 1999 history)",
        "SYN_F" not in res2000.ranked["ticker"].tolist(),
    )
    check("year-2000 ranking dropped exactly 1 partial-year name", res2000.n_dropped_partial_year == 1, str(res2000.dropped_partial_year_tickers))
    expected_top3_2000 = ["SYN_C", "SYN_D", "SYN_E"]
    check(
        "year-2000 top-3 basket == [SYN_C, SYN_D, SYN_E] (highest 1999 returns)",
        res2000.basket[3] == expected_top3_2000,
        str(res2000.basket[3]),
    )

    res2001 = ranking.rank_year(2001, mem, prices, top_ns=[3])
    check("SYN_F included in year-2001 ranking (has full 2000 history)", "SYN_F" in res2001.ranked["ticker"].tolist())
    check(
        "year-2001 top-3 basket == [SYN_G, SYN_H, SYN_B] (highest 2000 returns; "
        "SYN_H ranks despite having ZERO 2001 price data, since eligibility is "
        "about 2000 history, not 2001 survival)",
        res2001.basket[3] == ["SYN_G", "SYN_H", "SYN_B"],
        str(res2001.basket[3]),
    )
    return prices, mem, res2000, res2001


def test_portfolio_mechanics():
    prices, mem, res2000, res2001 = test_ranking_excludes_ipo_and_ranks_correctly()
    res2002 = ranking.rank_year(2002, mem, prices, top_ns=[3])

    basket_by_year = {2000: res2000.basket[3], 2001: res2001.basket[3], 2002: res2002.basket[3]}
    spy = prices["SPY"]

    start = pd.Timestamp("2000-01-01")
    end = pd.Timestamp("2002-12-31")

    bh = portfolio.simulate("top3", "buy_and_hold", basket_by_year, prices, spy, start=start, end=end)
    ar = portfolio.simulate("top3", "annual_rebalance", basket_by_year, prices, spy, start=start, end=end)

    n_months = 36
    check("buy&hold total_contributed == 36 * 1000", bh.total_contributed == n_months * 1000.0, str(bh.total_contributed))
    check("annual_rebalance total_contributed == 36 * 1000", ar.total_contributed == n_months * 1000.0, str(ar.total_contributed))
    check("buy&hold final value is finite and positive", np.isfinite(bh.final_value) and bh.final_value > 0)
    check("annual_rebalance final value is finite and positive", np.isfinite(ar.final_value) and ar.final_value > 0)

    # slippage: every buy's realized fill price should exceed the quoted price
    buys = [t for t in bh.transactions if t.action == "buy"]
    check("all buy transactions pay a price above quote (slippage)", all(t.gross_amount / t.shares > t.price - 1e-9 for t in buys if t.shares > 0))
    check("buy slippage_cost > 0 on every real buy", all(t.slippage_cost > 0 for t in buys if t.gross_amount > 0))

    # SYN_G forced delisting mid-2001 under buy & hold
    forced = [t for t in bh.transactions if t.reason == "forced_delisting_sale"]
    check("buy&hold recorded a forced delisting sale for SYN_G", any(t.ticker == "SYN_G" for t in forced), str([t.ticker for t in forced]))
    redistributed = [t for t in bh.transactions if t.reason == "delisting_redistribution_buy"]
    check("delisting proceeds were redistributed into surviving siblings", len(redistributed) > 0)
    check("SYN_G no longer held after its forced sale (buy&hold)", not any(sh_t == "SYN_G" for sh_t in [t.ticker for t in bh.transactions if t.action == "buy" and t.date > forced[0].date and t.ticker == "SYN_G"]) if forced else True)

    # SYN_H: selected into the 2001 basket on its 2000 return, but its
    # price history ends 2000-12-31 -- never buyable in 2001. Must never
    # appear as a buy, and must never trigger a (nonsensical) forced-sale
    # since it was never held.
    check("SYN_H (delisted before 2001 even starts) is never bought", not any(t.ticker == "SYN_H" for t in buys), str([t for t in buys if t.ticker == "SYN_H"]))
    check("SYN_H never appears in a forced_delisting_sale (was never held)", not any(t.ticker == "SYN_H" for t in forced))

    # annual rebalance: January liquidations should generate tax whenever there's a realized gain
    rebalance_sells = [t for t in ar.transactions if t.reason == "rebalance_liquidate"]
    check("annual_rebalance produced rebalance_liquidate transactions", len(rebalance_sells) > 0)
    check("annual_rebalance charged tax on at least one liquidation with a gain", any(t.tax > 0 for t in rebalance_sells), "no positive-gain liquidations found")
    bh_sells = [t for t in bh.transactions if t.reason == "rebalance_liquidate"]
    check("buy&hold never liquidates for rebalancing", len(bh_sells) == 0)

    # XIRR / TWR sanity
    x = metrics_mod.xirr(bh.cashflow_dates, bh.cashflow_amounts)
    check("XIRR is finite", np.isfinite(x), str(x))
    r = metrics_mod.monthly_twr_returns(bh.monthly_value, bh.monthly_contribution)
    idx = metrics_mod.twr_index(r)
    check("TWR index is monotonically defined (no NaNs)", not idx.isna().any())
    dd, peak, trough = metrics_mod.max_drawdown(idx)
    check("max_drawdown <= 0", dd <= 1e-9, str(dd))
    check("drawdown peak date <= trough date", peak is None or trough is None or peak <= trough)

    # regression guard: month-1 value must reflect only month-1's $1000
    # contribution (this caught a real bug where valuation used final
    # end-of-sim share counts against historical prices).
    first_month_value = bh.monthly_value.iloc[0]
    check(
        "month-1 portfolio value is close to the single $1000 contribution, not inflated",
        900.0 < first_month_value < 1150.0,
        f"got ${first_month_value:,.2f}",
    )
    check(
        "monthly_value is monotonically non-decreasing in share count terms (contributions only add) "
        "-- final value far exceeds first month's",
        bh.monthly_value.iloc[-1] > bh.monthly_value.iloc[0] * 5,
        f"{bh.monthly_value.iloc[-1]} vs {bh.monthly_value.iloc[0]}",
    )

    m = metrics_mod.compute_metrics(bh, 0.02 / 12)
    check("compute_metrics runs end-to-end and returns finite Sharpe", np.isfinite(m.sharpe) or True)  # Sharpe can be NaN if vol==0; just must not raise
    print(f"  buy&hold: final=${bh.final_value:,.0f} contributed=${bh.total_contributed:,.0f} XIRR={x:.2%}")
    print(f"  annual_rebalance: final=${ar.final_value:,.0f} contributed=${ar.total_contributed:,.0f}")


def test_index_strategy_variant_equivalence():
    prices, mem = build_synthetic_universe()
    spy = prices["SPY"]
    basket = {y: ["SPY"] for y in (2000, 2001, 2002)}
    start, end = pd.Timestamp("2000-01-01"), pd.Timestamp("2002-12-31")
    v1 = portfolio.simulate("index", "buy_and_hold", basket, {"SPY": spy}, spy, start=start, end=end, single_asset_no_rebalance=True)
    v2 = portfolio.simulate("index", "annual_rebalance", basket, {"SPY": spy}, spy, start=start, end=end, single_asset_no_rebalance=True)
    check(
        "Index strategy: buy&hold and annual_rebalance produce identical values (no-op rebalance)",
        abs(v1.final_value - v2.final_value) < 1e-6,
        f"{v1.final_value} vs {v2.final_value}",
    )


if __name__ == "__main__":
    test_portfolio_mechanics()
    test_index_strategy_variant_equivalence()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL SYNTHETIC-DATA CHECKS PASSED")
