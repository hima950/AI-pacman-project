#!/usr/bin/env python3
"""End-to-end validation of Strategy D ("All-Weather Diversified") against
synthetic, fabricated data: crash detection/re-arming, trailing-10-year
winner selection, cash accrual, and the 6-sleeve combiner's totals
reconciling. Same spirit as tests/test_synthetic.py -- proves the new
mechanics are implemented correctly; says nothing about real markets.

Run: python tests/test_diversified_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config, diversified, portfolio, ranking
from src.crash_sleeve import simulate_cash_crash_sleeve
from src.membership import constant_membership

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def piecewise_series(segments: list[tuple[str, str, float]], base_price: float) -> pd.Series:
    """segments: list of (start, end, daily_rate); each segment's start
    price continues from the previous segment's ending price."""
    pieces = []
    p = base_price
    for start, end, rate in segments:
        idx = pd.bdate_range(start, end)
        vals = []
        for _ in idx:
            p = p * (1 + rate)
            vals.append(p)
        pieces.append(pd.Series(vals, index=idx))
    return pd.concat(pieces)


def build_crash_universe():
    # SPY: bull 1998-2007, crash >25% in 2008, recovery to a NEW all-time
    # high by end of 2011, second crash >25% in 2012, partial recovery.
    spy = piecewise_series(
        [
            ("1998-01-01", "2007-12-31", 0.00035),   # steady bull -> roughly +160% over ~10y
            ("2008-01-01", "2008-11-30", -0.0018),    # sharp decline, well past -25%
            ("2008-12-01", "2011-12-31", 0.0012),     # recovery past the old peak
            ("2012-01-01", "2012-09-30", -0.0020),    # second crash, again past -25%
            ("2012-10-01", "2014-12-31", 0.0006),     # partial recovery
        ],
        base_price=100.0,
    )

    # "Winners": strong, steady growth for the full window (rankable on a
    # full 10-year trailing lookback at the first crash, in late 2008).
    winners = {}
    for i, rate in enumerate([0.0009, 0.0008, 0.0007], start=1):
        winners[f"SYN_W{i}"] = piecewise_series([("1998-01-01", "2014-12-31", rate)], base_price=50.0)

    # "Losers": weak/negative growth -- must NOT be picked as trailing-10y winners.
    losers = {}
    for i, rate in enumerate([-0.0003, 0.0001], start=1):
        losers[f"SYN_L{i}"] = piecewise_series([("1998-01-01", "2014-12-31", rate)], base_price=50.0)

    gld = piecewise_series([("1998-01-01", "2014-12-31", 0.0002)], base_price=40.0)
    agg = piecewise_series([("1998-01-01", "2014-12-31", 0.00008)], base_price=90.0)
    btc = piecewise_series([("1998-01-01", "2014-12-31", 0.0015)], base_price=1.0)  # fake, just needs to exist

    prices = {"SPY": spy, config.METALS_TICKER: gld, config.BONDS_TICKER: agg, config.HIGH_RISK_TICKER: btc}
    prices.update(winners)
    prices.update(losers)

    all_tickers = set(winners) | set(losers)
    mem = constant_membership(all_tickers, range(1997, 2015), source_label="synthetic-crash-test")
    return prices, mem, spy


def test_crash_detection_and_deployment():
    prices, mem, spy = build_crash_universe()
    start, end = pd.Timestamp("2000-01-01"), pd.Timestamp("2014-12-31")

    result = simulate_cash_crash_sleeve(
        mem, prices, spy,
        monthly_contribution=200.0,
        start=start, end=end,
        risk_free_monthly=0.02 / 12,
        crash_threshold=0.25,
        lookback_years=10,
        top_n=3,
    )

    check("cash sleeve total_contributed == 15*12*200", result.total_contributed == 15 * 12 * 200.0, str(result.total_contributed))
    check("cash sleeve detected at least 2 crash events (2008 and 2012)", len(result.crash_events) >= 2, str(len(result.crash_events)))

    if result.crash_events:
        first = result.crash_events[0]
        check("first crash event drawdown <= -25%", first.spy_drawdown <= -0.25, str(first.spy_drawdown))
        check("first crash event deployed > 0 cash", first.cash_deployed > 0, str(first.cash_deployed))
        check(
            "first crash event picked only strong-growth winners, no losers",
            all(w.startswith("SYN_W") for w in first.winners) and len(first.winners) > 0,
            str(first.winners),
        )
        check("first crash used close to the full 10y lookback (>= 8y)", first.lookback_years_used >= 8, str(first.lookback_years_used))

    if len(result.crash_events) >= 2:
        second = result.crash_events[1]
        check("second crash event is a distinct, later date than the first", second.date > result.crash_events[0].date)
        check("second crash event also deployed real winners (re-arm worked)", len(second.winners) > 0, str(second.winners))

    # after 2+ deployments, cash balance should have been drawn down close
    # to zero right after each event (verified indirectly: final value
    # should mostly sit in the deployed winner positions, not idle cash)
    check("final value is finite and positive", np.isfinite(result.final_value) and result.final_value > 0)
    check("distinct_positions <= number of winner tickers available", result.distinct_positions <= 3, str(result.distinct_positions))

    buys = [t for t in result.transactions if t.action == "buy"]
    check("every crash-deployment buy is tagged with the right reason", all(t.reason == "crash_deployment_buy" for t in buys))
    check("no buys of loser tickers ever occurred", not any(t.ticker.startswith("SYN_L") for t in buys), str([t.ticker for t in buys if t.ticker.startswith('SYN_L')]))

    return prices, mem, spy


def test_diversified_composer():
    prices, mem, spy = test_crash_detection_and_deployment()
    start, end = pd.Timestamp("2000-01-01"), pd.Timestamp("2014-12-31")
    holding_years = range(2000, 2015)

    sp_top10_basket = {y: ["SYN_W1", "SYN_W2"] for y in holding_years}
    # ranking year Y needs a Y-1 snapshot, so the membership range must
    # start one year before the first holding year (mirrors main.py's
    # `range(1999, end_ts.year)` for the real S&P universe).
    asia_mem = constant_membership({"SYN_W3", "SYN_L1"}, range(holding_years.start - 1, holding_years.stop), source_label="synthetic-asia-test")
    asia_rankings = ranking.rank_all_years(holding_years, asia_mem, prices, [10])

    div_result = diversified.build_diversified_strategy(
        holding_years=holding_years,
        membership=mem,
        all_prices=prices,
        spy_prices=spy,
        sp_top10_basket_by_year=sp_top10_basket,
        asia_rankings=asia_rankings,
        start=start,
        end=end,
        risk_free_monthly=0.02 / 12,
        total_monthly_contribution=1000.0,
    )

    combined = div_result.combined
    sleeves = div_result.sleeve_results

    expected_total_contributed = 1000.0 * 15 * 12
    check(
        "combined total_contributed == sum of sleeve contributions == 1000/mo",
        abs(combined.total_contributed - expected_total_contributed) < 1.0,
        f"{combined.total_contributed} vs {expected_total_contributed}",
    )

    sleeve_final_sum = sum(r.final_value for r in sleeves.values())
    check(
        "combined final_value == sum of sleeve final values",
        abs(combined.final_value - sleeve_final_sum) < 1e-6,
        f"{combined.final_value} vs {sleeve_final_sum}",
    )

    # spot-check the monthly_value series itself, not just the endpoints
    combined_mid = combined.monthly_value.iloc[len(combined.monthly_value) // 2]
    sleeve_mid_sum = sum(r.monthly_value.iloc[len(r.monthly_value) // 2] for r in sleeves.values())
    check(
        "combined monthly_value mid-point == sum of sleeve mid-points",
        abs(combined_mid - sleeve_mid_sum) < 1e-6,
        f"{combined_mid} vs {sleeve_mid_sum}",
    )

    check("sleeve_monthly_value has all 6 sleeve columns", set(div_result.sleeve_monthly_value.columns) == set(sleeves.keys()))
    check("crash_events surfaced on the DiversifiedResult", len(div_result.crash_events) >= 2)

    weights_used = {name: r.total_contributed / expected_total_contributed for name, r in sleeves.items()}
    for name, expected_w in config.DIVERSIFIED_WEIGHTS.items():
        check(
            f"sleeve '{name}' got its configured weight ({expected_w:.0%}) of total contributions",
            abs(weights_used[name] - expected_w) < 1e-6,
            f"got {weights_used[name]:.4f}",
        )

    print(f"  combined: final=${combined.final_value:,.0f} contributed=${combined.total_contributed:,.0f}")
    print(f"  crash events: {[(e.date.date(), f'{e.spy_drawdown:.1%}', e.winners) for e in div_result.crash_events]}")


def test_asset_starting_mid_backtest():
    """The high_risk (BTC-USD) sleeve's real-world situation: the asset has
    no price history for the first ~14 years of the 2000-2026 backtest
    window (BTC-USD starts trading ~2014). Contributions before that
    should sit idle (uninvested, no return) rather than error out, and
    should all lump-sum invest the first month data exists."""
    spy = piecewise_series([("2005-01-01", "2012-12-31", 0.0003)], base_price=100.0)
    crypto_start = "2010-01-01"
    crypto = piecewise_series([(crypto_start, "2012-12-31", 0.0005)], base_price=1.0)
    prices = {"SPY": spy, "SYN_CRYPTO": crypto}

    start, end = pd.Timestamp("2005-01-01"), pd.Timestamp("2012-12-31")
    basket = {y: ["SYN_CRYPTO"] for y in range(2005, 2013)}
    res = portfolio.simulate(
        "test_late_asset", "buy_and_hold", basket, prices, spy,
        start=start, end=end, monthly_contribution=100.0, single_asset_no_rebalance=True,
    )

    before = res.monthly_value[res.monthly_value.index < crypto_start]
    after = res.monthly_value[res.monthly_value.index >= crypto_start]
    check("monthly_value has an entry for every month even before the asset existed (no dropped months)", len(before) == 5 * 12, str(len(before)))
    check("value is exactly 0 while contributions have nowhere to go", (before == 0).all(), str(before[before != 0]))
    check("the accumulated pre-2010 contributions lump-sum invest once the asset starts trading", after.iloc[0] > 100.0 * 12, f"got {after.iloc[0]}")
    check("final value is finite and positive", np.isfinite(res.final_value) and res.final_value > 0)
    check("total_contributed unaffected by the asset's late start", res.total_contributed == 100.0 * 8 * 12, str(res.total_contributed))


if __name__ == "__main__":
    test_diversified_composer()
    test_asset_starting_mid_backtest()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL DIVERSIFIED-STRATEGY SYNTHETIC CHECKS PASSED")
