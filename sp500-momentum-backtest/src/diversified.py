"""Strategy D: "All-Weather Diversified".

Same $1,000/month contribution as strategies A/B/C, split across six
sleeves (see config.DIVERSIFIED_WEIGHTS):

    10% precious metals & bonds (50/50 GLD / AGG)
    20% top 10 S&P 500 stocks of the year (reuses the same point-in-time
        momentum basket as Strategy C)
    20% top 10 of a fixed Asia stock/ADR basket, by trailing 1-year return
        (see config.ASIA_UNIVERSE -- NOT a real point-in-time index)
    20% S&P 500 index fund (SPY, standing in for Vanguard's VOO -- see
        README, VOO only exists from 2010)
    20% cash, accruing at the risk-free rate, deployed in full into the
        top-10 trailing-10-year winners the first time SPY closes >25%
        below its running all-time high (src/crash_sleeve.py)
    10% high risk/high reward (BTC-USD; sits idle until BTC-USD price
        history begins in Sep 2014 -- see README)

Every stock-picking sleeve is buy-and-hold only (no annual-rebalance/tax
variant was requested for this strategy).

Five of the six sleeves are literally the existing single-basket
portfolio.simulate() engine, just fed a different monthly dollar amount
and basket schedule -- that's deliberate reuse of already-tested
machinery rather than a second implementation of the same mechanics. Only
the cash/crash sleeve needed new code (src/crash_sleeve.py), since
"accrue interest until a market-wide trigger fires" doesn't fit the
existing single-basket-per-year shape at all.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .crash_sleeve import CashCrashResult, CrashEvent, simulate_cash_crash_sleeve
from .membership import MembershipData
from .portfolio import SimulationResult, Transaction, simulate
from .ranking import RankingResult


@dataclass
class DiversifiedResult:
    combined: SimulationResult
    sleeve_results: dict[str, SimulationResult | CashCrashResult]
    sleeve_monthly_value: pd.DataFrame     # columns = sleeve name, for stacked-area plotting
    crash_events: list[CrashEvent]
    asia_rankings: dict[int, RankingResult] = field(default_factory=dict)


def build_diversified_strategy(
    holding_years: range,
    membership: MembershipData,
    all_prices: dict[str, pd.Series],
    spy_prices: pd.Series,
    sp_top10_basket_by_year: dict[int, list[str]],
    asia_rankings: dict[int, RankingResult],
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_monthly: float | pd.Series,
    total_monthly_contribution: float = config.MONTHLY_CONTRIBUTION,
) -> DiversifiedResult:
    w = config.DIVERSIFIED_WEIGHTS

    metals_bonds_basket = {y: [config.METALS_TICKER, config.BONDS_TICKER] for y in holding_years}
    index_fund_basket = {y: [config.INDEX_TICKER] for y in holding_years}
    high_risk_basket = {y: [config.HIGH_RISK_TICKER] for y in holding_years}
    asia_basket_by_year = {y: r.basket.get(10, []) for y, r in asia_rankings.items()}

    sleeve_results: dict[str, SimulationResult | CashCrashResult] = {}

    sleeve_results["metals_bonds"] = simulate(
        "diversified_metals_bonds", "buy_and_hold", metals_bonds_basket, all_prices, spy_prices,
        start=start, end=end, monthly_contribution=total_monthly_contribution * w["metals_bonds"],
        single_asset_no_rebalance=True,
    )
    sleeve_results["sp_top10"] = simulate(
        "diversified_sp_top10", "buy_and_hold", sp_top10_basket_by_year, all_prices, spy_prices,
        start=start, end=end, monthly_contribution=total_monthly_contribution * w["sp_top10"],
    )
    sleeve_results["asia_top10"] = simulate(
        "diversified_asia_top10", "buy_and_hold", asia_basket_by_year, all_prices, spy_prices,
        start=start, end=end, monthly_contribution=total_monthly_contribution * w["asia_top10"],
    )
    sleeve_results["index_fund"] = simulate(
        "diversified_index_fund", "buy_and_hold", index_fund_basket, all_prices, spy_prices,
        start=start, end=end, monthly_contribution=total_monthly_contribution * w["index_fund"],
        single_asset_no_rebalance=True,
    )
    sleeve_results["high_risk"] = simulate(
        "diversified_high_risk", "buy_and_hold", high_risk_basket, all_prices, spy_prices,
        start=start, end=end, monthly_contribution=total_monthly_contribution * w["high_risk"],
        single_asset_no_rebalance=True,
    )
    sleeve_results["cash"] = simulate_cash_crash_sleeve(
        membership, all_prices, spy_prices,
        monthly_contribution=total_monthly_contribution * w["cash"],
        start=start, end=end, risk_free_monthly=risk_free_monthly,
    )

    combined = _combine(sleeve_results, end)
    sleeve_value_df = pd.DataFrame({name: r.monthly_value for name, r in sleeve_results.items()})

    return DiversifiedResult(
        combined=combined,
        sleeve_results=sleeve_results,
        sleeve_monthly_value=sleeve_value_df,
        crash_events=sleeve_results["cash"].crash_events,
        asia_rankings=asia_rankings,
    )


def _combine(sleeve_results: dict[str, SimulationResult | CashCrashResult], end: pd.Timestamp) -> SimulationResult:
    monthly_value = None
    monthly_contribution = None
    for r in sleeve_results.values():
        monthly_value = r.monthly_value if monthly_value is None else monthly_value.add(r.monthly_value, fill_value=0.0)
        monthly_contribution = (
            r.monthly_contribution if monthly_contribution is None else monthly_contribution.add(r.monthly_contribution, fill_value=0.0)
        )

    total_contributed = sum(r.total_contributed for r in sleeve_results.values())
    final_value = float(monthly_value.iloc[-1]) if len(monthly_value) else 0.0

    transactions: list[Transaction] = []
    for sleeve_name, r in sleeve_results.items():
        for t in r.transactions:
            transactions.append(
                Transaction(t.date, t.ticker, t.action, t.shares, t.price, t.gross_amount, t.slippage_cost, t.tax, t.net_amount, f"{sleeve_name}:{t.reason}")
            )

    # every sleeve shares the same monthly contribution calendar
    first_days = sorted({d for r in sleeve_results.values() for d in r.cashflow_dates if d != end})
    cashflow_dates = first_days + [end]
    total_monthly = sum(r.total_contributed for r in sleeve_results.values()) / max(len(first_days), 1)
    cashflow_amounts = [-total_monthly for _ in first_days] + [final_value]

    final_positions: dict[str, float] = defaultdict(float)
    for r in sleeve_results.values():
        for tk, v in r.final_positions.items():
            final_positions[tk] += v

    distinct_positions = sum(r.distinct_positions for r in sleeve_results.values())

    tx_df = pd.DataFrame(
        [{"date": t.date, "action": t.action, "gross": t.gross_amount, "year": t.date.year} for t in transactions]
    )
    turnover = pd.Series(dtype=float)
    if not tx_df.empty:
        sells_by_year = tx_df[tx_df["action"] == "sell"].groupby("year")["gross"].sum()
        avg_val_by_year = monthly_value.groupby(monthly_value.index.year).mean()
        turnover = (sells_by_year / avg_val_by_year).dropna()

    return SimulationResult(
        strategy="diversified",
        variant="buy_and_hold",
        transactions=transactions,
        monthly_value=monthly_value,
        monthly_contribution=monthly_contribution,
        total_contributed=total_contributed,
        final_value=final_value,
        distinct_positions=distinct_positions,
        cashflow_dates=cashflow_dates,
        cashflow_amounts=cashflow_amounts,
        annual_turnover=turnover,
        final_positions=dict(final_positions),
    )
