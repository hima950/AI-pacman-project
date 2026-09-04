"""Cash sleeve with crash-triggered deployment.

Every month, $200 (by default) is added to a cash balance that accrues at
the prevailing risk-free rate (^IRX, same source used elsewhere in this
project; monthly accrual, applied at each month's contribution date). The
cash sleeve is otherwise idle -- until SPY's price closes more than
`crash_threshold` (25%) below its running all-time high, at which point
the *entire* accumulated cash balance is deployed in one shot into the
top-N S&P 500 names by trailing-N-year total return as of that date
(`ranking.rank_trailing_period`), split equally and bought with the same
5bp slippage used everywhere else. Deployed positions are held forever
(buy and hold) -- this sleeve is one-shot-per-crash, not itself rebalanced.

Re-arming: after a deployment, the trigger is disarmed until SPY makes a
*new* all-time high (i.e. fully recovers), so a single prolonged bear
market doesn't fire the trigger every single day. A fresh >25% drawdown
from that new peak fires it again -- this sleeve can deploy multiple times
over a multi-decade backtest (dot-com, GFC, COVID, etc., to the extent
each actually cleared 25% on a closing basis and we have the price history
to rank against).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import config, ranking
from .membership import MembershipData
from .portfolio import (
    Transaction,
    _Book,
    _first_trading_day_per_month,
    _last_trading_day_per_month,
    _price_on_or_before,
    _trading_calendar,
)

logger = logging.getLogger(__name__)


@dataclass
class CrashEvent:
    date: pd.Timestamp
    spy_drawdown: float
    cash_deployed: float
    lookback_years_used: int
    winners: list[str]


@dataclass
class CashCrashResult:
    monthly_value: pd.Series
    monthly_contribution: pd.Series
    transactions: list[Transaction]
    total_contributed: float
    final_value: float
    distinct_positions: int
    final_positions: dict[str, float]
    cashflow_dates: list[pd.Timestamp]
    cashflow_amounts: list[float]
    annual_turnover: pd.Series
    crash_events: list[CrashEvent] = field(default_factory=list)
    strategy: str = "cash_crash"
    variant: str = "buy_and_hold"


def simulate_cash_crash_sleeve(
    membership: MembershipData,
    prices: dict[str, pd.Series],
    spy_prices: pd.Series,
    monthly_contribution: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_monthly: float | pd.Series = 0.0,
    crash_threshold: float = config.CRASH_DRAWDOWN_THRESHOLD,
    lookback_years: int = config.CRASH_LOOKBACK_YEARS,
    top_n: int = config.CRASH_TOP_N,
) -> CashCrashResult:
    daily_calendar = _trading_calendar(spy_prices, start, end)
    first_days = set(_first_trading_day_per_month(daily_calendar))
    last_days = _last_trading_day_per_month(daily_calendar)
    last_day_set = set(last_days)

    def rf_at(date: pd.Timestamp) -> float:
        if isinstance(risk_free_monthly, pd.Series):
            sub = risk_free_monthly[risk_free_monthly.index <= date]
            return float(sub.iloc[-1]) if not sub.empty else float(risk_free_monthly.mean())
        return float(risk_free_monthly)

    def price_lookup(ticker: str, date: pd.Timestamp) -> float | None:
        s = prices.get(ticker)
        if s is None:
            return None
        return _price_on_or_before(s, date)

    book = _Book()
    transactions: list[Transaction] = []
    crash_events: list[CrashEvent] = []
    monthly_value_map: dict[pd.Timestamp, float] = {}

    cash_balance = 0.0
    running_peak = -float("inf")
    armed = True
    n_contribution_months = 0

    for date in daily_calendar:
        # daily_calendar is derived directly from spy_prices' own index, so
        # this is an exact (fast) hit, not a scan -- unlike price_lookup()
        # below, which is used for arbitrary tickers on arbitrary dates.
        spy_price = float(spy_prices.loc[date])

        if date in first_days:
            cash_balance *= 1.0 + rf_at(date)
            cash_balance += monthly_contribution
            n_contribution_months += 1

        running_peak = max(running_peak, spy_price)
        drawdown = spy_price / running_peak - 1.0

        if armed and drawdown <= -crash_threshold:
            eligible = membership.members_as_of_nearest_year_end(date.year - 1)
            trailing = ranking.rank_trailing_period(
                date, eligible, prices, top_n=top_n, lookback_years=lookback_years
            )
            if trailing.winners and cash_balance > 0:
                per_name = cash_balance / len(trailing.winners)
                deployed = 0.0
                for w in trailing.winners:
                    p = price_lookup(w, date)
                    if p is None or p <= 0:
                        continue
                    shares, slip = book.buy(w, per_name, p)
                    transactions.append(
                        Transaction(date, w, "buy", shares, p, per_name, slip, 0.0, per_name, "crash_deployment_buy")
                    )
                    deployed += per_name
                cash_balance -= deployed
                crash_events.append(
                    CrashEvent(date, drawdown, deployed, trailing.lookback_years, trailing.winners)
                )
                logger.info(
                    "CRASH TRIGGER %s: SPY drawdown %.1f%%, deployed $%.0f into %s (trailing %dy return)",
                    date.date(), drawdown * 100, deployed, trailing.winners, trailing.lookback_years,
                )
            else:
                crash_events.append(CrashEvent(date, drawdown, 0.0, trailing.lookback_years, []))
                logger.warning(
                    "CRASH TRIGGER %s: SPY drawdown %.1f%% but no eligible trailing-%dy winners "
                    "(or no cash to deploy) -- cash retained.",
                    date.date(), drawdown * 100, lookback_years,
                )
            armed = False

        if not armed and drawdown >= 0:  # new all-time high -> re-arm
            armed = True

        if date in last_day_set:
            value = cash_balance + book.value(lambda t, d=date: price_lookup(t, d))
            monthly_value_map[date] = value

    monthly_value = pd.Series(monthly_value_map).sort_index()
    monthly_contribution_series = pd.Series(monthly_contribution, index=monthly_value.index)

    total_contributed = monthly_contribution * n_contribution_months
    final_value = float(monthly_value.iloc[-1]) if len(monthly_value) else 0.0

    first_days_sorted = sorted(first_days)
    cashflow_dates = list(first_days_sorted) + [end]
    cashflow_amounts = [-monthly_contribution for _ in first_days_sorted] + [final_value]

    final_positions = {
        t: sh * price_lookup(t, end) for t, sh in book.shares.items() if sh > 0 and price_lookup(t, end) is not None
    }

    tx_df = pd.DataFrame(
        [{"date": t.date, "action": t.action, "gross": t.gross_amount, "year": t.date.year} for t in transactions]
    )
    turnover = pd.Series(dtype=float)
    if not tx_df.empty:
        sells_by_year = tx_df[tx_df["action"] == "sell"].groupby("year")["gross"].sum()
        avg_val_by_year = monthly_value.groupby(monthly_value.index.year).mean()
        turnover = (sells_by_year / avg_val_by_year).dropna()

    return CashCrashResult(
        monthly_value=monthly_value,
        monthly_contribution=monthly_contribution_series,
        transactions=transactions,
        total_contributed=total_contributed,
        final_value=final_value,
        distinct_positions=len(book.ever_held),
        final_positions=final_positions,
        cashflow_dates=cashflow_dates,
        cashflow_amounts=cashflow_amounts,
        annual_turnover=turnover,
        crash_events=crash_events,
    )
