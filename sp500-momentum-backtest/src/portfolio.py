"""Portfolio simulation engine.

Simulates $1,000 contributed on the first trading day of every month into a
basket of tickers, under two mechanics:

  Variant 1 (buy_and_hold): each month's contribution buys the *current
      year's* basket, equally split by dollar amount. Nothing is ever sold
      except forced sales of names that get delisted/acquired mid-year,
      whose proceeds are immediately redistributed into the surviving
      members of the basket(s) that ticker belonged to.

  Variant 2 (annual_rebalance): every January (first trading day), every
      existing holding is liquidated -- 5bp slippage on the sale, and 15%
      tax on any *realized gain* (average-cost basis; losses are not
      credited/rebated) -- and the after-tax proceeds plus that month's
      $1,000 are redeployed equally across the new basket. Forced
      delisting sales during the year are taxed the same way, since tax is
      a feature of this variant's realize-every-January regime.

Both variants: fractional shares, zero commissions, 5bp slippage on every
buy and every sell (modelled by shifting the fill price by 5bp against the
trader, not as a separate fee line).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

SLIPPAGE = config.SLIPPAGE_BPS / 10_000.0
TAX_RATE = config.CAPITAL_GAINS_TAX_RATE


@dataclass
class Transaction:
    date: pd.Timestamp
    ticker: str
    action: str
    shares: float
    price: float
    gross_amount: float
    slippage_cost: float
    tax: float
    net_amount: float
    reason: str


@dataclass
class SimulationResult:
    strategy: str
    variant: str
    transactions: list[Transaction]
    monthly_value: pd.Series
    monthly_contribution: pd.Series
    total_contributed: float
    final_value: float
    distinct_positions: int
    cashflow_dates: list[pd.Timestamp]
    cashflow_amounts: list[float]
    annual_turnover: pd.Series
    final_positions: dict[str, float] = field(default_factory=dict)
    dropped_no_price_events: list[str] = field(default_factory=list)


def _trading_calendar(reference_prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    idx = reference_prices.index
    idx = idx[(idx >= start) & (idx <= end)]
    return idx.sort_values()


def _first_trading_day_per_month(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(calendar, index=calendar)
    grouped = s.groupby([calendar.year, calendar.month]).min()
    return sorted(grouped.tolist())


def _last_trading_day_per_month(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(calendar, index=calendar)
    grouped = s.groupby([calendar.year, calendar.month]).max()
    return sorted(grouped.tolist())


def _price_on_or_before(series: pd.Series, date: pd.Timestamp) -> float | None:
    sub = series[series.index <= date]
    if sub.empty:
        return None
    return float(sub.iloc[-1])


class _Book:
    """Average-cost-basis holdings ledger for one simulation run."""

    def __init__(self) -> None:
        self.shares: dict[str, float] = {}
        self.cost_basis: dict[str, float] = {}
        self.ever_held: set[str] = set()

    def held(self, ticker: str) -> float:
        return self.shares.get(ticker, 0.0)

    def buy(self, ticker: str, dollars: float, price: float) -> tuple[float, float]:
        """Returns (shares_bought, slippage_cost_dollars)."""
        if dollars <= 0 or price <= 0:
            return 0.0, 0.0
        fill_price = price * (1 + SLIPPAGE)
        shares = dollars / fill_price
        slippage_cost = dollars - shares * price
        self.shares[ticker] = self.shares.get(ticker, 0.0) + shares
        self.cost_basis[ticker] = self.cost_basis.get(ticker, 0.0) + dollars
        self.ever_held.add(ticker)
        return shares, slippage_cost

    def sell_all(self, ticker: str, price: float, tax: bool) -> tuple[float, float, float, float]:
        """Liquidate full position. Returns (shares_sold, gross, slippage_cost, tax_paid).
        Net proceeds = gross - slippage_cost - tax_paid."""
        shares = self.shares.get(ticker, 0.0)
        if shares <= 0:
            return 0.0, 0.0, 0.0, 0.0
        fill_price = price * (1 - SLIPPAGE)
        gross_at_market = shares * price
        proceeds_before_tax = shares * fill_price
        slippage_cost = gross_at_market - proceeds_before_tax
        basis = self.cost_basis.get(ticker, 0.0)
        realized_gain = proceeds_before_tax - basis
        tax_paid = max(0.0, realized_gain) * TAX_RATE if tax else 0.0
        self.shares[ticker] = 0.0
        self.cost_basis[ticker] = 0.0
        return shares, gross_at_market, slippage_cost, tax_paid

    def value(self, price_lookup) -> float:
        total = 0.0
        for t, sh in self.shares.items():
            if sh <= 0:
                continue
            p = price_lookup(t)
            if p is not None:
                total += sh * p
        return total


def simulate(
    strategy_name: str,
    variant: str,                       # "buy_and_hold" | "annual_rebalance"
    basket_by_year: dict[int, list[str]],
    prices: dict[str, pd.Series],
    calendar_reference: pd.Series,
    start: pd.Timestamp = pd.Timestamp(config.START_DATE),
    end: pd.Timestamp = pd.Timestamp(config.END_DATE),
    monthly_contribution: float = config.MONTHLY_CONTRIBUTION,
    single_asset_no_rebalance: bool = False,
) -> SimulationResult:
    assert variant in ("buy_and_hold", "annual_rebalance")
    calendar = _trading_calendar(calendar_reference, start, end)
    first_days = _first_trading_day_per_month(calendar)

    book = _Book()
    transactions: list[Transaction] = []
    pending_cash = 0.0  # from redistribution events that found no surviving sibling

    # ticker -> set of basket-years it belongs to (for buy&hold delisting redistribution)
    ticker_to_years: dict[str, set[int]] = {}
    for y, tks in basket_by_year.items():
        for t in tks:
            ticker_to_years.setdefault(t, set()).add(y)

    last_avail: dict[str, pd.Timestamp] = {t: s.index.max() for t, s in prices.items() if not s.empty}
    already_force_sold: set[str] = set()

    def price_lookup(ticker: str, date: pd.Timestamp) -> float | None:
        s = prices.get(ticker)
        if s is None:
            return None
        return _price_on_or_before(s, date)

    def do_buy(ticker: str, dollars: float, date: pd.Timestamp, reason: str) -> float:
        p = price_lookup(ticker, date)
        if p is None or p <= 0 or dollars <= 0:
            return dollars  # couldn't invest; caller may choose to carry it forward
        shares, slip = book.buy(ticker, dollars, p)
        transactions.append(
            Transaction(date, ticker, "buy", shares, p, dollars, slip, 0.0, dollars, reason)
        )
        return 0.0

    def do_sell_all(ticker: str, date: pd.Timestamp, reason: str, tax: bool) -> float:
        p = price_lookup(ticker, date)
        if p is None:
            return 0.0
        shares, gross, slip, tax_paid = book.sell_all(ticker, p, tax=tax)
        if shares <= 0:
            return 0.0
        net = gross - slip - tax_paid
        transactions.append(
            Transaction(date, ticker, "sell", shares, p, gross, slip, tax_paid, net, reason)
        )
        return net

    def handle_forced_delistings(date: pd.Timestamp, active_year: int, tax: bool) -> None:
        nonlocal pending_cash
        for ticker in list(book.shares.keys()):
            if book.shares.get(ticker, 0.0) <= 0 or ticker in already_force_sold:
                continue
            la = last_avail.get(ticker)
            if la is None or date <= la:
                continue
            already_force_sold.add(ticker)
            net = do_sell_all(ticker, la, "forced_delisting_sale", tax=tax)
            if net <= 0:
                continue
            if variant == "annual_rebalance":
                siblings = [
                    t
                    for t in basket_by_year.get(active_year, [])
                    if t != ticker and book.shares.get(t, 0.0) > 0
                ]
            else:
                years = ticker_to_years.get(ticker, set())
                siblings = sorted(
                    {
                        t
                        for y in years
                        for t in basket_by_year.get(y, [])
                        if t != ticker and book.shares.get(t, 0.0) > 0
                    }
                )
            if not siblings:
                siblings = [t for t, sh in book.shares.items() if sh > 0 and t != ticker]
            if not siblings:
                pending_cash += net
                continue
            share_each = net / len(siblings)
            for s in siblings:
                do_buy(s, share_each, la, "delisting_redistribution_buy")

    first_ever_month = first_days[0] if first_days else None
    last_days = _last_trading_day_per_month(calendar)
    last_day_of_month = {d.to_period("M"): d for d in last_days}
    monthly_value_map: dict[pd.Timestamp, float] = {}

    for date in first_days:
        year = date.year
        basket = basket_by_year.get(year, [])
        handle_forced_delistings(date, year, tax=(variant == "annual_rebalance"))

        is_january_rebalance = (
            variant == "annual_rebalance"
            and date.month == 1
            and date != first_ever_month
            and not single_asset_no_rebalance
        )
        if is_january_rebalance:
            proceeds = 0.0
            for ticker in [t for t, sh in book.shares.items() if sh > 0]:
                proceeds += do_sell_all(ticker, date, "rebalance_liquidate", tax=True)
            pending_cash += proceeds

        contribution = monthly_contribution + pending_cash
        pending_cash = 0.0

        # A ticker already forced-sold for delisting -- or one that was
        # never held but has already delisted by this date (e.g. selected
        # into the basket off its Y-1 return, then acquired in the first
        # days of Y before we get a chance to buy it) -- is no longer
        # tradeable; route new contributions to the rest of the basket
        # instead of buying it at its last (stale, off-market) price.
        active_basket = [
            t
            for t in basket
            if t not in already_force_sold and (last_avail.get(t) is None or date <= last_avail[t])
        ]

        if active_basket:
            leftover_total = 0.0
            per_name = contribution / len(active_basket)
            for ticker in active_basket:
                leftover_total += do_buy(ticker, per_name, date, "contribution")
            pending_cash += leftover_total
            handle_forced_delistings(date, year, tax=(variant == "annual_rebalance"))
        else:
            # No tradeable basket this month (e.g. no ranking available for
            # this year yet) -- carry the contribution forward to be
            # invested once a basket exists, but still record this month's
            # valuation below rather than silently dropping it from the
            # output series (that used to shrink monthly_value's index
            # relative to every other concurrently-running sleeve).
            pending_cash += contribution

        # Snapshot value now, at this month's *last* trading day, using
        # shares as they stand right after this month's transactions --
        # nothing else trades until next month's first_days event, so this
        # is the correct point-in-time holding for the whole rest of the
        # month. (Valuing with the book's *final* end-of-simulation share
        # count against historical prices, computed in a separate pass
        # after the loop, was the original bug here: it massively
        # inflated every early-month value.)
        month_end = last_day_of_month[date.to_period("M")]
        monthly_value_map[month_end] = book.value(lambda t, d=month_end: price_lookup(t, d))

    monthly_value = pd.Series(monthly_value_map).sort_index()

    contrib_by_month: dict[pd.Period, float] = {}
    for d in first_days:
        contrib_by_month[d.to_period("M")] = contrib_by_month.get(d.to_period("M"), 0.0) + monthly_contribution
    monthly_contribution_series = pd.Series(
        {d: contrib_by_month.get(d.to_period("M"), 0.0) for d in last_days}
    )

    total_contributed = monthly_contribution * len(first_days)
    final_value = monthly_value.iloc[-1] if len(monthly_value) else 0.0

    cashflow_dates = list(first_days) + [end]
    cashflow_amounts = [-monthly_contribution for _ in first_days] + [final_value]

    # crude annual turnover: total sell $ in a year / average portfolio value that year
    tx_df = pd.DataFrame(
        [
            {"date": t.date, "action": t.action, "gross": t.gross_amount, "year": t.date.year}
            for t in transactions
        ]
    )
    turnover = pd.Series(dtype=float)
    if not tx_df.empty:
        sells_by_year = tx_df[tx_df["action"] == "sell"].groupby("year")["gross"].sum()
        avg_val_by_year = monthly_value.groupby(monthly_value.index.year).mean()
        turnover = (sells_by_year / avg_val_by_year).dropna()

    final_positions = {
        t: sh * price_lookup(t, end) for t, sh in book.shares.items() if sh > 0 and price_lookup(t, end) is not None
    }

    return SimulationResult(
        strategy=strategy_name,
        variant=variant,
        transactions=transactions,
        monthly_value=monthly_value,
        monthly_contribution=monthly_contribution_series,
        total_contributed=total_contributed,
        final_value=final_value,
        distinct_positions=len(book.ever_held),
        cashflow_dates=cashflow_dates,
        cashflow_amounts=cashflow_amounts,
        annual_turnover=turnover,
        final_positions=final_positions,
    )
