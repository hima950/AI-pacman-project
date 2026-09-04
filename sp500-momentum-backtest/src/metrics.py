"""Performance metrics.

Money-weighted return (XIRR) is the headline number because contributions
are spread over 26 years -- a naive CAGR on final-vs-first value would
massively overstate returns since almost all the capital was contributed
long after year one. Time-weighted return (TWR) is reported alongside it
to separate "the strategy's return" from "the investor's timing luck /
contribution schedule".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import SimulationResult


def xirr(dates: list[pd.Timestamp], amounts: list[float], guess: float = 0.1) -> float:
    """Money-weighted annualized return via Newton's method with a bisection
    fallback. `amounts` should be negative for outflows (contributions) and
    positive for the final inflow (ending value)."""
    dates = pd.to_datetime(pd.Index(dates))
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])
    amounts = np.array(amounts, dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    def dnpv(rate: float) -> float:
        return float(np.sum(-years * amounts / (1.0 + rate) ** (years + 1)))

    rate = guess
    for _ in range(100):
        f = npv(rate)
        fp = dnpv(rate)
        if abs(fp) < 1e-12:
            break
        new_rate = rate - f / fp
        if not np.isfinite(new_rate) or new_rate <= -0.999999:
            break
        if abs(new_rate - rate) < 1e-10:
            rate = new_rate
            break
        rate = new_rate
    else:
        pass

    if np.isfinite(rate) and abs(npv(rate)) < 1.0:
        return rate

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def monthly_twr_returns(monthly_value: pd.Series, monthly_contribution: pd.Series) -> pd.Series:
    """r_m = V_m / (V_{m-1} + C_m) - 1, contribution assumed at period start
    (true here since all buys happen on the first trading day of the
    month)."""
    v_prev = monthly_value.shift(1).fillna(0.0)
    denom = v_prev + monthly_contribution
    r = monthly_value / denom.replace(0.0, np.nan) - 1.0
    return r.fillna(0.0)


def twr_index(monthly_returns: pd.Series) -> pd.Series:
    return (1.0 + monthly_returns).cumprod()


def annualized_twr_return(monthly_returns: pd.Series) -> float:
    n = len(monthly_returns)
    if n == 0:
        return float("nan")
    total_growth = float((1.0 + monthly_returns).prod())
    return total_growth ** (12.0 / n) - 1.0


def annualized_volatility(monthly_returns: pd.Series) -> float:
    if len(monthly_returns) < 2:
        return float("nan")
    return float(monthly_returns.std(ddof=1) * np.sqrt(12))


def max_drawdown(index_series: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if index_series.empty:
        return float("nan"), None, None
    running_max = index_series.cummax()
    dd = index_series / running_max - 1.0
    trough_date = dd.idxmin()
    trough_value = dd.min()
    peak_date = index_series[:trough_date].idxmax()
    return float(trough_value), peak_date, trough_date


def sortino_ratio(monthly_returns: pd.Series, rf_monthly: float | pd.Series = 0.0) -> float:
    if isinstance(rf_monthly, pd.Series):
        excess = monthly_returns - rf_monthly.reindex(monthly_returns.index).fillna(rf_monthly.mean())
    else:
        excess = monthly_returns - rf_monthly
    downside = excess.clip(upper=0.0)
    downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(12)
    if downside_dev == 0 or not np.isfinite(downside_dev):
        return float("nan")
    ann_excess = annualized_twr_return(monthly_returns) - (
        rf_monthly.mean() * 12 if isinstance(rf_monthly, pd.Series) else rf_monthly * 12
    )
    return float(ann_excess / downside_dev)


def sharpe_ratio(monthly_returns: pd.Series, rf_monthly: float | pd.Series = 0.0) -> float:
    if isinstance(rf_monthly, pd.Series):
        excess = monthly_returns - rf_monthly.reindex(monthly_returns.index).fillna(rf_monthly.mean())
    else:
        excess = monthly_returns - rf_monthly
    vol = excess.std(ddof=1) * np.sqrt(12)
    if vol == 0 or not np.isfinite(vol):
        return float("nan")
    return float(excess.mean() * 12 / vol)


def best_worst_calendar_year(monthly_returns: pd.Series) -> tuple[tuple[int, float], tuple[int, float]]:
    years = monthly_returns.index.year
    annual = (1.0 + monthly_returns).groupby(years).prod() - 1.0
    if annual.empty:
        return (None, float("nan")), (None, float("nan"))
    best_year = int(annual.idxmax())
    worst_year = int(annual.idxmin())
    return (best_year, float(annual.loc[best_year])), (worst_year, float(annual.loc[worst_year]))


@dataclass
class Metrics:
    strategy: str
    variant: str
    final_value: float
    total_contributed: float
    xirr: float
    twr_annualized: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    drawdown_peak_date: pd.Timestamp | None
    drawdown_trough_date: pd.Timestamp | None
    best_year: int | None
    best_year_return: float
    worst_year: int | None
    worst_year_return: float
    distinct_positions: int
    avg_annual_turnover: float


def compute_metrics(result: SimulationResult, risk_free_monthly: float | pd.Series) -> Metrics:
    r = monthly_twr_returns(result.monthly_value, result.monthly_contribution)
    idx = twr_index(r)
    dd, peak_date, trough_date = max_drawdown(idx)
    (by, byr), (wy, wyr) = best_worst_calendar_year(r)

    money_weighted = xirr(result.cashflow_dates, result.cashflow_amounts)

    return Metrics(
        strategy=result.strategy,
        variant=result.variant,
        final_value=result.final_value,
        total_contributed=result.total_contributed,
        xirr=money_weighted,
        twr_annualized=annualized_twr_return(r),
        annualized_vol=annualized_volatility(r),
        sharpe=sharpe_ratio(r, risk_free_monthly),
        sortino=sortino_ratio(r, risk_free_monthly),
        max_drawdown=dd,
        drawdown_peak_date=peak_date,
        drawdown_trough_date=trough_date,
        best_year=by,
        best_year_return=byr,
        worst_year=wy,
        worst_year_return=wyr,
        distinct_positions=result.distinct_positions,
        avg_annual_turnover=float(result.annual_turnover.mean()) if len(result.annual_turnover) else 0.0,
    )
