"""Isolated unit tests for src/metrics.py -- each case is hand-computable,
unlike tests/test_synthetic.py's whole-pipeline scenarios."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import metrics


def test_xirr_single_period_10_percent():
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
    amounts = [-1000.0, 1100.0]
    rate = metrics.xirr(dates, amounts)
    assert rate == pytest.approx(0.10, abs=1e-3)


def test_xirr_matches_known_compounding_rate():
    # -100 at t=0, -100 at t=+1y, payout at t=+2y equal to what a known
    # rate r would produce: 100*(1+r)^2 + 100*(1+r)^1. XIRR should recover r.
    r_true = 0.15
    d0 = pd.Timestamp("2010-01-01")
    d1 = pd.Timestamp("2011-01-01")
    d2 = pd.Timestamp("2012-01-01")
    payout = 100 * (1 + r_true) ** 2 + 100 * (1 + r_true) ** 1
    rate = metrics.xirr([d0, d1, d2], [-100.0, -100.0, payout])
    assert rate == pytest.approx(r_true, abs=1e-3)


def test_xirr_zero_return():
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
    amounts = [-1000.0, 1000.0]
    rate = metrics.xirr(dates, amounts)
    assert rate == pytest.approx(0.0, abs=1e-3)


def test_monthly_twr_returns_single_period():
    monthly_value = pd.Series([1100.0], index=[pd.Timestamp("2020-01-31")])
    monthly_contribution = pd.Series([1000.0], index=[pd.Timestamp("2020-01-31")])
    r = metrics.monthly_twr_returns(monthly_value, monthly_contribution)
    assert r.iloc[0] == pytest.approx(0.10)


def test_monthly_twr_returns_two_periods_no_lookahead():
    idx = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    monthly_value = pd.Series([1100.0, 2310.0], index=idx)
    monthly_contribution = pd.Series([1000.0, 1000.0], index=idx)
    r = metrics.monthly_twr_returns(monthly_value, monthly_contribution)
    assert r.iloc[0] == pytest.approx(0.10)
    # r2 = 2310 / (1100 + 1000) - 1 = 0.10
    assert r.iloc[1] == pytest.approx(0.10)


def test_twr_index_and_annualized_return_one_year():
    monthly_returns = pd.Series([0.01] * 12)
    idx = metrics.twr_index(monthly_returns)
    assert idx.iloc[-1] == pytest.approx(1.01**12)
    ann = metrics.annualized_twr_return(monthly_returns)
    # exactly 12 months -> annualized return == total growth - 1
    assert ann == pytest.approx(1.01**12 - 1)


def test_annualized_twr_return_two_years_scales_correctly():
    monthly_returns = pd.Series([0.01] * 24)
    ann = metrics.annualized_twr_return(monthly_returns)
    total_growth = 1.01**24
    assert ann == pytest.approx(total_growth ** (12 / 24) - 1)


def test_annualized_volatility_zero_for_constant_returns():
    monthly_returns = pd.Series([0.01] * 12)
    assert metrics.annualized_volatility(monthly_returns) == pytest.approx(0.0, abs=1e-12)


def test_annualized_volatility_matches_manual_calc():
    values = [0.02, -0.01, 0.03, 0.0]
    monthly_returns = pd.Series(values)
    expected = np.std(values, ddof=1) * math.sqrt(12)
    assert metrics.annualized_volatility(monthly_returns) == pytest.approx(expected)


def test_max_drawdown_known_series():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    values = [1.0, 1.1, 1.05, 0.9, 1.2]
    s = pd.Series(values, index=idx)
    dd, peak_date, trough_date = metrics.max_drawdown(s)
    # running max at each point: 1, 1.1, 1.1, 1.1, 1.2 -> drawdowns: 0, 0, -0.04545, -0.18182, 0
    assert dd == pytest.approx(0.9 / 1.1 - 1.0, abs=1e-6)
    assert peak_date == idx[1]
    assert trough_date == idx[3]


def test_max_drawdown_no_drawdown_when_monotonic():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    s = pd.Series([1.0, 1.05, 1.1, 1.2], index=idx)
    dd, peak_date, trough_date = metrics.max_drawdown(s)
    assert dd == pytest.approx(0.0)


def test_sharpe_nan_when_zero_volatility():
    monthly_returns = pd.Series([0.01] * 6)
    sharpe = metrics.sharpe_ratio(monthly_returns, rf_monthly=0.0)
    assert math.isnan(sharpe)


def test_sharpe_matches_manual_calc():
    values = [0.02, -0.01, 0.03, 0.0, 0.01, -0.02]
    monthly_returns = pd.Series(values)
    rf = 0.001
    excess = monthly_returns - rf
    expected = excess.mean() * 12 / (excess.std(ddof=1) * math.sqrt(12))
    assert metrics.sharpe_ratio(monthly_returns, rf_monthly=rf) == pytest.approx(expected)


def test_sortino_ignores_positive_returns_in_downside_dev():
    # All positive returns -> zero downside deviation -> NaN Sortino (undefined, not infinite/zero)
    monthly_returns = pd.Series([0.01, 0.02, 0.015])
    sortino = metrics.sortino_ratio(monthly_returns, rf_monthly=0.0)
    assert math.isnan(sortino)


def test_sortino_finite_with_mixed_returns():
    monthly_returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    sortino = metrics.sortino_ratio(monthly_returns, rf_monthly=0.0)
    assert np.isfinite(sortino)


def test_best_worst_calendar_year():
    idx = pd.to_datetime(["2020-01-31", "2020-12-31", "2021-01-31", "2021-12-31"])
    # 2020: two periods of +10% each -> (1.1*1.1 - 1) ~ 21%
    # 2021: two periods of -5% each -> (0.95*0.95 - 1) ~ -9.75%
    monthly_returns = pd.Series([0.10, 0.10, -0.05, -0.05], index=idx)
    (best_year, best_ret), (worst_year, worst_ret) = metrics.best_worst_calendar_year(monthly_returns)
    assert best_year == 2020
    assert best_ret == pytest.approx(1.1 * 1.1 - 1.0)
    assert worst_year == 2021
    assert worst_ret == pytest.approx(0.95 * 0.95 - 1.0)
