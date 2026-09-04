"""Isolated unit test for the cash-accrual arithmetic in
src/crash_sleeve.py, with no crash ever firing -- pins down the exact
compounding formula independent of the fuller crash-detection scenario in
tests/test_diversified_synthetic.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.crash_sleeve import simulate_cash_crash_sleeve
from src.membership import constant_membership


def test_cash_accrues_at_the_configured_monthly_rate_with_no_crash():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    spy = pd.Series([100.0] * len(idx), index=idx)  # perfectly flat -- never crashes
    mem = constant_membership({"SYN_X"}, range(2019, 2021))
    prices = {"SPY": spy, "SYN_X": pd.Series([50.0] * len(idx), index=idx)}

    rf_monthly = 0.01
    result = simulate_cash_crash_sleeve(
        mem, prices, spy,
        monthly_contribution=100.0,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-03-31"),
        risk_free_monthly=rf_monthly,
        crash_threshold=0.25,
    )

    assert result.crash_events == []
    # cash *= (1+rf); cash += 100, applied once per month, starting from 0
    m1 = 0.0 * (1 + rf_monthly) + 100.0
    m2 = m1 * (1 + rf_monthly) + 100.0
    m3 = m2 * (1 + rf_monthly) + 100.0
    assert result.monthly_value.iloc[0] == pytest.approx(m1)
    assert result.monthly_value.iloc[1] == pytest.approx(m2)
    assert result.monthly_value.iloc[-1] == pytest.approx(m3)
    assert result.final_value == pytest.approx(m3)
    assert result.total_contributed == pytest.approx(300.0)


def test_no_crash_below_threshold_never_deploys():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    n = len(idx)
    # a steady 20% decline -- real, but short of the 25% trigger
    spy = pd.Series([100.0 - 20.0 * i / (n - 1) for i in range(n)], index=idx)
    mem = constant_membership({"SYN_X"}, range(2019, 2021))
    prices = {"SPY": spy, "SYN_X": pd.Series([50.0] * n, index=idx)}

    result = simulate_cash_crash_sleeve(
        mem, prices, spy,
        monthly_contribution=100.0,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-06-30"),
        risk_free_monthly=0.0,
        crash_threshold=0.25,
    )
    assert result.crash_events == []
    assert result.final_value == pytest.approx(600.0)  # no accrual (rf=0), all still sitting in cash
