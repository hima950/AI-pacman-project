"""Isolated unit tests for src/interpretation.py's concentration diagnostic."""
from __future__ import annotations

import pandas as pd
import pytest

from src.interpretation import concentration_summary, profit_by_ticker
from src.portfolio import SimulationResult, Transaction


def _result(transactions, final_positions):
    return SimulationResult(
        strategy="top3",
        variant="buy_and_hold",
        transactions=transactions,
        monthly_value=pd.Series(dtype=float),
        monthly_contribution=pd.Series(dtype=float),
        total_contributed=0.0,
        final_value=sum(final_positions.values()),
        distinct_positions=len(final_positions),
        cashflow_dates=[],
        cashflow_amounts=[],
        annual_turnover=pd.Series(dtype=float),
        final_positions=final_positions,
    )


def test_profit_by_ticker_buy_and_hold_no_sales():
    d = pd.Timestamp("2020-01-01")
    txs = [
        Transaction(d, "A", "buy", 10.0, 100.0, 1000.0, 5.0, 0.0, 1000.0, "contribution"),
        Transaction(d, "B", "buy", 20.0, 50.0, 1000.0, 5.0, 0.0, 1000.0, "contribution"),
    ]
    result = _result(txs, final_positions={"A": 1500.0, "B": 800.0})
    profit = profit_by_ticker(result)
    assert profit["A"] == pytest.approx(1500.0 - 1000.0)
    assert profit["B"] == pytest.approx(800.0 - 1000.0)


def test_profit_by_ticker_includes_realized_sales():
    d = pd.Timestamp("2020-01-01")
    txs = [
        Transaction(d, "A", "buy", 10.0, 100.0, 1000.0, 5.0, 0.0, 1000.0, "contribution"),
        Transaction(d, "A", "sell", 10.0, 150.0, 1500.0, 5.0, 50.0, 1445.0, "forced_delisting_sale"),
    ]
    result = _result(txs, final_positions={})  # A was fully sold, no remaining position
    profit = profit_by_ticker(result)
    assert profit["A"] == pytest.approx(1445.0 - 1000.0)


def test_concentration_summary_flags_single_name_dependency():
    d = pd.Timestamp("2020-01-01")
    txs = [
        Transaction(d, "WINNER", "buy", 1.0, 100.0, 1000.0, 0.0, 0.0, 1000.0, "contribution"),
        Transaction(d, "LOSER", "buy", 1.0, 100.0, 1000.0, 0.0, 0.0, 1000.0, "contribution"),
    ]
    result = _result(txs, final_positions={"WINNER": 10000.0, "LOSER": 900.0})
    c = concentration_summary(result)
    assert c["top1_ticker"] == "WINNER"
    # WINNER profit=9000, LOSER profit=-100 (excluded from "positive" pool) -> top1_share == 1.0
    assert c["top1_share"] == pytest.approx(1.0)
    assert c["top2_share"] == pytest.approx(1.0)


def test_concentration_summary_balanced_book_has_no_single_dependency():
    d = pd.Timestamp("2020-01-01")
    txs = [
        Transaction(d, "A", "buy", 1.0, 100.0, 1000.0, 0.0, 0.0, 1000.0, "contribution"),
        Transaction(d, "B", "buy", 1.0, 100.0, 1000.0, 0.0, 0.0, 1000.0, "contribution"),
        Transaction(d, "C", "buy", 1.0, 100.0, 1000.0, 0.0, 0.0, 1000.0, "contribution"),
    ]
    result = _result(txs, final_positions={"A": 1200.0, "B": 1200.0, "C": 1200.0})
    c = concentration_summary(result)
    assert c["top1_share"] == pytest.approx(1 / 3, abs=1e-6)
    assert c["top2_share"] == pytest.approx(2 / 3, abs=1e-6)


def test_concentration_summary_no_positive_profit():
    d = pd.Timestamp("2020-01-01")
    txs = [Transaction(d, "A", "buy", 1.0, 100.0, 1000.0, 0.0, 0.0, 1000.0, "contribution")]
    result = _result(txs, final_positions={"A": 500.0})  # a loss, not a gain
    c = concentration_summary(result)
    assert c["top1_ticker"] is None
    assert c["top1_share"] == 0.0
