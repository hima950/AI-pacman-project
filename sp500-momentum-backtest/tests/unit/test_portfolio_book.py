"""Isolated unit tests for the _Book ledger in src/portfolio.py -- the
average-cost-basis buy/sell/slippage/tax arithmetic, in isolation from any
full monthly-contribution simulation."""
from __future__ import annotations

import pytest

from src.portfolio import SLIPPAGE, TAX_RATE, _Book


def test_buy_applies_slippage_against_the_buyer():
    book = _Book()
    shares, slippage_cost = book.buy("A", dollars=1000.0, price=100.0)
    expected_fill_price = 100.0 * (1 + SLIPPAGE)
    expected_shares = 1000.0 / expected_fill_price
    assert shares == pytest.approx(expected_shares)
    assert slippage_cost == pytest.approx(1000.0 - expected_shares * 100.0)
    assert slippage_cost > 0
    assert book.shares["A"] == pytest.approx(expected_shares)
    assert book.cost_basis["A"] == pytest.approx(1000.0)
    assert "A" in book.ever_held


def test_buy_zero_or_negative_inputs_are_no_ops():
    book = _Book()
    assert book.buy("A", dollars=0.0, price=100.0) == (0.0, 0.0)
    assert book.buy("A", dollars=100.0, price=0.0) == (0.0, 0.0)
    assert book.held("A") == 0.0


def test_sell_all_with_gain_and_tax():
    book = _Book()
    book.buy("A", dollars=1000.0, price=100.0)  # cost basis == 1000.0 (slippage-inclusive)
    shares, gross, slippage_cost, tax_paid = book.sell_all("A", price=150.0, tax=True)
    expected_fill_price = 150.0 * (1 - SLIPPAGE)
    expected_gross = shares * 150.0
    proceeds_before_tax = shares * expected_fill_price
    expected_gain = proceeds_before_tax - 1000.0
    assert gross == pytest.approx(expected_gross)
    assert expected_gain > 0
    assert tax_paid == pytest.approx(expected_gain * TAX_RATE)
    assert book.shares["A"] == 0.0
    assert book.cost_basis["A"] == 0.0
    net = gross - slippage_cost - tax_paid
    assert net < gross  # slippage and tax both bite


def test_sell_all_with_loss_pays_no_tax():
    book = _Book()
    book.buy("A", dollars=1000.0, price=100.0)
    _, gross, _, tax_paid = book.sell_all("A", price=50.0, tax=True)  # sold at a loss
    assert gross < 1000.0
    assert tax_paid == 0.0


def test_sell_all_respects_tax_flag_even_with_a_gain():
    book = _Book()
    book.buy("A", dollars=1000.0, price=100.0)
    _, _, _, tax_paid = book.sell_all("A", price=200.0, tax=False)  # buy-and-hold variant: no tax modeled
    assert tax_paid == 0.0


def test_sell_all_on_empty_position_is_a_no_op():
    book = _Book()
    result = book.sell_all("NEVER_BOUGHT", price=100.0, tax=True)
    assert result == (0.0, 0.0, 0.0, 0.0)


def test_value_sums_only_positive_holdings():
    book = _Book()
    book.buy("A", dollars=1000.0, price=100.0)
    book.buy("B", dollars=500.0, price=50.0)
    book.sell_all("B", price=60.0, tax=False)  # B now has 0 shares
    value = book.value(lambda t: {"A": 120.0, "B": 999.0}[t])
    # B's price is irrelevant now -- only A's shares * A's price should count
    expected = book.shares["A"] * 120.0
    assert value == pytest.approx(expected)


def test_value_skips_tickers_with_no_price_available():
    book = _Book()
    book.buy("A", dollars=1000.0, price=100.0)
    value = book.value(lambda t: None)
    assert value == 0.0
