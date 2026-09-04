"""Auto-generated, numbers-driven narrative read of the results.

Deliberately built from the computed metrics/transactions rather than
hardcoded prose, so the commentary tracks whatever data the pipeline was
actually run against instead of describing results that may not match.
"""
from __future__ import annotations

import pandas as pd

from .metrics import Metrics
from .portfolio import SimulationResult


def profit_by_ticker(result: SimulationResult) -> pd.Series:
    """Net profit attributable to each ticker ever held: realized proceeds
    + remaining market value - dollars invested. A rough per-name P&L
    decomposition (ignores that pooled cash from redistributions blurs
    attribution slightly), good enough to flag concentration."""
    invested: dict[str, float] = {}
    realized: dict[str, float] = {}
    for t in result.transactions:
        if t.action == "buy":
            invested[t.ticker] = invested.get(t.ticker, 0.0) + t.gross_amount
        else:
            realized[t.ticker] = realized.get(t.ticker, 0.0) + t.net_amount

    tickers = set(invested) | set(realized) | set(result.final_positions)
    profit = {}
    for tk in tickers:
        profit[tk] = (
            realized.get(tk, 0.0) + result.final_positions.get(tk, 0.0) - invested.get(tk, 0.0)
        )
    return pd.Series(profit).sort_values(ascending=False)


def concentration_summary(result: SimulationResult) -> dict:
    p = profit_by_ticker(result)
    positive = p[p > 0]
    total_positive = positive.sum()
    if total_positive <= 0 or len(positive) == 0:
        return {"top1_ticker": None, "top1_share": 0.0, "top2_tickers": [], "top2_share": 0.0}
    top2 = positive.head(2)
    return {
        "top1_ticker": positive.index[0],
        "top1_share": float(positive.iloc[0] / total_positive),
        "top2_tickers": list(top2.index),
        "top2_share": float(top2.sum() / total_positive),
        "profit_by_ticker": p,
    }


def build_report(
    metrics_by_label: dict[str, Metrics],
    results_by_label: dict[str, SimulationResult],
    membership_source: str,
    n_failed_tickers: int,
    n_dropped_ipo_events: int,
) -> str:
    lines: list[str] = []
    lines.append("# Interpretation\n")

    lines.append("## Data quality caveats\n")
    if membership_source == "bundled-fallback":
        lines.append(
            "- Point-in-time S&P 500 membership came from a **bundled fallback snapshot** "
            "(the live fetch of the membership dataset failed at run time), not a live pull. "
            "The snapshot is a community-maintained (MIT-licensed) reconstruction from Wikipedia's "
            "index-change history, not an official vendor point-in-time feed -- treat exact "
            "membership on any given date as *approximately* right, especially pre-2001.\n"
        )
    else:
        lines.append(
            "- Point-in-time S&P 500 membership was fetched live from a community-maintained "
            "(MIT-licensed) reconstruction of Wikipedia's index-change history. This is close to "
            "approach (2) in the brief (reconstructed from documented additions/removals) but is "
            "a third-party pre-built reconstruction rather than this code parsing Wikipedia "
            "directly, and it is not an official vendor point-in-time feed -- treat exact "
            "membership on any given date, especially pre-2001, as approximately right rather "
            "than certain.\n"
        )
    lines.append(f"- {n_failed_tickers} ticker(s) failed to fetch price data entirely (see failed_tickers.csv) and were excluded from ranking pools, not silently treated as zero-return.\n")
    lines.append(f"- {n_dropped_ipo_events} (strategy, year) basket-eligibility checks dropped a name for not having a full year of prior-year price history (mid-year IPOs or gaps) -- see annual_baskets.csv / n_dropped_partial_year.\n")

    lines.append("\n## Headline comparison\n")
    for label, m in metrics_by_label.items():
        lines.append(
            f"- **{label}**: XIRR (money-weighted) {m.xirr:.2%}, TWR annualized {m.twr_annualized:.2%}, "
            f"vol {m.annualized_vol:.2%}, Sharpe {m.sharpe:.2f}, Sortino {m.sortino:.2f}, "
            f"max drawdown {m.max_drawdown:.1%} ({m.drawdown_peak_date} -> {m.drawdown_trough_date}), "
            f"final value ${m.final_value:,.0f} on ${m.total_contributed:,.0f} contributed."
        )

    lines.append("\n## Concentration check (does the result hinge on one or two names?)\n")
    for label, res in results_by_label.items():
        if res.strategy == "index":
            continue
        c = concentration_summary(res)
        if c["top1_ticker"] is None:
            lines.append(f"- **{label}**: no positive-profit positions to analyze.")
            continue
        flag = " -- **large single-name dependency**" if c["top2_share"] > 0.5 else ""
        lines.append(
            f"- **{label}**: top single name ({c['top1_ticker']}) accounts for "
            f"{c['top1_share']:.1%} of total gross profit; top two names "
            f"({', '.join(c['top2_tickers'])}) account for {c['top2_share']:.1%}.{flag}"
        )

    lines.append("\n## Reading the gap\n")
    idx_labels = [l for l in metrics_by_label if results_by_label[l].strategy == "index"]
    momentum_labels = [l for l in metrics_by_label if results_by_label[l].strategy != "index"]
    if idx_labels and momentum_labels:
        idx_m = metrics_by_label[idx_labels[0]]
        best_mom_label = max(momentum_labels, key=lambda l: metrics_by_label[l].xirr)
        best_mom = metrics_by_label[best_mom_label]
        gap = best_mom.xirr - idx_m.xirr
        sharpe_worse = best_mom.sharpe < idx_m.sharpe
        dd_deeper = best_mom.max_drawdown < idx_m.max_drawdown  # more negative = deeper
        direction = "beat" if gap >= 0 else "trailed"
        lines.append(
            f"The best momentum variant ({best_mom_label}) {direction} the index on raw "
            f"money-weighted return, by {abs(gap):.2%} XIRR "
            f"({best_mom.xirr:.2%} vs {idx_m.xirr:.2%}). "
        )
        sharpe_clause = (
            "its Sharpe ratio was *lower* than the index's" if sharpe_worse
            else "its Sharpe ratio came in *ahead* of the index's"
        )
        dd_clause = (
            f"its max drawdown ({best_mom.max_drawdown:.1%}) was deeper than the index's "
            f"({idx_m.max_drawdown:.1%})" if dd_deeper
            else f"its max drawdown ({best_mom.max_drawdown:.1%}) was not deeper than the index's "
            f"({idx_m.max_drawdown:.1%})"
        )
        if sharpe_worse and dd_deeper:
            lines.append(
                f"But {sharpe_clause}, and {dd_clause} -- consistent with the momentum literature: "
                "extreme short-horizon (prior-year) winners tend to mean-revert, and a naive "
                "'buy last year's biggest gainers' basket collects a concentrated, high-beta set "
                "of names whose blow-ups outweigh their persistence on a risk-adjusted basis."
            )
        else:
            lines.append(
                f"On risk-adjusted terms, {sharpe_clause} and {dd_clause}. This cuts against the "
                "usual prior-year-winners-mean-revert prior at least on this slice of the data, "
                "and is worth treating with extra skepticism given the membership-data caveats "
                "above -- re-check the concentration numbers before trusting it at face value."
            )
    lines.append(
        "\nGiven the membership-reconstruction and price-gap caveats above, treat differences "
        "smaller than a few percentage points of annualized return as noise rather than signal; "
        "only gaps that are large relative to those data-quality uncertainties, and that hold up "
        "after checking the concentration numbers above, should be read as a real finding."
    )
    return "\n".join(lines)
