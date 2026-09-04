"""Tables, plots and CSV exports."""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics as metrics_mod
from .portfolio import SimulationResult
from .ranking import RankingResult

logger = logging.getLogger(__name__)


def summary_table(all_metrics: list[metrics_mod.Metrics]) -> pd.DataFrame:
    rows = []
    for m in all_metrics:
        rows.append(
            {
                "strategy": m.strategy,
                "variant": m.variant,
                "final_value": m.final_value,
                "total_contributed": m.total_contributed,
                "xirr_money_weighted": m.xirr,
                "twr_annualized": m.twr_annualized,
                "annualized_vol": m.annualized_vol,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "max_drawdown": m.max_drawdown,
                "drawdown_peak_date": m.drawdown_peak_date,
                "drawdown_trough_date": m.drawdown_trough_date,
                "best_year": m.best_year,
                "best_year_return": m.best_year_return,
                "worst_year": m.worst_year,
                "worst_year_return": m.worst_year_return,
                "distinct_positions": m.distinct_positions,
                "avg_annual_turnover": m.avg_annual_turnover,
            }
        )
    return pd.DataFrame(rows)


def per_year_basket_table(rankings_by_strategy: dict[str, dict[int, RankingResult]]) -> pd.DataFrame:
    """One row per (strategy, year, held ticker) so the picks can be eyeballed."""
    rows = []
    for strat_name, by_year in rankings_by_strategy.items():
        for year, res in sorted(by_year.items()):
            n = int("".join(filter(str.isdigit, strat_name)) or 0)
            basket = res.basket.get(n, [])
            ranked_lookup = res.ranked.set_index("ticker")
            for pos, ticker in enumerate(basket, start=1):
                prior_return = ranked_lookup.loc[ticker, "return"] if ticker in ranked_lookup.index else None
                rows.append(
                    {
                        "strategy": strat_name,
                        "holding_year": year,
                        "ranking_year": res.ranking_year,
                        "position": pos,
                        "ticker": ticker,
                        "prior_year_return": prior_return,
                        "n_eligible_members": res.n_membership,
                        "n_dropped_partial_year": res.n_dropped_partial_year,
                        "n_missing_price_data": res.n_missing_price_data,
                    }
                )
    return pd.DataFrame(rows)


def plot_value_over_time(
    results: dict[str, SimulationResult], contributions: pd.Series, out_path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, res in results.items():
        ax.plot(res.monthly_value.index, res.monthly_value.values, label=label, linewidth=1.6)
    ax.plot(contributions.index, contributions.values, label="Cumulative contributions", color="black", linestyle="--", linewidth=1.4)
    ax.set_yscale("log")
    ax.set_title("Portfolio value over time (log scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($, log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdowns(results: dict[str, SimulationResult], out_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, res in results.items():
        r = metrics_mod.monthly_twr_returns(res.monthly_value, res.monthly_contribution)
        idx = metrics_mod.twr_index(r)
        dd = idx / idx.cummax() - 1.0
        ax.plot(dd.index, dd.values * 100, label=label, linewidth=1.4)
    ax.set_title("Drawdown of strategy returns (time-weighted, % from peak)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


SLEEVE_COLORS = {
    "metals_bonds": "#B8860B",
    "sp_top10": "#1f77b4",
    "asia_top10": "#d62728",
    "index_fund": "#2ca02c",
    "cash": "#7f7f7f",
    "high_risk": "#9467bd",
}
SLEEVE_LABELS = {
    "metals_bonds": "Metals & Bonds (10%)",
    "sp_top10": "S&P Top 10 (20%)",
    "asia_top10": "Asia Top 10 (20%)",
    "index_fund": "Index Fund (20%)",
    "cash": "Cash (20%, crash-deployed)",
    "high_risk": "High Risk (BTC, 10%)",
}


def plot_sleeve_composition(sleeve_monthly_value: pd.DataFrame, crash_dates: list, out_path) -> None:
    """Stacked-area chart of the diversified strategy's six sleeves over
    time, with vertical markers where the crash-deployment rule fired."""
    fig, ax = plt.subplots(figsize=(12, 7))
    cols = [c for c in SLEEVE_COLORS if c in sleeve_monthly_value.columns]
    values = [sleeve_monthly_value[c].clip(lower=0).values for c in cols]
    ax.stackplot(
        sleeve_monthly_value.index,
        *values,
        labels=[SLEEVE_LABELS.get(c, c) for c in cols],
        colors=[SLEEVE_COLORS[c] for c in cols],
        alpha=0.85,
    )
    for i, d in enumerate(crash_dates):
        ax.axvline(d, color="black", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(
            d, ax.get_ylim()[1] * 0.97, f" crash {i + 1}", rotation=90, va="top", ha="left", fontsize=7,
        )
    ax.set_title("Diversified strategy: sleeve composition over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_allocation_pie(weights: dict[str, float], out_path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    labels = [SLEEVE_LABELS.get(k, k) for k in weights]
    colors = [SLEEVE_COLORS.get(k, "#999999") for k in weights]
    ax.pie(
        list(weights.values()), labels=labels, autopct="%1.0f%%", colors=colors,
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 8},
    )
    ax.set_title("Diversified strategy: target monthly-contribution allocation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_bars(all_metrics: list[metrics_mod.Metrics], labels: list[str], out_path) -> None:
    """Grouped bar chart comparing XIRR, Sharpe, and max drawdown across
    strategies -- the three numbers most people actually compare."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xirrs = [m.xirr * 100 for m in all_metrics]
    sharpes = [m.sharpe for m in all_metrics]
    dds = [m.max_drawdown * 100 for m in all_metrics]
    colors = plt.cm.tab10.colors[: len(labels)]

    for ax, values, title, ylabel in (
        (axes[0], xirrs, "Money-weighted return (XIRR)", "%"),
        (axes[1], sharpes, "Sharpe ratio", "ratio"),
        (axes[2], dds, "Max drawdown", "%"),
    ):
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        for b, v in zip(bars, values):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("Headline metric comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_all_csv(
    out_dir,
    summary_df: pd.DataFrame,
    basket_df: pd.DataFrame,
    results: dict[str, SimulationResult],
    failed_tickers_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    basket_df.to_csv(out_dir / "annual_baskets.csv", index=False)
    failed_tickers_df.to_csv(out_dir / "failed_tickers.csv", index=False)

    value_df = pd.DataFrame({label: r.monthly_value for label, r in results.items()})
    value_df.to_csv(out_dir / "portfolio_value_timeseries.csv")

    tx_rows = []
    for label, r in results.items():
        for t in r.transactions:
            tx_rows.append(
                {
                    "strategy_variant": label,
                    "date": t.date,
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": t.shares,
                    "price": t.price,
                    "gross_amount": t.gross_amount,
                    "slippage_cost": t.slippage_cost,
                    "tax": t.tax,
                    "net_amount": t.net_amount,
                    "reason": t.reason,
                }
            )
    pd.DataFrame(tx_rows).to_csv(out_dir / "transactions.csv", index=False)
