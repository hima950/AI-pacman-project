#!/usr/bin/env python3
"""Generate a DEMO dataset for the dashboard by running the real, tested
pipeline (ranking -> portfolio -> diversified -> metrics -> interpretation)
against a fabricated synthetic universe, since this sandbox cannot reach
Yahoo Finance or Wikipedia for real data (see README.md).

Every number in the resulting dashboard is genuinely computed by the same
code that would run on real data -- nothing here is hand-typed into the
dashboard -- but the *inputs* (prices, membership) are synthetic and
clearly marked as such throughout. Reproducible via a fixed random seed.

Run: python scripts/generate_demo_data.py
Writes: dashboard/demo_data.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config, diversified, interpretation, metrics as metrics_mod, portfolio, ranking
from src.membership import MembershipData, constant_membership

RNG = np.random.default_rng(20260904)

START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2024-12-31")
HOLDING_YEARS = range(2000, 2025)
RANKING_YEARS = range(1999, 2024)


def gbm_series(start, end, mu, sigma, base_price, seed_offset=0):
    """Geometric-Brownian-motion-ish daily price path (fabricated)."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(20260904 + seed_offset)
    daily_mu = mu / 252
    daily_sigma = sigma / np.sqrt(252)
    shocks = rng.normal(daily_mu, daily_sigma, len(idx))
    log_prices = np.cumsum(shocks)
    prices = base_price * np.exp(log_prices)
    return pd.Series(prices, index=idx)


def apply_crash(series: pd.Series, crash_start: str, trough: str, recover_by: str, depth: float) -> pd.Series:
    """Overlay a multiplicative drawdown-and-recovery shape onto a series,
    scaled by that series' own beta-ish sensitivity (depth param)."""
    s = series.copy()
    idx = s.index
    t0, t1, t2 = pd.Timestamp(crash_start), pd.Timestamp(trough), pd.Timestamp(recover_by)
    down_mask = (idx >= t0) & (idx <= t1)
    up_mask = (idx > t1) & (idx <= t2)
    if down_mask.any():
        frac = np.linspace(0, 1, down_mask.sum())
        mult = 1 - depth * frac
        s.loc[down_mask] = s.loc[down_mask].values * mult
    if up_mask.any():
        frac = np.linspace(0, 1, up_mask.sum())
        bottom_mult = 1 - depth
        mult = bottom_mult + (1 - bottom_mult) * frac
        s.loc[up_mask] = s.loc[up_mask].values * mult
    if idx[-1] > t2:
        after_mask = idx > t2
        last_mult = s.loc[idx[idx <= t2][-1]] / series.loc[idx[idx <= t2][-1]] if (idx <= t2).any() else 1.0
        s.loc[after_mask] = s.loc[after_mask].values * last_mult
    return s


def build_universe():
    spy = gbm_series(START - pd.Timedelta(days=730), END, mu=0.075, sigma=0.16, base_price=100.0, seed_offset=0)
    spy = apply_crash(spy, "2000-03-15", "2002-10-01", "2007-05-01", depth=0.47)
    spy = apply_crash(spy, "2008-01-01", "2009-03-09", "2013-03-01", depth=0.53)
    spy = apply_crash(spy, "2020-02-15", "2020-03-23", "2020-08-15", depth=0.32)
    spy = apply_crash(spy, "2022-01-03", "2022-10-12", "2024-01-19", depth=0.24)

    sp_names = [f"SIM_{c}" for c in "ABCDEFGHIJKLMNOPQR"]
    sp_prices = {}
    for i, name in enumerate(sp_names):
        mu = RNG.uniform(0.04, 0.22)
        sigma = RNG.uniform(0.20, 0.55)
        s = gbm_series(START - pd.Timedelta(days=730), END, mu=mu, sigma=sigma, base_price=RNG.uniform(15, 120), seed_offset=10 + i)
        s = apply_crash(s, "2000-03-15", "2002-10-01", "2007-05-01", depth=min(0.9, 0.47 * RNG.uniform(0.6, 1.6)))
        s = apply_crash(s, "2008-01-01", "2009-03-09", "2013-03-01", depth=min(0.9, 0.53 * RNG.uniform(0.6, 1.6)))
        s = apply_crash(s, "2020-02-15", "2020-03-23", "2020-08-15", depth=min(0.9, 0.32 * RNG.uniform(0.6, 1.6)))
        sp_prices[name] = s

    # one mid-2015 IPO (must be excluded from the 2016 ranking, eligible from 2017)
    sp_prices["SIM_IPO"] = gbm_series("2015-07-01", END, mu=0.18, sigma=0.4, base_price=25.0, seed_offset=99)
    # one delisting mid-2013 (forced-sale/redistribution texture)
    delisted = gbm_series(START - pd.Timedelta(days=730), "2013-06-14", mu=0.05, sigma=0.3, base_price=40.0, seed_offset=77)
    sp_prices["SIM_DELISTED"] = delisted

    sp_all_tickers = set(sp_prices.keys())
    sp_yearend_rows = [{"as_of_year_end": y, "snapshot_date": None, "tickers": set(sp_all_tickers)} for y in range(1998, 2025)]
    sp_mem = MembershipData(yearend=pd.DataFrame(sp_yearend_rows), ticker_ranges=None, source="synthetic-demo")

    asia_names = [f"ASIA_{c}" for c in "ABCDEFGHIJ"]
    asia_prices = {}
    for i, name in enumerate(asia_names):
        mu = RNG.uniform(0.03, 0.20)
        sigma = RNG.uniform(0.22, 0.5)
        s = gbm_series(START - pd.Timedelta(days=730), END, mu=mu, sigma=sigma, base_price=RNG.uniform(10, 90), seed_offset=200 + i)
        s = apply_crash(s, "2008-01-01", "2009-03-09", "2013-03-01", depth=min(0.9, 0.5 * RNG.uniform(0.6, 1.5)))
        asia_prices[name] = s
    asia_mem = constant_membership(set(asia_names), range(1999, 2025), source_label="synthetic-demo-asia")

    gld = gbm_series(START - pd.Timedelta(days=730), END, mu=0.055, sigma=0.16, base_price=280.0, seed_offset=300)
    agg = gbm_series(START - pd.Timedelta(days=730), END, mu=0.028, sigma=0.06, base_price=90.0, seed_offset=301)
    btc = gbm_series("2014-09-17", END, mu=0.55, sigma=0.85, base_price=1.0, seed_offset=302)
    btc = apply_crash(btc, "2018-01-07", "2018-12-15", "2020-11-01", depth=0.83)
    btc = apply_crash(btc, "2021-11-08", "2022-11-21", "2024-02-01", depth=0.77)

    all_prices = dict(sp_prices)
    all_prices.update(asia_prices)
    all_prices["SPY"] = spy
    all_prices[config.METALS_TICKER] = gld
    all_prices[config.BONDS_TICKER] = agg
    all_prices[config.HIGH_RISK_TICKER] = btc

    irx = pd.Series(np.clip(2.0 + 3.0 * np.sin(np.linspace(0, 8, len(pd.bdate_range(START, END)))) + RNG.normal(0, 0.3, len(pd.bdate_range(START, END))), 0.01, 6.0), index=pd.bdate_range(START, END))

    return all_prices, sp_mem, asia_mem, spy, irx


def markdown_lite_to_html(text: str) -> str:
    """Just enough markdown -> HTML for interpretation.build_report's output
    (headers, bold, bullet lists, paragraphs) -- avoids shipping a markdown
    parser to the browser for a handful of predictable patterns."""
    import re

    lines = text.strip().split("\n")
    html = []
    in_list = False
    for line in lines:
        line = line.rstrip()
        if not line:
            # CommonMark treats consecutive "- " bullets separated by blank
            # lines as one "loose" list, not separate lists -- so a blank
            # line alone shouldn't close the list; only a genuine change of
            # block type (header / plain paragraph) should.
            continue
        line_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line_html)
        if line.startswith("## "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h3>{line_html[3:]}</h3>")
        elif line.startswith("# "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h2>{line_html[2:]}</h2>")
        elif line.startswith("- "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{line_html[2:]}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p>{line_html}</p>")
    if in_list:
        html.append("</ul>")
    return "\n".join(html)


def to_series_json(s: pd.Series) -> list[list]:
    return [[d.strftime("%Y-%m-%d"), None if pd.isna(v) else round(float(v), 2)] for d, v in s.items()]


def main():
    print("Building synthetic demo universe...")
    all_prices, sp_mem, asia_mem, spy, irx = build_universe()

    print("Ranking S&P and Asia baskets...")
    sp_rankings = ranking.rank_all_years(HOLDING_YEARS, sp_mem, all_prices, [5, 10])
    asia_rankings = ranking.rank_all_years(HOLDING_YEARS, asia_mem, all_prices, [10])

    basket_index = {y: [config.INDEX_TICKER] for y in HOLDING_YEARS}
    basket_top5 = {y: sp_rankings[y].basket[5] for y in sp_rankings}
    basket_top10 = {y: sp_rankings[y].basket[10] for y in sp_rankings}

    print("Simulating Index / Top5 / Top10 (buy & hold)...")
    results = {}
    results["index"] = portfolio.simulate("index", "buy_and_hold", basket_index, all_prices, spy, start=START, end=END, single_asset_no_rebalance=True)
    results["top5"] = portfolio.simulate("top5", "buy_and_hold", basket_top5, all_prices, spy, start=START, end=END)
    results["top10"] = portfolio.simulate("top10", "buy_and_hold", basket_top10, all_prices, spy, start=START, end=END)

    rf_monthly = {}
    for d in results["index"].monthly_value.index:
        sub = irx[irx.index <= d]
        rf_monthly[d] = (float(sub.iloc[-1]) / 100.0) / 12.0 if not sub.empty else config.RISK_FREE_FLAT_FALLBACK / 12.0
    rf_monthly = pd.Series(rf_monthly)

    print("Simulating Diversified (6 sleeves)...")
    div = diversified.build_diversified_strategy(
        holding_years=HOLDING_YEARS, membership=sp_mem, all_prices=all_prices, spy_prices=spy,
        sp_top10_basket_by_year=basket_top10, asia_rankings=asia_rankings,
        start=START, end=END, risk_free_monthly=rf_monthly,
    )
    results["diversified"] = div.combined

    print("Computing metrics...")
    all_metrics = {k: metrics_mod.compute_metrics(v, rf_monthly) for k, v in results.items()}

    STRAT_META = {
        "index": {"label": "Index (SPY)"},
        "top5": {"label": "Top 5"},
        "top10": {"label": "Top 10"},
        "diversified": {"label": "Diversified"},
    }

    strategies_out = {}
    for key, res in results.items():
        r = metrics_mod.monthly_twr_returns(res.monthly_value, res.monthly_contribution)
        idx = metrics_mod.twr_index(r)
        dd = (idx / idx.cummax() - 1.0)
        m = all_metrics[key]
        strategies_out[key] = {
            "label": STRAT_META[key]["label"],
            "value_series": to_series_json(res.monthly_value),
            "drawdown_series": to_series_json(dd * 100),
            "metrics": {
                "final_value": round(m.final_value, 2),
                "total_contributed": round(m.total_contributed, 2),
                "xirr": round(m.xirr, 4),
                "twr_annualized": round(m.twr_annualized, 4),
                "annualized_vol": round(m.annualized_vol, 4),
                "sharpe": round(m.sharpe, 3) if np.isfinite(m.sharpe) else None,
                "sortino": round(m.sortino, 3) if np.isfinite(m.sortino) else None,
                "max_drawdown": round(m.max_drawdown, 4),
                "drawdown_peak_date": m.drawdown_peak_date.strftime("%Y-%m-%d") if m.drawdown_peak_date is not None else None,
                "drawdown_trough_date": m.drawdown_trough_date.strftime("%Y-%m-%d") if m.drawdown_trough_date is not None else None,
                "best_year": m.best_year,
                "best_year_return": round(m.best_year_return, 4),
                "worst_year": m.worst_year,
                "worst_year_return": round(m.worst_year_return, 4),
                "distinct_positions": m.distinct_positions,
                "avg_annual_turnover": round(m.avg_annual_turnover, 4),
            },
        }

    cum_contrib = to_series_json(pd.Series(
        [config.MONTHLY_CONTRIBUTION * (i + 1) for i in range(len(results["index"].monthly_value))],
        index=results["index"].monthly_value.index,
    ))

    sleeve_out = {name: to_series_json(r.monthly_value) for name, r in div.sleeve_results.items()}
    crash_events_out = [
        {
            "date": e.date.strftime("%Y-%m-%d"),
            "spy_drawdown": round(e.spy_drawdown, 4),
            "cash_deployed": round(e.cash_deployed, 2),
            "lookback_years_used": e.lookback_years_used,
            "winners": e.winners,
        }
        for e in div.crash_events
    ]

    def basket_rows(rankings, n, strategy_key):
        rows = []
        for year, res in sorted(rankings.items()):
            basket = res.basket.get(n, [])
            ranked_lookup = res.ranked.set_index("ticker")
            for pos, ticker in enumerate(basket, start=1):
                ret = ranked_lookup.loc[ticker, "return"] if ticker in ranked_lookup.index else None
                rows.append({
                    "strategy": strategy_key, "year": year, "position": pos, "ticker": ticker,
                    "prior_year_return": round(float(ret), 4) if ret is not None else None,
                })
        return rows

    baskets_out = (
        basket_rows(sp_rankings, 5, "top5")
        + basket_rows(sp_rankings, 10, "top10")
        + basket_rows(asia_rankings, 10, "asia_top10")
    )

    pretty_metrics = {STRAT_META[k]["label"]: v for k, v in all_metrics.items()}
    pretty_results = {STRAT_META[k]["label"]: v for k, v in results.items()}
    report_text = interpretation.build_report(
        pretty_metrics, pretty_results, "synthetic-demo (fabricated, not real market data)", 0, 1
    )

    out = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"),
        "is_demo_data": True,
        "start_date": START.strftime("%Y-%m-%d"),
        "end_date": END.strftime("%Y-%m-%d"),
        "monthly_contribution": config.MONTHLY_CONTRIBUTION,
        "cumulative_contributions": cum_contrib,
        "strategies": strategies_out,
        "diversified_weights": config.DIVERSIFIED_WEIGHTS,
        "sleeve_series": sleeve_out,
        "crash_events": crash_events_out,
        "baskets": baskets_out,
        "interpretation_markdown": report_text,
        "interpretation_html": markdown_lite_to_html(report_text),
    }

    out_dir = Path(__file__).resolve().parent.parent / "dashboard"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demo_data.json"
    out_path.write_text(json.dumps(out, indent=None, separators=(",", ":")))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
