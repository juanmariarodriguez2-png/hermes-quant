"""Does a rule for picking single stocks beat simply holding the index?

The owner wants the model to buy individual companies, not only index funds. This
study asks whether a systematic rule earns its keep after costs, with the same
discipline as the rest of the project: decide on data to 2021, look at 2022 onwards once.

Universe      S&P 500 members as they stood on each date (not today's list), from the
              public membership record. Yahoo has lost many dead tickers, so the bias
              is reduced, not removed; 09_stocks_data.py measures the gap per year and
              a haircut on names whose prices stop is run as a sensitivity.
Decision      monthly, on the first session, from the previous close. Trades at that
              day's close. A holding is kept while it ranks in the top 2N (less churn).
Rules tried   MOM    12-month return skipping the last month (classic momentum)
              TREND  the model's own Kalman trend probability
              MOMG   momentum, only among names whose trend probability is above 50%
              each with N = 10 or 20 names, and with or without a market gate (out of
              stocks while the index's own trend probability is below 50%).
Sizing        inverse volatility, at most 2/N of the basket in one name.
Costs         0.20% a side (a commission-free broker: 0.15% currency + spread) as the
              base case; 1.07% a side (Kraken's Convert, measured) as the other.
Adoption rule fixed before looking: pick the variant with the best design-sample Sharpe
              among those turning over less than 6 times a year. Adopt it only if on
              the design sample it beats the index fund's Sharpe by 0.10 or more AND its
              alpha against the index has a t-statistic of at least 2. Otherwise the
              answer is "no evidence", whatever the later years show.

    python research/10_stock_sleeve.py            design sample only
    python research/10_stock_sleeve.py --holdout  also the years after 2021, for the chosen rule
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import RESULTS, ROOT
from hermes_quant.backtest.metrics import summary
from hermes_quant.backtest.stats import market_model
from hermes_quant.filters.kalman import trend_probability

DATA = ROOT / "data" / "research"
START, SPLIT = "2007-01-01", "2021-12-31"      # the same windows as the main model
TRADING_DAYS = 252
COST_BROKER, COST_CONVERT = 0.0020, 0.0107
MIN_HISTORY = 400
MAX_TURNOVER, MIN_EDGE, MIN_ALPHA_T = 6.0, 0.10, 2.0
FAVOURITES = ["NVDA", "TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "META"]


def load() -> dict:
    px = pd.read_parquet(DATA / "stocks_daily.parquet")
    close = px.pivot_table(index=px.index, columns="symbol", values="close").sort_index()
    ref = pd.read_parquet(DATA / "yahoo_daily.parquet")
    spy = ref[ref["symbol"] == "SPY"]["close"].sort_index()
    close = close.reindex(spy.index)                                  # the index fund's sessions are the calendar
    members = pd.read_csv(DATA / "sp500_membership.csv", parse_dates=["date"]).sort_values("date")
    mask = np.zeros(close.shape, dtype=bool)
    col = {s: j for j, s in enumerate(close.columns)}
    dates = close.index.to_numpy()
    starts = np.searchsorted(dates, members["date"].to_numpy(), side="left")
    ends = np.append(starts[1:], len(dates))
    for a, b, tickers in zip(starts, ends, members["tickers"]):
        if b > a:
            mask[a:b, [col[t] for t in tickers.split(",") if t in col]] = True
    listed = np.array([len(set(t.split(","))) for t in members["tickers"]])
    return {"close": close, "spy": spy, "member": pd.DataFrame(mask, index=close.index, columns=close.columns),
            "members_listed": pd.Series(listed, index=members["date"].to_numpy())}


def signals(close: pd.DataFrame, spy: pd.Series) -> dict:
    logret = np.log(close).diff()
    var = logret ** 2
    vol = np.sqrt((0.7 * var.ewm(halflife=10, min_periods=60).mean() + 0.3 * var.ewm(halflife=126, min_periods=60).mean()) * TRADING_DAYS)
    ones = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    trend = trend_probability(logret, vol, periods_per_year=TRADING_DAYS, spans=ones)["prob"]
    mom = close.shift(21) / close.shift(252) - 1.0
    s = spy.to_frame("SPY")
    slog = np.log(s).diff()
    svol = np.sqrt((0.7 * (slog ** 2).ewm(halflife=10, min_periods=60).mean() + 0.3 * (slog ** 2).ewm(halflife=126, min_periods=60).mean()) * TRADING_DAYS)
    market = trend_probability(slog, svol, periods_per_year=TRADING_DAYS, spans=pd.DataFrame(1.0, index=s.index, columns=s.columns))["prob"]["SPY"]
    return {"vol": vol, "trend": trend, "mom": mom, "market": market, "history": close.notna().cumsum()}


def simulate(data: dict, sig: dict, rule: str, n: int, gate: bool, cost: float, *, start: str = START,
             end: str | None = None, haircut: float = 0.0, universe: list[str] | None = None) -> dict:
    """Daily simulation of one basket. Returns its equity curve (starting at 1) and bookkeeping."""
    close, member = data["close"], data["member"]
    c = close.to_numpy()
    days = close.index
    rets = np.nan_to_num(c[1:] / c[:-1] - 1.0, nan=0.0)
    rets = np.vstack([np.zeros(c.shape[1]), rets])
    alive_later = np.flip(np.maximum.accumulate(np.flip(np.isfinite(c), axis=0), axis=0), axis=0)   # a price exists on or after t
    mom, trend, vol, hist = (sig[k].to_numpy() for k in ("mom", "trend", "vol", "history"))
    market = sig["market"].to_numpy()
    mem = member.to_numpy()
    if universe is not None:
        allowed = np.isin(close.columns, universe)
        mem = np.ones_like(mem) & allowed[None, :]
    first = np.searchsorted(days.to_numpy(), np.datetime64(start))
    last = len(days) if end is None else np.searchsorted(days.to_numpy(), np.datetime64(end), side="right")
    month = days.month.to_numpy()
    w = np.zeros(c.shape[1])
    equity, curve, traded, paid, held_count, lost = 1.0, [], 0.0, 0.0, [], 0.0
    for t in range(first, last):
        if t > first:
            gross = float(w @ rets[t])
            equity *= 1.0 + gross
            w = w * (1.0 + rets[t]) / (1.0 + gross) if gross > -1.0 else w * 0.0
        gone = (w > 0) & ~np.isfinite(c[t]) & ~alive_later[t]          # prices stopped for good: delisted or bought out
        if gone.any():
            loss = float(w[gone].sum()) * haircut
            equity *= 1.0 - loss
            lost += loss
            w[gone] = 0.0
            w = w / (1.0 - loss) if loss < 1.0 else w
        if t == first or month[t] != month[t - 1]:                      # first session of the month: decide from yesterday's close
            y = t - 1
            score = mom[y] if rule in ("MOM", "MOMG") else trend[y]
            ok = mem[y] & np.isfinite(c[y]) & np.isfinite(c[t]) & (hist[y] >= MIN_HISTORY) & np.isfinite(score) & np.isfinite(vol[y]) & (vol[y] > 0)
            if rule in ("MOMG", "TRENDG"):
                ok &= trend[y] > 0.5
            if rule == "ALL":                                           # every member, equally: the "no selection" yardstick
                target = np.where(ok, 1.0, 0.0)
                target = target / target.sum() if target.sum() else target
            else:
                order = np.argsort(-np.where(ok, score, -np.inf))
                rank = np.empty(len(order), dtype=int)
                rank[order] = np.arange(len(order))
                keep = (w > 0) & ok & (rank < 2 * n)
                chosen = list(np.flatnonzero(keep))
                for j in order:
                    if len(chosen) >= n or not ok[j]:
                        break
                    if j not in chosen:
                        chosen.append(j)
                target = np.zeros_like(w)
                if chosen and (not gate or market[y] > 0.5):
                    inv = 1.0 / vol[y][chosen]
                    share = inv / inv.sum()
                    for _ in range(5):                                  # cap one name at 2/N, hand the excess to the others
                        over = share > 2.0 / n
                        if not over.any():
                            break
                        share = np.where(over, 2.0 / n, share * (1.0 - (2.0 / n) * over.sum()) / max(share[~over].sum(), 1e-12))
                    target[chosen] = share
            turn = float(np.abs(target - w).sum())
            equity *= 1.0 - turn * cost
            traded += turn
            paid += turn * cost
            w = target
        curve.append(equity)
        held_count.append(int((w > 0).sum()))
    eq = pd.Series(curve, index=days[first:last])
    years = len(eq) / TRADING_DAYS
    return {"equity": eq, "turnover_per_year": traded / 2.0 / years, "cost_per_year": paid / years,
            "avg_names": float(np.mean(held_count)), "lost_to_delisting": lost}


def stats(eq: pd.Series, bench: pd.Series) -> dict:
    s = summary(eq, periods=TRADING_DAYS)
    wk = pd.concat([eq, bench], axis=1, keys=["y", "index"]).resample("W-FRI").last().pct_change().dropna()
    mm = market_model(wk["y"], wk[["index"]])
    active = wk["y"] - wk["index"]
    return {"cagr": s["cagr"], "vol": s["vol"], "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"],
            "alpha_annual": mm["alpha_annual"], "alpha_t": mm["alpha_t"], "beta": mm["beta_index"],
            "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(52)) if active.std(ddof=1) > 0 else float("nan")}


def window(eq: pd.Series, lo: str | None, hi: str | None) -> pd.Series:
    part = eq.loc[lo:hi]
    return part / part.iloc[0]


def main() -> None:
    holdout = "--holdout" in sys.argv
    data = load()
    sig = signals(data["close"], data["spy"])
    spy = data["spy"].loc[START:]
    spy = spy / spy.iloc[0]
    out: dict = {"window": {"start": START, "split": SPLIT, "end": str(data["close"].index[-1].date())},
                 "costs": {"broker": COST_BROKER, "convert": COST_CONVERT}, "design": {}}
    cov = pd.read_csv(DATA / "stocks_coverage.csv", index_col=0)
    out["coverage"] = {int(y): float(v) for y, v in cov["share"].items()}

    bench_d = stats(window(spy, START, SPLIT), window(spy, START, SPLIT))
    bench_d.update(alpha_annual=0.0, alpha_t=float("nan"), beta=1.0)       # the yardstick against itself
    out["design"]["index fund (SPY)"] = bench_d
    print(f"design sample {START} -> {SPLIT}; price coverage of members: "
          f"{cov['share'].loc[2007:2021].min():.0%} to {cov['share'].loc[2007:2021].max():.0%} a year\n")
    head = f"{'':34s} {'CAGR':>7s} {'vol':>6s} {'Sharpe':>6s} {'maxDD':>7s} {'alpha':>7s} {'t':>5s} {'IR':>5s} {'turn':>5s} {'cost':>6s}"
    print(head)
    line = lambda name, s, r=None: print(f"{name:34s} {s['cagr']:7.1%} {s['vol']:6.1%} {s['sharpe']:6.2f} {s['max_drawdown']:7.1%} "       # noqa: E731
                                         f"{s['alpha_annual']:7.1%} {s['alpha_t']:5.1f} {s['information_ratio']:5.2f} "
                                         + (f"{r['turnover_per_year']:5.1f} {r['cost_per_year']:6.2%}" if r else ""))
    line("index fund (SPY)", bench_d)
    runs = {}
    until = None if holdout else SPLIT                                  # without --holdout nothing after 2021 is even computed
    everyone = simulate(data, sig, "ALL", 0, False, 0.0, end=until)
    out["design"]["all members, equal weight, no costs"] = stats(window(everyone["equity"], START, SPLIT), window(spy, START, SPLIT))
    line("all members, equal weight", out["design"]["all members, equal weight, no costs"], everyone)
    for rule in ("MOM", "TREND", "MOMG"):
        for n in (10, 20):
            for gate in (False, True):
                name = f"{rule} top {n}{' + market gate' if gate else ''}"
                d = simulate(data, sig, rule, n, gate, COST_BROKER, end=SPLIT)
                s = stats(window(d["equity"], START, SPLIT), window(spy, START, SPLIT))
                s.update(turnover_per_year=d["turnover_per_year"], cost_per_year=d["cost_per_year"], avg_names=d["avg_names"])
                runs[name] = (rule, n, gate)
                out["design"][name] = s
                line(name, s, d)

    eligible = {k: v for k, v in out["design"].items() if k in runs and v["turnover_per_year"] < MAX_TURNOVER}
    best = max(eligible, key=lambda k: eligible[k]["sharpe"]) if eligible else None
    edge = out["design"][best]["sharpe"] - bench_d["sharpe"] if best else float("nan")
    adopt = bool(best and edge >= MIN_EDGE and out["design"][best]["alpha_t"] >= MIN_ALPHA_T)
    out["selection"] = {"rule": "best design Sharpe with turnover below 6x a year", "chosen": best, "sharpe_edge_over_index": edge,
                        "alpha_t": out["design"][best]["alpha_t"] if best else None, "adopted": adopt,
                        "bar": {"min_sharpe_edge": MIN_EDGE, "min_alpha_t": MIN_ALPHA_T}, "variants_tried": len(runs)}
    print(f"\nchosen by the rule: {best}  (Sharpe {out['design'][best]['sharpe']:.2f} against the index fund's {bench_d['sharpe']:.2f}; "
          f"alpha t = {out['design'][best]['alpha_t']:.1f})")
    print("ADOPT" if adopt else "DO NOT ADOPT: on the design sample it does not clear the bar set beforehand")

    rule, n, gate = runs[best]
    out["sensitivity_design"] = {}
    for label, kw in (("at Kraken's Convert cost (1.07% a side)", dict(cost=COST_CONVERT)),
                      ("names whose prices stop lose 30%", dict(cost=COST_BROKER, haircut=0.30))):
        r = simulate(data, sig, rule, n, gate, end=SPLIT, **kw)
        s = stats(window(r["equity"], START, SPLIT), window(spy, START, SPLIT))
        out["sensitivity_design"][label] = s
        line(f"  {best}: {label}"[:34], s, r)

    # the owner's own list: every name on it is known today to have won. Not evidence.
    fav = [f for f in FAVOURITES if f in data["close"].columns]
    if holdout:                                                         # it runs to today, so it waits for the flag too
        r = simulate(data, sig, "TRENDG", len(fav), False, COST_BROKER, start="2013-01-01", universe=fav)
        b = data["spy"].loc["2013-01-01":]
        out["favourites_hindsight"] = {"names": fav, "from": "2013-01-01", **stats(r["equity"], b / b.iloc[0])}
        print(f"\nthe owner's list ({', '.join(fav)}), each held only in an uptrend, since 2013: {out['favourites_hindsight']['cagr']:.1%} "
              f"a year, Sharpe {out['favourites_hindsight']['sharpe']:.2f}. Chosen with hindsight: this number proves nothing.")

    if holdout:
        full = simulate(data, sig, rule, n, gate, COST_BROKER)
        conv = simulate(data, sig, rule, n, gate, COST_CONVERT)
        everyone_h = window(everyone["equity"], SPLIT, None)
        b = window(spy, SPLIT, None)
        out["holdout"] = {"index fund (SPY)": {**stats(b, b), "alpha_annual": 0.0, "alpha_t": float("nan"), "beta": 1.0},
                          "all members, equal weight, no costs": stats(everyone_h, b),
                          best: stats(window(full["equity"], SPLIT, None), b),
                          f"{best}, at Kraken's Convert cost": stats(window(conv["equity"], SPLIT, None), b)}
        print(f"\nhold-out {SPLIT} -> {out['window']['end']}\n{head}")
        for k, v in out["holdout"].items():
            line(k[:34], v)
        pd.DataFrame({"stock basket": full["equity"], "index fund": spy}).to_csv(RESULTS / "stock_sleeve_curves.csv")
    (RESULTS / "stock_sleeve.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
