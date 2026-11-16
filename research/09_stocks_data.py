"""Data for the single-stock study: who was in the S&P 500, when, and what they traded at.

Testing a stock rule on today's large companies is the classic mistake: it only buys
names already known to have won. This script instead takes the index membership as it
was on each date (the public record kept at github.com/fja05680/sp500, MIT licence,
1996 onwards) and downloads every company that was ever a member since 2005, including
the ones that were later dropped, acquired or went bankrupt.

Yahoo no longer serves many dead tickers, so the survivorship problem is reduced, not
removed. The script measures the gap: for each year, the share of that year's members
with prices. The study reports it next to every result.

    python research/09_stocks_data.py
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from common import ROOT

OUT = ROOT / "data" / "research"
MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
START = "2004-01-01"            # a year of history before the first decision date
BATCH = 80

# The membership record uses the ticker of the day; Yahoo files a company's whole history
# under its latest one. Without this table a renamed company has no prices for the years
# it traded under the old name, which would quietly drop winners (Facebook, Priceline) and
# losers alike (Fannie Mae and Freddie Mac after 2008, the banks that failed in 2023).
# Only one-to-one renames and the over-the-counter tickers of failed companies are listed;
# mergers into another listed company are left out.
RENAMES = {
    "FB": "META", "PCLN": "BKNG", "KFT": "MDLZ", "UTX": "RTX", "HCN": "WELL", "LUK": "JEF", "DPS": "KDP",
    "ANTM": "ELV", "WLP": "ELV", "COG": "CTRA", "CTL": "LUMN", "HRS": "LHX", "BBT": "TFC", "KORS": "CPRI",
    "CBG": "CBRE", "TMK": "GL", "FISV": "FI", "ABC": "COR", "PKI": "RVTY", "FLT": "CPAY", "CDAY": "DAY",
    "PEAK": "DOC", "HCP": "DOC", "JEC": "J", "HFC": "DINO", "DISCA": "WBD", "SYMC": "GEN", "NLOK": "GEN",
    "WYN": "TNL", "RE": "EG", "GPS": "GAP", "BLL": "BALL", "ADS": "BFH", "FBHS": "FBIN", "LB": "BBWI",
    "COH": "TPR", "WAG": "WBA", "MHP": "SPGI", "MHFI": "SPGI", "PX": "LIN",
    "FNM": "FNMA", "FRE": "FMCC", "SIVB": "SIVBQ", "FRC": "FRCB", "LEH": "LEHMQ",
}


def membership() -> pd.DataFrame:
    """One row per change date; ``tickers`` is the full member list from that date."""
    path = OUT / "sp500_membership.csv"
    if not path.exists():
        r = requests.get(MEMBERSHIP_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(io.BytesIO(path.read_bytes()), parse_dates=["date"])
    df["tickers"] = df["tickers"].str.split(",")
    return df.sort_values("date").reset_index(drop=True)


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")          # BRK.B -> BRK-B


def download(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        for attempt in range(3):
            try:
                raw = yf.download([yahoo_symbol(t) for t in chunk], start=START, interval="1d", auto_adjust=True,
                                  progress=False, threads=True, group_by="ticker")
                break
            except Exception as exc:  # noqa: BLE001 - network hiccup: wait and retry
                print(f"  batch {i // BATCH}: {type(exc).__name__}, retry {attempt + 1}")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        got = 0
        for t in chunk:
            y = yahoo_symbol(t)
            if y not in raw.columns.get_level_values(0):
                continue
            df = raw[y].dropna(how="all").rename(columns=str.lower)
            if len(df) < 60:
                continue
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            rows.append(df[["open", "high", "low", "close", "volume"]].astype("float64").assign(symbol=t))
            got += 1
        print(f"  batch {i // BATCH + 1}/{-(-len(tickers) // BATCH)}: {got}/{len(chunk)} with prices", flush=True)
        time.sleep(1.0)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "symbol"], index=pd.DatetimeIndex([], name="date"))
    return pd.concat(rows)


def restore_renamed(prices: pd.DataFrame, members: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Give a renamed company its history back under the ticker the record uses. Needed
    when Yahoo has nothing under the old ticker, or has a different company there (tickers
    get reused: "FB" is no longer Facebook) whose prices never meet the membership dates."""
    first, last = {}, {}
    for date, lst in zip(members["date"], members["tickers"]):
        for t in lst:
            first.setdefault(t, date)
            last[t] = date
    span = prices.groupby("symbol").apply(lambda d: (d.index.min(), d.index.max()), include_groups=False)
    orphan = [old for old in RENAMES if old in tickers and (old not in span.index or span[old][0] > last[old] or span[old][1] < first[old])]
    extra = sorted({RENAMES[old] for old in orphan if RENAMES[old] not in span.index})
    pool = pd.concat([prices, download(extra)]) if extra else prices
    restored = [pool[pool["symbol"] == RENAMES[old]].assign(symbol=old) for old in orphan if (pool["symbol"] == RENAMES[old]).any()]
    print(f"  {len(restored)} renamed companies given their history back: {', '.join(r['symbol'].iloc[0] for r in restored)}")
    return pd.concat([prices[~prices["symbol"].isin([r["symbol"].iloc[0] for r in restored])], *restored])


def coverage(members: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per year: members of the index at any time that year, and how many have prices then."""
    span = prices.groupby("symbol").apply(lambda d: (d.index.min(), d.index.max()), include_groups=False)
    out = []
    for year in range(2005, members["date"].max().year + 1):
        rows = members[(members["date"] >= f"{year}-01-01") & (members["date"] <= f"{year}-12-31")]
        if rows.empty:
            continue
        names = set(t for lst in rows["tickers"] for t in lst)
        lo, hi = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        have = sum(1 for t in names if t in span.index and span[t][0] <= hi and span[t][1] >= lo)
        out.append({"year": year, "members": len(names), "with_prices": have, "share": have / len(names)})
    return pd.DataFrame(out).set_index("year")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    members = membership()
    since = members[members["date"] >= "2005-01-01"]
    first = members[members["date"] < "2005-01-01"].iloc[-1:]            # the list in force on 1 Jan 2005
    tickers = sorted(set(t for lst in pd.concat([first, since])["tickers"] for t in lst))
    print(f"membership {members['date'].min():%Y-%m-%d} -> {members['date'].max():%Y-%m-%d}; "
          f"{len(tickers)} companies were members at some point since 2005")
    prices = restore_renamed(download(tickers), members, tickers)
    prices.to_parquet(OUT / "stocks_daily.parquet")
    cov = coverage(members, prices)
    cov.to_csv(OUT / "stocks_coverage.csv")
    print(f"\nprices for {prices['symbol'].nunique()} of {len(tickers)} companies, {len(prices):,} daily bars")
    print(cov.assign(share=lambda d: (d["share"] * 100).round(1)).to_string())


if __name__ == "__main__":
    main()
