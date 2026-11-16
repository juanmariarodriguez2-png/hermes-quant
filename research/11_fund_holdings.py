"""Which companies the index funds hold, so the dashboard can show what the account
already owns of each one through them.

A fund's ten largest positions, as published by Yahoo Finance. It is a snapshot: the
weights drift a little every day, and the file says when it was taken. Run it again
now and then.

    python research/11_fund_holdings.py
"""

from __future__ import annotations

import json
from datetime import date

from common import ROOT

FUNDS = ("SPY", "QQQ")
SAME_COMPANY = {"GOOG": "GOOGL"}          # two share classes of Alphabet count as one company


def main() -> None:
    import yfinance as yf

    out = {"as_of": date.today().isoformat(), "source": "Yahoo Finance, ten largest holdings of each fund", "funds": {}}
    for fund in FUNDS:
        top = yf.Ticker(fund).funds_data.top_holdings
        merged: dict[str, dict] = {}
        for symbol, row in top.iterrows():
            key = SAME_COMPANY.get(symbol, symbol)
            name = str(row["Name"]).replace(" Class A", "").replace(" Class C", "")
            entry = merged.setdefault(key, {"symbol": key, "name": name, "weight": 0.0})
            entry["weight"] = round(entry["weight"] + float(row["Holding Percent"]), 6)
        out["funds"][fund] = sorted(merged.values(), key=lambda e: -e["weight"])
        print(fund, ", ".join(f"{e['symbol']} {e['weight']:.1%}" for e in out["funds"][fund]))
    (ROOT / "site" / "holdings.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
