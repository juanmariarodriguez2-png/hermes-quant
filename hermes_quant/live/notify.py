"""Optional daily notification. Silent unless credentials are configured; a failed
message never interrupts trading."""

from __future__ import annotations

import os

import requests


def daily_message(out: dict) -> str:
    a = out["account"]
    lines = [f"Hermes Quant · {out['meta']['bar_date']} · {a['mode']}",
             f"Cartera: {a['nav']:.2f} EUR ({a['return_since_inception']:+.2%} desde {a['inception']})",
             f"Invertido: {a['invested_pct']:.0%} · caída desde máximo: {a['drawdown']:.1%}"]
    if a["halted"]:
        lines.append("ATENCIÓN: cortacircuitos activado, cartera en liquidez.")
    for f in a["orders_today"]:
        verb = "Compra" if f["notional_eur"] > 0 else "Venta"
        lines.append(f"{verb} {f['symbol']}: {abs(f['notional_eur']):.2f} EUR ({f['status']})")
    if not a["orders_today"]:
        lines.append("Sin órdenes hoy.")
    for o in a.get("manual_orders", []):
        verb = "comprar" if o["notional_eur"] > 0 else "vender"
        lines.append(f"A MANO en kraken.com: {verb} {abs(o['notional_eur']):.0f} EUR de {o['pair'][:-3]}")
    held = [f"{p['symbol']} {p['weight']:.0%}" for p in a["positions"] if p["weight"] > 0.005]
    lines.append("Posiciones: " + (", ".join(held) if held else "todo en liquidez"))
    return "\n".join(lines)


def send(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text}, timeout=20)
        return r.ok
    except requests.RequestException:
        return False
