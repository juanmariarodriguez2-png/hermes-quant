"""Trading 212 account access, with pacing and no automatic write retries."""

from __future__ import annotations

import base64
import math
import time

import requests

# Separate names keep unrelated endpoints from needlessly delaying one another.
SPACING = {"summary": 5.0, "positions": 1.0, "instruments": 50.0, "exchanges": 30.0,
           "pending": 5.0, "order": 1.0, "history": 3.0, "market": 1.2, "cancel": 1.2}
MAX_RATE_WAIT = 60.0      # seconds: the longest the client will sit out a rate limit before trying again
PATHS = {"/equity/account/summary": "summary", "/equity/positions": "positions",
         "/equity/metadata/instruments": "instruments", "/equity/metadata/exchanges": "exchanges",
         "/equity/orders": "pending", "/equity/history/orders": "history", "/equity/orders/market": "market"}


class Trading212Error(RuntimeError):
    def __init__(self, body: str, status: int | None = None, uncertain: bool = False):
        self.status, self.body, self.uncertain = status, body, uncertain
        super().__init__(body)


class Trading212Client:
    def __init__(self, key: str, secret: str, *, env: str = "demo", session=None, sleep=time.sleep, clock=time.monotonic):
        if env not in ("demo", "live"):
            raise ValueError("env must be demo or live")
        self.env = env
        self._base = f"https://{env}.trading212.com/api/v0"
        self._session = session if session is not None else requests.Session()
        encoded = base64.b64encode(f"{key}:{secret}".encode()).decode()
        self._headers = {"Authorization": f"Basic {encoded}"}
        self._credentials = tuple(v for v in (key, secret, encoded) if v)
        self._sleep, self._clock = sleep, clock
        self._last: dict[str, float] = {}

    def _error(self, body: str, status=None, uncertain=False) -> Trading212Error:
        error = Trading212Error(body, status, uncertain)
        # Keep the original server body for inspection, but redact credentials if a server echoes them.
        message = body
        for credential in self._credentials:
            message = message.replace(credential, "[redacted]")
        error.args = (message,)
        return error

    def _pace(self, endpoint: str) -> None:
        if endpoint in self._last:
            delay = SPACING[endpoint] - (self._clock() - self._last[endpoint])
            if delay > 0:
                self._sleep(delay)
        self._last[endpoint] = self._clock()

    def _request(self, method, path, *, params=None, json=None):
        method = method.upper()
        endpoint = "cancel" if method == "DELETE" else PATHS.get(path, "order")
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            self._pace(endpoint)
            try:
                response = self._session.request(method, self._base + path, params=params, json=json,
                                                 headers=self._headers, timeout=30, allow_redirects=False)
            except requests.RequestException:
                if attempt + 1 < attempts:
                    self._sleep(0.5 * (attempt + 1))
                    continue
                raise self._error("request failed (transport error)", uncertain=method in ("POST", "DELETE")) from None
            status = response.status_code
            if 200 <= status < 300:
                if method == "DELETE" or status == 204:
                    return None
                try:
                    return response.json()
                except ValueError:
                    raise self._error("invalid JSON response", status, method == "POST") from None
            retryable = status in (408, 429) or status >= 500
            if retryable and attempt + 1 < attempts:
                if status == 429:
                    try:
                        delay = float(response.headers.get("x-ratelimit-reset", "0")) - time.time()
                    except (ValueError, TypeError):
                        delay = 0
                    # a reset far in the future (a clock that disagrees with ours) must not hang the run
                    self._sleep(min(delay, MAX_RATE_WAIT) if math.isfinite(delay) and delay > 0 else SPACING[endpoint])
                else:
                    self._sleep(0.5 * (attempt + 1))
                continue
            uncertain = method in ("POST", "DELETE") and (status == 408 or status >= 500)
            raise self._error(response.text, status, uncertain)

    def summary(self) -> dict:
        return self._request("GET", "/equity/account/summary")

    def positions(self) -> list[dict]:
        return self._request("GET", "/equity/positions")

    def instruments(self) -> list[dict]:
        return self._request("GET", "/equity/metadata/instruments")

    def exchanges(self) -> list[dict]:
        return self._request("GET", "/equity/metadata/exchanges")

    def pending_orders(self) -> list[dict]:
        return self._request("GET", "/equity/orders")

    def order(self, order_id: int) -> dict:
        return self._request("GET", f"/equity/orders/{int(order_id)}")

    def history(self, ticker: str | None = None, limit: int = 50) -> list[dict]:
        if not 1 <= limit <= 50:
            raise ValueError("history limit must be between 1 and 50")
        params = {"limit": limit}
        if ticker is not None:
            params["ticker"] = ticker
        return self._request("GET", "/equity/history/orders", params=params)["items"]

    def market_order(self, ticker: str, quantity: float, extended_hours: bool = False) -> dict:
        if not math.isfinite(quantity) or quantity == 0:
            raise ValueError("quantity must be finite and nonzero")
        return self._request("POST", "/equity/orders/market",
                             json={"ticker": ticker, "quantity": quantity, "extendedHours": extended_hours})

    def cancel(self, order_id: int) -> None:
        self._request("DELETE", f"/equity/orders/{int(order_id)}")


class ReadOnlyClient(Trading212Client):
    """A rehearsal cannot trade even if a caller accidentally tries to."""

    def _request(self, method, path, *, params=None, json=None):
        if method.upper() in ("POST", "DELETE"):
            raise RuntimeError("read-only client refuses POST and DELETE")
        return super()._request(method, path, params=params, json=json)
