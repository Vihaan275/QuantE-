import os
import math
import logging
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PAPER_BASE = "https://paper-api.alpaca.markets"


class AlpacaTrader:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    @property
    def headers(self):
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    @property
    def is_configured(self):
        return bool(self.api_key and self.secret_key
                     and self.api_key != "your_alpaca_key"
                     and self.secret_key != "your_alpaca_secret")

    def _get(self, path: str) -> dict:
        r = requests.get(f"{PAPER_BASE}{path}", headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{PAPER_BASE}{path}", headers=self.headers, json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict | None:
        r = requests.delete(f"{PAPER_BASE}{path}", headers=self.headers, timeout=10)
        r.raise_for_status()
        if r.text:
            return r.json()
        return None

    # --- Account ---

    def get_account(self) -> dict:
        acc = self._get("/v2/account")
        return {
            "account_id": acc.get("id", ""),
            "status": acc.get("status", ""),
            "equity": float(acc.get("equity", 0)),
            "cash": float(acc.get("cash", 0)),
            "buying_power": float(acc.get("buying_power", 0)),
            "portfolio_value": float(acc.get("portfolio_value", 0)),
            "last_equity": float(acc.get("last_equity", 0)),
            "day_pnl": round(float(acc.get("equity", 0)) - float(acc.get("last_equity", 0)), 2),
            "day_pnl_pct": round(
                ((float(acc.get("equity", 0)) - float(acc.get("last_equity", 0)))
                 / max(float(acc.get("last_equity", 1)), 1)) * 100, 2
            ),
            "shorting_enabled": acc.get("shorting_enabled", False),
            "trading_blocked": acc.get("trading_blocked", False),
        }

    # --- Positions ---

    def get_positions(self) -> list[dict]:
        positions = self._get("/v2/positions")
        result = []
        for p in positions:
            result.append({
                "symbol": p.get("symbol", ""),
                "qty": float(p.get("qty", 0)),
                "side": p.get("side", "long"),
                "avg_entry": float(p.get("avg_entry_price", 0)),
                "current_price": float(p.get("current_price", 0)),
                "market_value": float(p.get("market_value", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
                "unrealized_plpc": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
            })
        return result

    def close_position(self, symbol: str) -> dict:
        try:
            self._delete(f"/v2/positions/{symbol}")
            return {"success": True, "message": f"Closed position in {symbol}"}
        except requests.HTTPError as e:
            return {"success": False, "message": f"Failed to close {symbol}: {e.response.text}"}

    # --- Orders ---

    def get_orders(self, limit: int = 10) -> list[dict]:
        orders = self._get(f"/v2/orders?limit={limit}&status=all")
        result = []
        for o in orders:
            result.append({
                "id": o.get("id", ""),
                "symbol": o.get("symbol", ""),
                "side": o.get("side", ""),
                "qty": o.get("qty", ""),
                "type": o.get("type", ""),
                "status": o.get("status", ""),
                "filled_avg_price": o.get("filled_avg_price"),
                "filled_qty": o.get("filled_qty"),
                "submitted_at": o.get("submitted_at", ""),
                "filled_at": o.get("filled_at"),
            })
        return result

    # --- Quote ---

    def get_quote(self, symbol: str) -> dict | None:
        """Get latest quote via Alpaca data API."""
        try:
            url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
            r = requests.get(url, headers=self.headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            q = data.get("quote", {})
            return {
                "symbol": symbol,
                "ask": float(q.get("ap", 0)),
                "bid": float(q.get("bp", 0)),
                "mid": round((float(q.get("ap", 0)) + float(q.get("bp", 0))) / 2, 2),
            }
        except Exception as e:
            logger.warning(f"Could not get quote for {symbol}: {e}")
            return None

    # --- Execute Signal Trade ---

    def execute_signal_trade(
        self,
        ticker: str,
        signal_label: str,
        composite_signal: float,
        confidence: float,
        base_notional: float = 2000.0,
    ) -> dict:
        """
        Execute a paper trade based on the analysis signal.

        - LONG  -> market buy
        - SHORT -> market sell (short)
        - NEUTRAL -> no trade

        Position size = base_notional * confidence, converted to whole shares
        using the current mid price.
        """
        if signal_label == "NEUTRAL":
            return {
                "executed": False,
                "reason": "Signal is NEUTRAL -- no trade executed.",
            }

        if not ticker:
            return {"executed": False, "reason": "No ticker provided."}

        # Get current price
        quote = self.get_quote(ticker)
        if not quote or quote["mid"] <= 0:
            return {"executed": False, "reason": f"Could not get price for {ticker}."}

        price = quote["mid"]
        notional = base_notional * confidence
        qty = max(1, math.floor(notional / price))
        side = "buy" if signal_label == "LONG" else "sell"

        order_body = {
            "symbol": ticker,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }

        try:
            order = self._post("/v2/orders", order_body)
            return {
                "executed": True,
                "order_id": order.get("id", ""),
                "symbol": ticker,
                "side": side,
                "qty": qty,
                "type": "market",
                "status": order.get("status", ""),
                "signal_label": signal_label,
                "composite_signal": composite_signal,
                "confidence": confidence,
                "estimated_price": price,
                "estimated_notional": round(qty * price, 2),
                "submitted_at": order.get("submitted_at", ""),
            }
        except requests.HTTPError as e:
            error_msg = e.response.text if e.response else str(e)
            return {"executed": False, "reason": f"Order rejected: {error_msg}"}
