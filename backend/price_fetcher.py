import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def fetch_price_data(ticker: str, earnings_date: str) -> dict | None:
    """Fetch price data around an earnings date. Returns None on failure."""
    cache_file = CACHE_DIR / f"prices_{ticker}_{earnings_date}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    try:
        dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        start = dt - timedelta(days=10)
        end = dt + timedelta(days=35)

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            logger.warning(f"No price data for {ticker} around {earnings_date}")
            return None

        # Find the earnings day (or nearest trading day)
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        earnings_dt = dt

        # Find nearest trading day on or after earnings date
        trading_days = hist.index.tolist()
        earnings_day_idx = None
        for i, td in enumerate(trading_days):
            if td.date() >= earnings_dt.date():
                earnings_day_idx = i
                break

        if earnings_day_idx is None:
            logger.warning(f"Could not find trading day for {ticker} on/after {earnings_date}")
            return None

        base_price = float(hist.iloc[earnings_day_idx]["Open"])

        def get_price_at_offset(offset: int) -> dict | None:
            idx = earnings_day_idx + offset
            if 0 <= idx < len(hist):
                close = float(hist.iloc[idx]["Close"])
                pct = round(((close - base_price) / base_price) * 100, 2)
                return {"close": round(close, 2), "pct_change": pct}
            return None

        # Build daily series for chart
        daily_series = []
        for i in range(len(trading_days)):
            rel_day = i - earnings_day_idx
            if -5 <= rel_day <= 25:
                daily_series.append({
                    "day": rel_day,
                    "date": trading_days[i].strftime("%Y-%m-%d"),
                    "close": round(float(hist.iloc[i]["Close"]), 2),
                    "open": round(float(hist.iloc[i]["Open"]), 2),
                })

        result = {
            "ticker": ticker,
            "earnings_date": earnings_date,
            "base_price": round(base_price, 2),
            "day_1": get_price_at_offset(1),
            "day_5": get_price_at_offset(5),
            "day_20": get_price_at_offset(20),
            "daily_series": daily_series,
        }

        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {e}")
        return None


def fetch_multi_horizon_returns(ticker: str, earnings_date: str) -> dict:
    """Fetch returns at multiple horizons (1D-20D) for signal decay analysis."""
    horizons = [1, 2, 3, 5, 7, 10, 15, 20]
    # Try to use cached price data first
    cache_file = CACHE_DIR / f"prices_{ticker}_{earnings_date}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cached = json.load(f)
        daily_series = cached.get("daily_series", [])
        base_price = cached.get("base_price")
        if daily_series and base_price:
            result = {}
            for h in horizons:
                match = [d for d in daily_series if d["day"] == h]
                if match:
                    pct = round(((match[0]["close"] - base_price) / base_price) * 100, 2)
                    result[f"{h}d"] = pct
                else:
                    result[f"{h}d"] = None
            return result

    # Fetch fresh if no cache
    price_data = fetch_price_data(ticker, earnings_date)
    if not price_data:
        return {f"{h}d": None for h in horizons}

    base_price = price_data.get("base_price")
    daily_series = price_data.get("daily_series", [])
    result = {}
    for h in horizons:
        match = [d for d in daily_series if d["day"] == h]
        if match and base_price:
            pct = round(((match[0]["close"] - base_price) / base_price) * 100, 2)
            result[f"{h}d"] = pct
        else:
            result[f"{h}d"] = None
    return result


def fetch_replay_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> dict | None:
    """Fetch OHLCV candle data for replay simulation."""
    cache_key = f"replay_{ticker}_{start_date}_{end_date}_{interval}".replace("-", "")
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    try:
        stock = yf.Ticker(ticker)

        # yfinance only supports intraday data for recent dates (~60 days)
        yf_interval = interval if interval in ("1d", "1h") else "1d"
        hist = stock.history(start=start_date, end=end_date, interval=yf_interval)

        # If hourly fails or returns empty, fall back to daily
        if hist.empty and yf_interval == "1h":
            logger.info(f"Hourly data unavailable for {ticker}, falling back to daily")
            hist = stock.history(start=start_date, end=end_date, interval="1d")
            yf_interval = "1d"

        if hist.empty:
            return None

        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        candles = []
        for ts, row in hist.iterrows():
            candle = {
                "time": ts.strftime("%Y-%m-%d") if yf_interval == "1d" else ts.isoformat(),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
            candles.append(candle)

        # Cap at 200 candles
        candles = candles[:200]

        result = {
            "ticker": ticker,
            "interval": yf_interval,
            "start_date": start_date,
            "end_date": end_date,
            "candles": candles,
        }

        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        logger.error(f"Error fetching replay data for {ticker}: {e}")
        return None
