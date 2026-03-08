import os
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.alphavantage.co/query"

# Quarter month ranges (calendar quarters)
QUARTER_MONTHS = {
    1: (1, 3),
    2: (4, 6),
    3: (7, 9),
    4: (10, 12),
}


def _parse_quarter_string(quarter: str) -> tuple[int, int]:
    """Parse '2024Q2' into (2024, 2)."""
    quarter = quarter.upper().strip()
    year = int(quarter[:4])
    q = int(quarter[-1])
    if q < 1 or q > 4:
        raise ValueError(f"Invalid quarter number in '{quarter}': must be 1-4")
    return year, q


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"earnings_{ticker.upper()}.json"


def _fetch_raw_earnings(ticker: str) -> dict | None:
    """Fetch full earnings data from Alpha Vantage, using cache if available."""
    cache_file = _cache_path(ticker)
    if cache_file.exists():
        logger.info(f"Loading earnings for {ticker} from cache")
        with open(cache_file, "r") as f:
            return json.load(f)

    api_key = os.getenv("AV_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        logger.error("AV_API_KEY not set or is placeholder. Cannot fetch earnings.")
        return None

    logger.info(f"Fetching earnings for {ticker} from Alpha Vantage")
    params = {
        "function": "EARNINGS",
        "symbol": ticker,
        "apikey": api_key,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"Request failed for {ticker} earnings: {e}")
        return None

    if "Error Message" in data:
        logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
        return None
    if "Note" in data:
        logger.error(f"API rate limit hit: {data['Note']}")
        return None
    if "Information" in data:
        logger.error(f"API info: {data['Information']}")
        return None

    if "quarterlyEarnings" not in data:
        logger.error(f"No quarterlyEarnings in response for {ticker}")
        return None

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

    # Rate limit after non-cached API call
    time.sleep(1.2)

    return data


def _match_quarter(quarterly_earnings: list[dict], year: int, q: int) -> dict | None:
    """Find the quarterly earnings entry matching the given year and quarter.

    Tries calendar-quarter matching first (fiscalDateEnding month falls in the
    expected range).  If nothing matches, tries shifting by +/-1 quarter to
    accommodate companies with non-calendar fiscal years (e.g. AAPL fiscal Q4
    ends in September).
    """
    month_lo, month_hi = QUARTER_MONTHS[q]

    # First pass: exact calendar-quarter match on fiscalDateEnding
    for entry in quarterly_earnings:
        try:
            fd = datetime.strptime(entry["fiscalDateEnding"], "%Y-%m-%d")
        except (ValueError, KeyError):
            continue
        if fd.year == year and month_lo <= fd.month <= month_hi:
            return entry

    # Second pass: try adjacent quarters (Q+1 and Q-1) for non-calendar FY
    for offset in (1, -1):
        adj_q = q + offset
        adj_year = year
        if adj_q < 1:
            adj_q = 4
            adj_year -= 1
        elif adj_q > 4:
            adj_q = 1
            adj_year += 1
        adj_lo, adj_hi = QUARTER_MONTHS[adj_q]
        for entry in quarterly_earnings:
            try:
                fd = datetime.strptime(entry["fiscalDateEnding"], "%Y-%m-%d")
            except (ValueError, KeyError):
                continue
            if fd.year == adj_year and adj_lo <= fd.month <= adj_hi:
                return entry

    return None


def fetch_earnings_surprise(ticker: str, quarter: str) -> dict | None:
    """Fetch EPS surprise data for a given ticker and quarter.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        quarter: Quarter string in format "2024Q2".

    Returns:
        Dict with reported_eps, estimated_eps, surprise, surprise_pct,
        beat, and quarter_matched.  Returns None on error or if no
        matching quarter is found.
    """
    year, q = _parse_quarter_string(quarter)

    data = _fetch_raw_earnings(ticker)
    if data is None:
        return None

    quarterly = data.get("quarterlyEarnings", [])
    if not quarterly:
        logger.error(f"No quarterly earnings entries for {ticker}")
        return None

    entry = _match_quarter(quarterly, year, q)
    if entry is None:
        logger.error(f"No matching quarter found for {ticker} {quarter}")
        return None

    try:
        reported_eps = float(entry.get("reportedEPS", 0) or 0)
        estimated_eps = float(entry.get("estimatedEPS", 0) or 0)
        surprise = float(entry.get("surprise", 0) or 0)
        surprise_pct = float(entry.get("surprisePercentage", 0) or 0)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to parse EPS values for {ticker} {quarter}: {e}")
        return None

    return {
        "reported_eps": reported_eps,
        "estimated_eps": estimated_eps,
        "surprise": surprise,
        "surprise_pct": surprise_pct,
        "beat": reported_eps > estimated_eps,
        "quarter_matched": entry.get("fiscalDateEnding", ""),
        "reported_date": entry.get("reportedDate", ""),
    }
