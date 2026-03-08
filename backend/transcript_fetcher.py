import os
import json
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


class TranscriptFetcher:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.api_key = os.getenv("AV_API_KEY", "")
        if not self.api_key or self.api_key == "your_key_here":
            logger.warning("No AV_API_KEY set. Transcript fetching will fail.")
            self.api_key = ""

    def _cache_path(self, ticker: str, quarter: str) -> Path:
        return CACHE_DIR / f"{ticker}_{quarter}.json"

    def fetch_transcript(self, ticker: str, quarter: str) -> dict:
        cache_file = self._cache_path(ticker, quarter)
        if cache_file.exists():
            logger.info(f"Loading {ticker} {quarter} from cache")
            with open(cache_file, "r") as f:
                return json.load(f)

        logger.info(f"Fetching {ticker} {quarter} from Alpha Vantage")
        params = {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": ticker,
            "quarter": quarter,
            "apikey": self.api_key,
        }

        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error for {ticker} {quarter}: {data['Error Message']}")
        if "Note" in data or "Information" in data:
            raise ValueError(f"API rate limited: {data.get('Note') or data.get('Information')}")

        # Extract transcript text from response
        transcript_text = ""
        if "transcript" in data:
            for entry in data["transcript"]:
                speaker = entry.get("speaker", "Unknown")
                title = entry.get("title", "")
                text = entry.get("content", "") or entry.get("speech", "")
                if title and title.lower() not in speaker.lower():
                    transcript_text += f"{speaker} ({title}): {text}\n\n"
                else:
                    transcript_text += f"{speaker}: {text}\n\n"

        result = {
            "ticker": ticker,
            "quarter": quarter,
            "raw_response": data,
            "transcript_text": transcript_text,
        }

        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def fetch_multiple(self, items: list[tuple[str, str]]) -> list[dict]:
        results = []
        for ticker, quarter in items:
            was_cached = self._cache_path(ticker, quarter).exists()
            try:
                result = self.fetch_transcript(ticker, quarter)
                results.append(result)
                # Rate limit: 1 request per second for non-cached fetches
                if not was_cached:
                    time.sleep(1.2)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker} {quarter}: {e}")
                results.append({
                    "ticker": ticker,
                    "quarter": quarter,
                    "transcript_text": "",
                    "error": str(e),
                })
        return results
