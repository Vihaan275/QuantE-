import os
import re
import time
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"

# Map tickers to full company names so TTS says "Nvidia" not "N-V-D-A"
TICKER_TO_NAME = {
    "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "GOOG": "Google", "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla",
    "NFLX": "Netflix", "AMD": "AMD", "INTC": "Intel", "CRM": "Salesforce",
    "ORCL": "Oracle", "ADBE": "Adobe", "CSCO": "Cisco", "QCOM": "Qualcomm",
    "TXN": "Texas Instruments", "AVGO": "Broadcom", "IBM": "IBM", "UBER": "Uber",
    "LYFT": "Lyft", "SNAP": "Snap", "PINS": "Pinterest", "SQ": "Block",
    "PYPL": "PayPal", "SHOP": "Shopify", "SPOT": "Spotify", "DIS": "Disney",
    "WMT": "Walmart", "TGT": "Target", "COST": "Costco", "HD": "Home Depot",
    "JPM": "JP Morgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard", "BA": "Boeing",
    "CAT": "Caterpillar", "JNJ": "Johnson and Johnson", "PFE": "Pfizer",
    "UNH": "United Health", "XOM": "Exxon Mobil", "CVX": "Chevron",
    "KO": "Coca Cola", "PEP": "Pepsi", "MCD": "McDonalds", "NKE": "Nike",
    "SBUX": "Starbucks", "F": "Ford", "GM": "General Motors", "LLY": "Eli Lilly",
    "ABBV": "AbbVie", "MRK": "Merck", "TMO": "Thermo Fisher", "NOW": "ServiceNow",
    "PANW": "Palo Alto Networks", "SNOW": "Snowflake", "PLTR": "Palantir",
    "COIN": "Coinbase", "ROKU": "Roku", "ZM": "Zoom", "DDOG": "Datadog",
    "NET": "Cloudflare", "CRWD": "CrowdStrike", "MU": "Micron",
    "MRVL": "Marvell", "ARM": "ARM Holdings", "SMCI": "Super Micro",
}


class TTSEngine:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id: str | None = None
        self.voice_name: str = ""
        self._client = httpx.Client(timeout=30)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def resolve_voice(self):
        """Fetch voice list and pick a conversational, natural-sounding voice."""
        if self.voice_id:
            return
        if not self.is_configured:
            logger.warning("ELEVENLABS_API_KEY not set — TTS disabled")
            return
        try:
            r = self._client.get(
                f"{ELEVENLABS_BASE}/voices",
                headers={"xi-api-key": self.api_key},
            )
            r.raise_for_status()
            voices = r.json().get("voices", [])
            # Deep, dark, bass business voice
            for preferred in ["Brian", "Charlie", "George", "Daniel", "Adam"]:
                for v in voices:
                    if v["name"].split(" - ")[0].strip() == preferred:
                        self.voice_id = v["voice_id"]
                        self.voice_name = v["name"]
                        logger.info(f"TTS voice resolved: {self.voice_name} ({self.voice_id})")
                        return
            # Last resort: first available voice
            if voices:
                self.voice_id = voices[0]["voice_id"]
                self.voice_name = voices[0]["name"]
                logger.info(f"TTS fallback voice: {self.voice_name} ({self.voice_id})")
        except Exception as e:
            logger.error(f"Failed to resolve TTS voice: {e}")

    def build_summary_script(self, signal_result: dict) -> str:
        """Build a natural, conversational spoken script. Sounds like a friend briefing you."""
        raw_ticker = signal_result.get("ticker", "the stock")
        ticker = TICKER_TO_NAME.get(raw_ticker.upper(), raw_ticker) if raw_ticker else "the stock"
        label = signal_result.get("signal_label", "NEUTRAL")
        composite = signal_result.get("composite_signal", 0)
        confidence = signal_result.get("confidence", 0)
        conf_pct = int(confidence * 100)

        sub_scores = signal_result.get("sub_scores", {})

        # Sentence 1: conversational opener
        if label == "LONG":
            s1 = f"Alright, so {ticker} is looking pretty bullish here — {conf_pct} percent confidence on this one."
        elif label == "SHORT":
            s1 = f"Okay so {ticker} — this one's coming in bearish, {conf_pct} percent confidence."
        else:
            s1 = f"So {ticker} is reading neutral right now, sitting at {conf_pct} percent confidence."

        # Sentence 2: the why — conversational
        factor_order = ["language_momentum", "guidance_hedge", "analyst_pressure"]
        s2 = ""
        for factor in factor_order:
            fd = sub_scores.get(factor, {})
            if not fd:
                continue
            score = fd.get("score", 0)
            if abs(score) < 0.1:
                continue
            if factor == "language_momentum":
                if score > 0:
                    s2 = "The language throughout the call is really leaning positive — lots of growth-oriented vocabulary."
                else:
                    s2 = "The tone of the call is pretty defensive honestly — cautious language all over."
            elif factor == "guidance_hedge":
                if score > 0:
                    s2 = "Management sounds genuinely confident in their forward guidance, not a lot of hedging."
                else:
                    s2 = "There's a ton of hedging in the guidance — they're being really careful with their words."
            elif factor == "analyst_pressure":
                if score > 0:
                    s2 = "The Q and A went well — management gave straight answers with real numbers."
                else:
                    s2 = "Analysts were pressing hard and management kept dodging — not a great look."
            break
        if not s2:
            s2 = "The signals are honestly pretty mixed across the board."

        # Sentence 3: standout quote — casual delivery
        s3 = ""
        for factor in factor_order:
            fd = sub_scores.get(factor, {})
            top = fd.get("top_sentences", [])
            if top:
                text = top[0].get("text", "")
                if len(text.split()) > 20:
                    text = text.split(",")[0].split(";")[0]
                    text = text.strip().rstrip(".")
                s3 = f'One thing that stood out — they said "{text}."'
                break
        if not s3:
            s3 = "Nothing really jumped out quote-wise from the transcript."

        # Sentence 4: what it means
        if label == "LONG":
            s4 = "Overall this is looking like a solid long setup."
        elif label == "SHORT":
            s4 = "I'd be cautious here — the data points toward the downside."
        else:
            s4 = "Probably best to sit this one out and wait for a clearer signal."

        script = f"{s1} {s2} {s3} {s4}"

        # Clean up: replace $ with "dollars", % with "percent"
        script = script.replace("$", "").replace("%", " percent")
        script = re.sub(r"\s+", " ", script).strip()

        return script

    def synthesize(self, text: str) -> bytes | None:
        """Call ElevenLabs TTS API. Returns MP3 bytes or None on failure."""
        if not self.voice_id or not self.api_key:
            logger.warning("TTS not configured — skipping synthesis")
            return None

        try:
            r = self._client.post(
                f"{ELEVENLABS_BASE}/text-to-speech/{self.voice_id}",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.6,
                        "similarity_boost": 0.85,
                        "style": 0.45,
                        "use_speaker_boost": True,
                    },
                },
            )
            r.raise_for_status()
            return r.content
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    def generate_and_cache(self, signal_result: dict, ticker: str, quarter: str) -> tuple[str | None, str | None]:
        """Generate TTS audio, cache it, return (relative_path, script) or (None, None)."""
        script = self.build_summary_script(signal_result)
        char_count = len(script)

        logger.info(f"TTS generating | ticker={ticker} quarter={quarter} chars={char_count}")

        audio_data = self.synthesize(script)
        if not audio_data:
            logger.warning(f"TTS failed | ticker={ticker} quarter={quarter}")
            return None, None

        ts = int(time.time())
        filename = f"tts_{ticker}_{quarter}_{ts}.mp3"
        filepath = CACHE_DIR / filename
        filepath.write_bytes(audio_data)

        logger.info(f"TTS success | ticker={ticker} quarter={quarter} chars={char_count} file={filename}")
        return filename, script
