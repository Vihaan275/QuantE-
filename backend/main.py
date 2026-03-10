import json
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from transcript_fetcher import TranscriptFetcher
from transcript_parser import parse_transcript
from signal_engine import compute_signal, compute_naive_signal, compute_guidance_hedge, compute_analyst_pressure, compute_language_momentum, compute_earnings_surprise_score, compute_bootstrap_and_attributions, _sanitize
from price_fetcher import fetch_price_data, fetch_replay_data, fetch_multi_horizon_returns
from earnings_fetcher import fetch_earnings_surprise
from backtester import run_backtest
from trader import AlpacaTrader
from tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent
SAMPLES_FILE = BACKEND_DIR / "samples_metadata.json"

PRELOAD_SAMPLES = [
    {"ticker": "NVDA", "quarter": "2024Q2", "known_outcome_5d_pct": 25.6,
     "description": "Blowout quarter -- data center demand explosion, Blackwell ramp"},
    {"ticker": "META", "quarter": "2022Q3", "known_outcome_5d_pct": -19.1,
     "description": "Metaverse capex spiral -- Zuckerberg doubled down, market revolted"},
    {"ticker": "NFLX", "quarter": "2022Q1", "known_outcome_5d_pct": -35.1,
     "description": "First subscriber loss in a decade -- hedging language throughout"},
    {"ticker": "AAPL", "quarter": "2023Q4", "known_outcome_5d_pct": 2.1,
     "description": "Steady beat -- services growth offsetting hardware softness"},
    {"ticker": "MSFT", "quarter": "2024Q1", "known_outcome_5d_pct": 8.9,
     "description": "Azure AI inflection -- Copilot narrative lands with analysts"},
    {"ticker": "TSLA", "quarter": "2023Q3", "known_outcome_5d_pct": -10.2,
     "description": "Margin compression -- price cuts eating into profitability"},
]


def _generate_samples_metadata():
    samples = []
    for i, s in enumerate(PRELOAD_SAMPLES, 1):
        samples.append({
            "id": i,
            "ticker": s["ticker"],
            "quarter": s["quarter"],
            "known_outcome_5d_pct": s["known_outcome_5d_pct"],
            "description": s["description"],
        })
    data = {"samples": samples}
    with open(SAMPLES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Generated samples_metadata.json")
    return data


tts_engine = TTSEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: preloading transcripts...")
    fetcher = TranscriptFetcher()
    items = [(s["ticker"], s["quarter"]) for s in PRELOAD_SAMPLES]
    results = fetcher.fetch_multiple(items)
    for r in results:
        if r.get("error"):
            logger.warning(f"Failed to preload {r['ticker']} {r['quarter']}: {r['error']}")
        else:
            logger.info(f"Preloaded {r['ticker']} {r['quarter']}")
    _generate_samples_metadata()
    # Resolve TTS voice on startup
    tts_engine.resolve_voice()
    yield


app = FastAPI(title="Earnings Alpha Scanner", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---

class AnalyzeRequest(BaseModel):
    transcript: str
    ticker: str = ""
    quarter: str = ""
    earnings_date: str = ""


class FetchTranscriptRequest(BaseModel):
    ticker: str
    quarter: str


class BacktestRequest(BaseModel):
    sample_ids: list[int]


class TradeRequest(BaseModel):
    ticker: str
    signal_label: str
    composite_signal: float
    confidence: float
    base_notional: float = 2000.0


class ClosePositionRequest(BaseModel):
    symbol: str


class TTSRequest(BaseModel):
    signal_result: dict
    ticker: str = ""
    quarter: str = ""


# --- Helpers ---

def _get_cached_texts(exclude_ticker: str = "") -> list[str]:
    cache_dir = BACKEND_DIR / "cache"
    texts = []
    for f in cache_dir.glob("*.json"):
        if f.name.startswith("prices_"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            t = data.get("transcript_text", "")
            if t and (not exclude_ticker or not f.name.startswith(exclude_ticker)):
                texts.append(t)
        except Exception:
            pass
    return texts


def _load_samples() -> dict:
    if SAMPLES_FILE.exists():
        with open(SAMPLES_FILE) as f:
            return json.load(f)
    return _generate_samples_metadata()


# --- Endpoints ---

@app.get("/health")
def health():
    cache_dir = BACKEND_DIR / "cache"
    cached = []
    for f in cache_dir.glob("*.json"):
        if not f.name.startswith("prices_"):
            cached.append(f.stem)
    return {"status": "ok", "cached_transcripts": cached}


@app.get("/samples")
def get_samples():
    return _load_samples()


@app.get("/sample/{sample_id}")
def get_sample(sample_id: int):
    data = _load_samples()
    sample = None
    for s in data["samples"]:
        if s["id"] == sample_id:
            sample = s
            break
    if not sample:
        raise HTTPException(404, "Sample not found")

    cache_file = BACKEND_DIR / "cache" / f"{sample['ticker']}_{sample['quarter']}.json"
    transcript_text = ""
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        transcript_text = cached.get("transcript_text", "")

    return {
        "ticker": sample["ticker"],
        "quarter": sample["quarter"],
        "transcript_text": transcript_text,
        "metadata": sample,
    }


@app.post("/fetch-transcript")
def fetch_transcript_endpoint(req: FetchTranscriptRequest):
    fetcher = TranscriptFetcher()
    try:
        result = fetcher.fetch_transcript(req.ticker.upper(), req.quarter.upper())
        return {
            "ticker": req.ticker.upper(),
            "quarter": req.quarter.upper(),
            "transcript_text": result.get("transcript_text", ""),
            "cached": True,
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    async def event_generator():
        try:
            # Parse transcript
            parsed = parse_transcript(req.transcript)
            yield {
                "event": "parsing_complete",
                "data": json.dumps({
                    "sentence_count": parsed["stats"]["total_sentences"],
                    "section_counts": {
                        "prepared": parsed["stats"]["prepared_count"],
                        "qa": parsed["stats"]["qa_count"],
                    },
                }),
            }
            await asyncio.sleep(0.1)

            all_sentences = parsed["all_sentences"]
            cached_texts = _get_cached_texts(req.ticker)

            # Sub-score 1: Guidance Hedge
            guidance = compute_guidance_hedge(all_sentences)
            yield {
                "event": "subscore",
                "data": json.dumps(_sanitize({"name": "guidance_hedge", **guidance})),
            }
            await asyncio.sleep(0.1)

            # Sub-score 2: Analyst Pressure
            pressure = compute_analyst_pressure(parsed)
            yield {
                "event": "subscore",
                "data": json.dumps(_sanitize({"name": "analyst_pressure", **pressure})),
            }
            await asyncio.sleep(0.1)

            # Sub-score 3: Language Momentum
            momentum = compute_language_momentum(parsed, cached_texts)
            yield {
                "event": "subscore",
                "data": json.dumps(_sanitize({"name": "language_momentum", **momentum})),
            }
            await asyncio.sleep(0.1)

            # Sub-score 5: Earnings Surprise (EPS vs estimates)
            earnings_surprise_data = None
            if req.ticker and req.quarter:
                try:
                    earnings_surprise_data = fetch_earnings_surprise(req.ticker.upper(), req.quarter.upper())
                except Exception as e:
                    logger.warning(f"Failed to fetch earnings surprise: {e}")

            earnings = compute_earnings_surprise_score(earnings_surprise_data)
            if earnings["available"]:
                yield {
                    "event": "subscore",
                    "data": json.dumps(_sanitize({"name": "earnings_surprise", **earnings})),
                }
                await asyncio.sleep(0.1)

            # Fixed 3-factor composite (EPS is confidence modifier, not weight)
            composite = (
                0.35 * guidance["score"]
                + 0.15 * pressure["score"]
                + 0.50 * momentum["score"]
            )
            composite = round(max(-1.0, min(1.0, composite)), 2)

            if composite >= 0.20:
                label = "LONG"
            elif composite <= -0.20:
                label = "SHORT"
            else:
                label = "NEUTRAL"

            # Confidence: blend of signal strength, factor agreement, data coverage
            strength = min(abs(composite) / 0.6, 1.0)
            scores = [guidance["score"], pressure["score"], momentum["score"]]
            sign = 1 if composite >= 0 else -1
            agreeing = sum(1 for s in scores if s * sign > 0)
            agreement = agreeing / len(scores)
            analyzed = [
                guidance.get("sentences_analyzed", 0),
                pressure.get("sentences_analyzed", 0),
                momentum.get("sentences_analyzed", 0),
            ]
            avg_analyzed = sum(analyzed) / len(analyzed) if analyzed else 0
            coverage = min(avg_analyzed / 300, 1.0)
            raw_confidence = 0.40 * strength + 0.35 * agreement + 0.25 * coverage
            confidence = round(max(0.10, min(0.85, raw_confidence)), 2)

            # EPS surprise modifies confidence, not composite
            eps_modifier = 0.0
            if earnings["available"]:
                eps_score = earnings["score"]
                if (composite > 0 and eps_score > 0) or (composite < 0 and eps_score < 0):
                    eps_modifier = min(0.10, abs(eps_score) * 0.10)
                elif (composite > 0 and eps_score < -0.2) or (composite < 0 and eps_score > 0.2):
                    eps_modifier = -min(0.15, abs(eps_score) * 0.15)
                confidence = round(max(0.10, min(0.85, confidence + eps_modifier)), 2)

            # Calibration context (no hardcoded accuracy — only live backtest data is trustworthy)
            abs_comp = abs(composite)
            if abs_comp >= 0.5:
                cal = {"bucket": "strong"}
            elif abs_comp >= 0.3:
                cal = {"bucket": "moderate"}
            elif abs_comp >= 0.2:
                cal = {"bucket": "weak"}
            else:
                cal = {"bucket": "neutral"}

            # Build summary
            parts = []
            if earnings["available"]:
                if earnings["score"] > 0.2: parts.append(f"EPS beat ({earnings.get('surprise_pct', 0):+.1f}%)")
                elif earnings["score"] < -0.2: parts.append(f"EPS miss ({earnings.get('surprise_pct', 0):+.1f}%)")
            if guidance["score"] > 0.2: parts.append("confident guidance")
            elif guidance["score"] < -0.2: parts.append("hedging in guidance")
            if pressure["score"] > 0.1: parts.append("direct analyst responses")
            elif pressure["score"] < -0.1: parts.append("evasive Q&A")
            if momentum["score"] > 0.2: parts.append("bullish momentum")
            elif momentum["score"] < -0.2: parts.append("bearish momentum")
            detail = " and ".join(parts) if parts else "mixed signals"

            if label == "LONG":
                summary = f"Strong bullish signal driven by {detail}."
            elif label == "SHORT":
                summary = f"Bearish signal driven by {detail}."
            else:
                summary = f"Neutral signal with {detail}."

            # Bootstrap CIs and attributions (computed from already-available sub-scores, no double inference)
            bootstrap_cis, sentence_attributions = compute_bootstrap_and_attributions(
                guidance, momentum, all_sentences
            )

            yield {
                "event": "composite",
                "data": json.dumps(_sanitize({
                    "composite_signal": composite,
                    "signal_label": label,
                    "confidence": confidence,
                    "one_line_summary": summary,
                    "calibration": cal,
                    "eps_modifier": round(eps_modifier, 3) if earnings["available"] else None,
                    "bootstrap_cis": bootstrap_cis,
                })),
            }
            await asyncio.sleep(0.1)

            # Sentence attributions for heatmap
            if sentence_attributions:
                yield {
                    "event": "attributions",
                    "data": json.dumps(_sanitize(sentence_attributions)),
                }
                await asyncio.sleep(0.1)

            # Price data — use user-supplied date, or fall back to reported_date from EPS data
            effective_date = req.earnings_date
            if not effective_date and earnings_surprise_data:
                effective_date = earnings_surprise_data.get("reported_date", "")
            if req.ticker and effective_date:
                price_data = fetch_price_data(req.ticker.upper(), effective_date)
                if price_data:
                    yield {
                        "event": "price_data",
                        "data": json.dumps(_sanitize(price_data)),
                    }

            yield {"event": "done", "data": "{}"}

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


@app.post("/backtest")
def backtest(req: BacktestRequest):
    samples_data = _load_samples()
    sample_map = {s["id"]: s for s in samples_data["samples"]}

    results = []
    for sid in req.sample_ids:
        sample = sample_map.get(sid)
        if not sample:
            continue

        cache_file = BACKEND_DIR / "cache" / f"{sample['ticker']}_{sample['quarter']}.json"
        if not cache_file.exists():
            continue

        with open(cache_file) as f:
            cached = json.load(f)
        transcript_text = cached.get("transcript_text", "")
        if not transcript_text:
            continue

        parsed = parse_transcript(transcript_text)
        cached_texts = _get_cached_texts(sample["ticker"])
        try:
            surprise_data = fetch_earnings_surprise(sample["ticker"], sample["quarter"])
        except Exception:
            surprise_data = None
        signal_result = compute_signal(parsed, cached_texts, surprise_data)

        # Naive baseline signal (keyword-only, no embeddings)
        naive_result = compute_naive_signal(parsed)

        # Multi-horizon returns for signal decay analysis
        reported_date = ""
        if surprise_data and surprise_data.get("reported_date"):
            reported_date = surprise_data["reported_date"]
        decay_returns = {}
        if reported_date:
            try:
                decay_returns = fetch_multi_horizon_returns(sample["ticker"], reported_date)
            except Exception:
                pass

        results.append({
            "ticker": sample["ticker"],
            "quarter": sample["quarter"],
            "signal": signal_result["composite_signal"],
            "actual_5d_pct": sample["known_outcome_5d_pct"],
            "signal_label": signal_result["signal_label"],
            "naive_signal": naive_result["composite_signal"],
            "decay_returns": decay_returns,
        })

    backtest_stats = run_backtest(results)
    return backtest_stats


# --- Trading Endpoints ---

@app.get("/trading/status")
def trading_status():
    trader = AlpacaTrader()
    if not trader.is_configured:
        return {"connected": False, "reason": "Alpaca API keys not configured in .env"}
    try:
        acc = trader.get_account()
        return {"connected": True, "account": acc}
    except Exception as e:
        return {"connected": False, "reason": str(e)}


@app.get("/trading/account")
def trading_account():
    trader = AlpacaTrader()
    if not trader.is_configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return trader.get_account()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/trading/positions")
def trading_positions():
    trader = AlpacaTrader()
    if not trader.is_configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return trader.get_positions()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/trading/orders")
def trading_orders():
    trader = AlpacaTrader()
    if not trader.is_configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return trader.get_orders(limit=10)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/trading/quote/{symbol}")
def trading_quote(symbol: str):
    trader = AlpacaTrader()
    if not trader.is_configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        quote = trader.get_quote(symbol.upper())
        if not quote:
            raise HTTPException(404, f"No quote for {symbol}")
        return quote
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/trading/execute")
def trading_execute(req: TradeRequest):
    trader = AlpacaTrader()
    if not trader.is_configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        result = trader.execute_signal_trade(
            ticker=req.ticker.upper(),
            signal_label=req.signal_label,
            composite_signal=req.composite_signal,
            confidence=req.confidence,
            base_notional=req.base_notional,
        )
        return result
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/trading/close")
def trading_close(req: ClosePositionRequest):
    trader = AlpacaTrader()
    if not trader.is_configured:
        raise HTTPException(400, "Alpaca API keys not configured")
    try:
        return trader.close_position(req.symbol.upper())
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/replay/{ticker}")
def replay_data(ticker: str, start_date: str, end_date: str, interval: str = "1d"):
    if interval not in ("1d", "1h"):
        raise HTTPException(400, "interval must be '1d' or '1h'")
    result = fetch_replay_data(ticker.upper(), start_date, end_date, interval)
    if not result:
        raise HTTPException(404, f"No price data for {ticker} in range {start_date} to {end_date}")
    return result


# --- TTS Endpoints ---

@app.post("/tts/generate")
def tts_generate(req: TTSRequest):
    if not tts_engine.is_configured or not tts_engine.voice_id:
        return {"audio_url": None, "script": None, "error": "TTS unavailable"}
    try:
        filename, script = tts_engine.generate_and_cache(
            req.signal_result, req.ticker.upper(), req.quarter.upper()
        )
        if filename and script:
            return {
                "audio_url": f"/tts/audio/{filename}",
                "script": script,
            }
        return {"audio_url": None, "script": None, "error": "TTS synthesis failed"}
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        return {"audio_url": None, "script": None, "error": "TTS unavailable"}


@app.get("/tts/audio/{filename}")
def tts_audio(filename: str):
    # Sanitize filename
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    filepath = BACKEND_DIR / "cache" / filename
    if not filepath.exists() or not filename.endswith(".mp3"):
        raise HTTPException(404, "Audio file not found")
    return FileResponse(
        filepath,
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


@app.get("/simulator")
def serve_simulator():
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
