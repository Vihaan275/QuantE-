# QuantE

**Earnings alpha through NLP.** QuantE analyzes earnings call transcripts using semantic embeddings and multi-factor signal composition to generate directional trading signals — then lets you replay them against real price action.

---

## What It Does

QuantE pulls earnings call transcripts from Alpha Vantage, runs them through a 3-factor NLP pipeline, and produces a composite long/short signal with calibrated confidence. You can then replay the signal against historical candlestick data in a simulated trading environment.

### Signal Pipeline

The composite signal is built from three independent sub-scores:

| Factor | Weight | What It Measures |
|---|---|---|
| **Language Momentum** | 50% | Semantic similarity to bullish/bearish anchor phrases using `all-MiniLM-L6-v2` embeddings, layered with keyword content analysis and cross-transcript relative comparison |
| **Guidance Hedge** | 35% | Forward-looking language hedging — detects when management softens guidance with qualifiers like "approximately", "subject to", etc. |
| **Analyst Pressure** | 15% | Q&A dynamics — whether analysts are pressing on the same topics, and whether management responds with hard numbers or deflections |

**EPS surprise** from reported vs estimated earnings modifies confidence (+10% confirming, -15% contradicting) but does not affect the directional signal.

### Confidence Scoring

Confidence is a 3-factor blend, hard-capped at 85%:
- **40%** signal strength (how far from zero)
- **35%** factor agreement (how many sub-scores agree on direction)
- **25%** data coverage (sentences analyzed vs expected)

### Replay Simulator

A candlestick replay that steps through post-earnings price action bar by bar:
- Auto-calculated TP/SL levels based on signal strength and confidence
- Shorts use less ambitious take-profit targets
- **Comparative Analysis** mode splits the view into two side-by-side charts — the full NLP pipeline vs a keyword-only baseline — with independent P&L tracking and a winner badge at completion

### Backtesting

Run the pipeline against historical transcripts with:
- Per-factor ablation study (accuracy and Sharpe per sub-score)
- Signal decay analysis across 1d–20d horizons with half-life computation
- Live calibration buckets (Strong Short → Strong Long)
- Baseline comparison (NLP vs keyword sentiment)
- Bootstrap confidence intervals for sub-scores

---

## Tech Stack

| Layer | Tech |
|---|---|
| Frontend | React 18 (CDN, no build step), Tailwind CSS, LightweightCharts, Recharts, Babel JSX |
| Backend | FastAPI, SSE streaming, sentence-transformers, NumPy |
| Data | Alpha Vantage (transcripts, earnings, prices) |
| Voice | ElevenLabs TTS for audio summaries |

---

## Setup

### 1. Environment

Create a `.env` in the project root:

```
AV_API_KEY=your_alpha_vantage_key
ELEVENLABS_API_KEY=your_elevenlabs_key  # optional, for voice summaries
```

### 2. Backend

```bash
cd backend
pip install fastapi uvicorn sentence-transformers numpy requests python-dotenv
uvicorn main:app --reload --port 8000
```

The first run will download the `all-MiniLM-L6-v2` model (~80MB).

### 3. Frontend

Open `frontend/index.html` in a browser. No build step needed — everything loads from CDN.

---

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI endpoints, SSE streaming
│   ├── signal_engine.py        # 3-factor NLP pipeline
│   ├── transcript_fetcher.py   # Alpha Vantage transcript retrieval
│   ├── transcript_parser.py    # Earnings call text parsing
│   ├── earnings_fetcher.py     # EPS surprise data
│   ├── price_fetcher.py        # Historical candlestick data
│   ├── backtester.py           # Ablation, decay, calibration
│   ├── trader.py               # Simulated paper trading logic
│   └── tts_engine.py           # ElevenLabs voice summaries
├── frontend/
│   └── index.html              # Full React SPA
└── .env                        # API keys (not committed)
```
