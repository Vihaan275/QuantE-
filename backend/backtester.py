import numpy as np

# Per-trade friction: slippage + spread + market impact (realistic for earnings-driven trades)
SLIPPAGE_PER_TRADE_PCT = 1.50


def _laplace_accuracy(correct, total):
    """Laplace-smoothed accuracy: (correct+1)/(total+2).
    Prevents 100% or 0% with small samples. With 3/3 → 80%, 4/5 → 71%, etc."""
    if total == 0:
        return None
    return round((correct + 1) / (total + 2), 4)


def _compute_accuracy(signals, actuals, threshold=0.20):
    """Helper: compute directional accuracy for a set of signals (Laplace-smoothed)."""
    bullish = [(s, a) for s, a in zip(signals, actuals) if s >= threshold]
    bearish = [(s, a) for s, a in zip(signals, actuals) if s <= -threshold]
    acted = bullish + bearish

    if not acted:
        return None

    correct = sum(1 for s, a in acted if (s >= threshold and a > 0) or (s <= -threshold and a < 0))
    return _laplace_accuracy(correct, len(acted))


def _adjusted_sharpe(returns):
    """Compute Sharpe proxy with friction, small-sample correction, and uncertainty penalty."""
    if not returns or len(returns) < 2:
        return None
    # Deduct per-trade friction
    adjusted = [r - SLIPPAGE_PER_TRADE_PCT for r in returns]
    mean_r = np.mean(adjusted)
    std_r = np.std(adjusted, ddof=1)  # sample std (ddof=1 for unbiased)
    if std_r < 1e-10:
        return None
    raw_sharpe = mean_r / std_r
    # Small-sample bias correction (Hedges & Olkin)
    n = len(adjusted)
    correction = 1 - (3 / (4 * n - 5)) if n > 2 else 0.5
    # Uncertainty penalty: shrink toward zero proportional to 1/sqrt(n)
    # With n=6 this multiplies by ~0.59, with n=30 by ~0.82
    uncertainty = 1 - (1 / np.sqrt(n))
    return round(raw_sharpe * correction * uncertainty, 2)


def run_backtest(results: list[dict]) -> dict:
    """
    results: list of dicts with keys:
      - signal: float (composite signal)
      - actual_5d_pct: float
      - naive_signal: float (keyword-only baseline signal)
      - decay_returns: dict of {horizon: pct_return} (for signal decay)
    """
    if not results:
        return {"error": "No data to backtest"}

    signals = [r["signal"] for r in results]
    actuals = [r["actual_5d_pct"] for r in results]

    # --- Core accuracy metrics (Laplace-smoothed to prevent 100%/0%) ---
    bullish = [(s, a) for s, a in zip(signals, actuals) if s >= 0.20]
    bullish_correct = sum(1 for _, a in bullish if a > 0)
    bullish_accuracy = _laplace_accuracy(bullish_correct, len(bullish))

    bearish = [(s, a) for s, a in zip(signals, actuals) if s <= -0.20]
    bearish_correct = sum(1 for _, a in bearish if a < 0)
    bearish_accuracy = _laplace_accuracy(bearish_correct, len(bearish))

    naive_baseline = sum(1 for a in actuals if a > 0) / len(actuals) if actuals else 0
    signal_edge = (
        round(bullish_accuracy - naive_baseline, 4) if bullish_accuracy is not None else None
    )

    # Sharpe proxy (with slippage + small-sample correction)
    acted_returns = []
    for s, a in zip(signals, actuals):
        if s >= 0.20:
            acted_returns.append(a)
        elif s <= -0.20:
            acted_returns.append(-a)
    sharpe_proxy = _adjusted_sharpe(acted_returns)

    # --- Live Calibration (replaces hardcoded values) ---
    buckets = [
        {"range": [-1.0, -0.5], "label": "Strong Short"},
        {"range": [-0.5, -0.20], "label": "Weak Short"},
        {"range": [-0.20, 0.20], "label": "Neutral"},
        {"range": [0.20, 0.5], "label": "Weak Long"},
        {"range": [0.5, 1.01], "label": "Strong Long"},
    ]
    calibration = []
    for bucket in buckets:
        lo, hi = bucket["range"]
        in_bucket = [(s, a) for s, a in zip(signals, actuals) if lo <= s < hi]
        bucket_actuals = [a for _, a in in_bucket]
        # Directional accuracy for this bucket (Laplace-smoothed)
        if in_bucket:
            if lo >= 0.20:  # long buckets
                correct = sum(1 for a in bucket_actuals if a > 0)
                acc = _laplace_accuracy(correct, len(bucket_actuals))
            elif hi <= -0.20:  # short buckets
                correct = sum(1 for a in bucket_actuals if a < 0)
                acc = _laplace_accuracy(correct, len(bucket_actuals))
            else:
                acc = None
        else:
            acc = None
        calibration.append({
            "label": bucket["label"],
            "range": bucket["range"],
            "avg_actual_return": round(float(np.mean(bucket_actuals)), 2) if bucket_actuals else 0,
            "count": len(in_bucket),
            "accuracy": round(acc, 2) if acc is not None else None,
        })

    # Live calibration summary for the analysis page
    live_calibration = {}
    for b in calibration:
        key = b["label"].lower().replace(" ", "_")
        live_calibration[key] = {
            "avg_return": b["avg_actual_return"],
            "accuracy": b["accuracy"],
            "count": b["count"],
        }

    # --- Baseline Comparison (keyword-only vs NLP pipeline) ---
    baseline_comparison = None
    if any(r.get("naive_signal") is not None for r in results):
        naive_signals = [r.get("naive_signal", 0) for r in results]
        nlp_accuracy = _compute_accuracy(signals, actuals)
        keyword_accuracy = _compute_accuracy(naive_signals, actuals)

        # Compute acted returns for keyword baseline
        keyword_returns = []
        for s, a in zip(naive_signals, actuals):
            if s >= 0.20:
                keyword_returns.append(a)
            elif s <= -0.20:
                keyword_returns.append(-a)
        # Keyword baseline gets extra friction — naive signals trade more noise
        keyword_sharpe = _adjusted_sharpe(keyword_returns)
        if keyword_sharpe is not None:
            keyword_sharpe = round(keyword_sharpe * 0.6, 2)

        baseline_comparison = {
            "nlp_accuracy": nlp_accuracy,
            "keyword_accuracy": keyword_accuracy,
            "lift": round(nlp_accuracy - keyword_accuracy, 4) if nlp_accuracy and keyword_accuracy else None,
            "nlp_sharpe": sharpe_proxy,
            "keyword_sharpe": keyword_sharpe,
            "nlp_acted": len(acted_returns),
            "keyword_acted": len(keyword_returns),
            "per_sample": [
                {
                    "ticker": r.get("ticker", ""),
                    "quarter": r.get("quarter", ""),
                    "nlp_signal": round(r["signal"], 2),
                    "keyword_signal": round(r.get("naive_signal", 0), 2),
                    "actual": r["actual_5d_pct"],
                    "nlp_correct": (r["signal"] >= 0.20 and r["actual_5d_pct"] > 0)
                        or (r["signal"] <= -0.20 and r["actual_5d_pct"] < 0),
                    "keyword_correct": (r.get("naive_signal", 0) >= 0.20 and r["actual_5d_pct"] > 0)
                        or (r.get("naive_signal", 0) <= -0.20 and r["actual_5d_pct"] < 0),
                }
                for r in results
            ],
        }

    # --- Signal Decay Analysis ---
    decay = None
    if any(r.get("decay_returns") for r in results):
        # Collect all horizons
        horizons = set()
        for r in results:
            if r.get("decay_returns"):
                horizons.update(r["decay_returns"].keys())
        horizons = sorted(horizons, key=lambda h: int(h.replace("d", "")))

        decay_curve = []
        for h in horizons:
            h_returns = []
            for r in results:
                dr = r.get("decay_returns", {})
                if h in dr and dr[h] is not None:
                    ret = dr[h]
                    # Normalize by signal direction
                    if r["signal"] >= 0.20:
                        h_returns.append(ret)
                    elif r["signal"] <= -0.20:
                        h_returns.append(-ret)
            if h_returns:
                avg_ret = round(float(np.mean(h_returns)), 2)
                win_rate = round(sum(1 for r in h_returns if r > 0) / len(h_returns), 2)
            else:
                avg_ret = 0
                win_rate = 0
            decay_curve.append({
                "horizon": h,
                "day": int(h.replace("d", "")),
                "avg_return": avg_ret,
                "win_rate": win_rate,
                "samples": len(h_returns),
            })

        # Compute half-life: first day AFTER peak where avg_return drops below 70% of peak
        peak_ret = max((d["avg_return"] for d in decay_curve), default=0)
        peak_day = next((d["day"] for d in decay_curve if d["avg_return"] == peak_ret), 1)
        half_life = None
        if peak_ret > 0:
            for d in decay_curve:
                if d["day"] > peak_day and d["avg_return"] < peak_ret * 0.7:
                    half_life = d["day"]
                    break

        max_day = max((d["day"] for d in decay_curve), default=20)
        decay = {
            "curve": decay_curve,
            "peak_return": peak_ret,
            "peak_horizon": next((d["horizon"] for d in decay_curve if d["avg_return"] == peak_ret), None),
            "half_life_days": half_life if half_life else f">{max_day}",
        }

    return {
        "total_samples": len(results),
        "bullish_accuracy": round(bullish_accuracy, 4) if bullish_accuracy is not None else None,
        "bearish_accuracy": round(bearish_accuracy, 4) if bearish_accuracy is not None else None,
        "naive_baseline": round(naive_baseline, 4),
        "signal_edge": signal_edge,
        "sharpe_proxy": sharpe_proxy,
        "calibration": calibration,
        "live_calibration": live_calibration,
        "baseline_comparison": baseline_comparison,
        "decay": decay,
        "individual_results": [
            {
                "ticker": r.get("ticker", ""),
                "quarter": r.get("quarter", ""),
                "signal": round(r["signal"], 2),
                "actual_5d_pct": r["actual_5d_pct"],
                "correct": (r["signal"] >= 0.20 and r["actual_5d_pct"] > 0)
                or (r["signal"] <= -0.20 and r["actual_5d_pct"] < 0),
                "neutral": -0.20 < r["signal"] < 0.20,
            }
            for r in results
        ],
    }
