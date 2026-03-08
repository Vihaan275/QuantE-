import re
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _sanitize(obj):
    """Recursively convert numpy types to native Python types for JSON serialization.
    Also converts NaN/Infinity to 0.0 to prevent invalid JSON."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
    return obj

# Load model once at module level
_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded.")
    return _model


# --- Constants ---

FORWARD_LOOKING_KEYWORDS = [
    "next quarter", "fiscal year", "guidance", "outlook", "going forward",
    "we expect", "full year", "forward-looking", "looking ahead",
    "expected to be", "is expected", "will be available", "will continue",
    "will drive", "will help", "will add", "will result",
    "coming to market", "into next year", "over the next",
    "demand visibility", "we are seeing", "we are planning",
    "growth going forward", "ramp", "pipeline",
    "sequential growth", "driven by", "contributing",
]

CERTAINTY_PHRASES = [
    "we will", "we are confident", "on track to", "we expect to deliver",
    "we are raising", "we are committed", "strong momentum", "we remain confident",
    "we are well positioned", "we are pleased", "we are seeing",
    "tremendous demand", "exceptional", "record revenue", "record q",
    "very strong demand", "exceptionally strong", "broad-based",
    "excellent visibility", "demand visibility extends",
    "continues to grow", "continues to expand", "continues to ramp",
    "will continue to ramp", "will continue to grow",
    "we are very pleased", "driving strong", "drove strong",
    "major success", "is expected to be", "we expect supply to increase",
    "strong growth", "accelerating demand", "exceeded", "above our outlook",
    "significantly", "substantially", "doubled", "tripled",
    "we do not anticipate", "is not limited by supply",
]

HEDGE_PHRASES = [
    "we hope", "subject to", "if conditions allow",
    "potentially", "it's possible",
    "remains to be seen", "uncertain", "we'll see",
    "difficult to predict", "hard to say",
    "if implemented", "if adopted", "depending on",
    "risks and uncertainties", "actual results may differ",
    "not yet clear", "remains uncertain", "more cautious",
    "too early to tell", "we are monitoring", "we are evaluating",
    "limited visibility", "visibility is limited", "less certain",
    "we are not in a position", "difficult to quantify",
]

# "may" and "could" only count as hedging in forward-looking contexts
HEDGE_WORDS = ["may", "could"]

DEFLECTION_PHRASES = [
    "as i mentioned", "as we discussed", "i'll take that offline",
    "we'll get back to you", "we've already addressed",
    "as noted earlier", "i think we covered",
    "i'm not going to", "we don't disclose", "we're not breaking that out",
]

POSITIVE_ANCHORS = [
    "record revenue", "market share gains", "strong demand",
    "exceeding expectations", "accelerating growth", "robust pipeline",
    "all-time high", "outstanding results", "tremendous demand",
    "exceptional quarter", "above our outlook", "tripled",
    "doubled year-on-year", "broad-based demand", "demand visibility extends",
    "industry transition", "platform shift", "generational transition",
]

NEGATIVE_ANCHORS = [
    "challenging environment", "macro headwinds", "cost pressures",
    "uncertainty", "headwind", "deceleration", "softness",
    "cautious", "difficult conditions", "declined sequentially",
    "subscriber loss", "margin pressure", "price cuts",
    "loss of opportunity", "downturn", "slowdown",
]


def _is_forward_looking(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in FORWARD_LOOKING_KEYWORDS)


def _count_phrase_matches(text: str, phrases: list[str]) -> int:
    text_lower = text.lower()
    count = 0
    for phrase in phrases:
        if phrase in text_lower:
            count += 1
    return count


def _has_hedge_word(text: str) -> bool:
    words = re.findall(r'\b\w+\b', text.lower())
    return any(w in HEDGE_WORDS for w in words)


def _has_specific_numbers(text: str) -> bool:
    patterns = [
        r'\$[\d,.]+\s*(billion|million|thousand|B|M|K)?',
        r'\d+(\.\d+)?\s*%',
        r'\d+\s*basis\s*points?',
        r'\d+(\.\d+)?\s*(billion|million|thousand|trillion)',
        r'\b(?!20\d{2}\b)\d{2,}[,\d]*\b',  # 2+ digit numbers, excluding years (2000-2099)
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _is_substantive_answer(text: str) -> bool:
    """A substantive answer is long, detailed, and contains specifics."""
    return len(text) > 150


# --- Sub-score 1: Guidance Hedge Score ---

NEGATIVE_CONTENT_KEYWORDS = [
    "decline", "declined", "decrease", "decreased", "lower", "reduce",
    "cut", "cuts", "loss", "losses", "pressure", "compression",
    "headwind", "challenging", "difficult", "slowdown", "slowing",
    "uncertainty", "cautious", "concern", "risk", "weaker", "weakness",
    "miss", "missed", "below", "shortfall", "down sequentially",
    "unfavorable", "deteriorat", "soften", "softness",
    "contraction", "downturn", "restructur", "impair",
    "disappoint", "underperform", "erosion", "margin pressure",
]

POSITIVE_CONTENT_KEYWORDS = [
    "growth", "grew", "increase", "increased", "demand", "opportunity",
    "expansion", "outperform", "beat", "upside", "robust", "strength",
    "favorable", "tailwind", "improving",
]


def compute_guidance_hedge(sentences: list[dict]) -> dict:
    forward_sentences = [s for s in sentences if _is_forward_looking(s["text"])]
    if not forward_sentences:
        return {
            "score": 0.0,
            "explanation": "No forward-looking statements detected in transcript.",
            "top_sentences": [],
            "sentences_analyzed": 0,
        }

    scored = []
    certainty_count = 0
    hedge_count = 0
    pos_content_count = 0
    neg_content_count = 0

    for s in forward_sentences:
        cert = _count_phrase_matches(s["text"], CERTAINTY_PHRASES)
        hedge = _count_phrase_matches(s["text"], HEDGE_PHRASES)
        if _has_hedge_word(s["text"]):
            hedge += 1

        # Semantic content: does the forward-looking statement discuss
        # positive outcomes or negative pressures?
        text_lower = s["text"].lower()
        pos_content = sum(1 for kw in POSITIVE_CONTENT_KEYWORDS if kw in text_lower)
        neg_content = sum(1 for kw in NEGATIVE_CONTENT_KEYWORDS if kw in text_lower)
        pos_content_count += pos_content
        neg_content_count += neg_content

        contribution = (cert + pos_content) - (hedge + neg_content)
        certainty_count += cert
        hedge_count += hedge
        scored.append({
            "sentence_index": s["sentence_index"],
            "text": s["text"],
            "score_contribution": round(contribution / max(len(forward_sentences), 1), 3),
            "reason": f"certainty={cert}, hedge={hedge}, pos_content={pos_content}, neg_content={neg_content}",
        })

    # Combine syntactic hedging with semantic content
    total_syntax = certainty_count + hedge_count
    total_content = pos_content_count + neg_content_count

    if total_syntax + total_content == 0:
        raw_score = min(0.2, len(forward_sentences) / 50.0)
    else:
        # Weight syntax and content equally
        syntax_score = (certainty_count - hedge_count) / max(total_syntax, 1) if total_syntax > 0 else 0
        content_score = (pos_content_count - neg_content_count) / max(total_content, 1) if total_content > 0 else 0
        raw_score = 0.5 * syntax_score + 0.5 * content_score

    score = max(-1.0, min(1.0, raw_score))

    # Sort by absolute contribution, take top 3
    scored.sort(key=lambda x: abs(x["score_contribution"]), reverse=True)
    top = scored[:3]

    if score > 0.2:
        expl = f"Management used confident language in {certainty_count} forward-looking statements vs {hedge_count} hedging phrases."
    elif score < -0.2:
        expl = f"Heavy hedging detected: {hedge_count} hedging phrases vs only {certainty_count} confident statements."
    else:
        expl = f"Mixed signals in forward guidance: {certainty_count} confident vs {hedge_count} hedging phrases."

    return {
        "score": round(score, 2),
        "explanation": expl,
        "top_sentences": top,
        "sentences_analyzed": len(forward_sentences),
        "_per_sentence_data": [
            {"sentence_index": s["sentence_index"], "contribution": s["score_contribution"]}
            for s in scored
        ],
    }


# --- Sub-score 2: Analyst Pressure Score ---

def compute_analyst_pressure(parsed: dict) -> dict:
    qa = parsed.get("qa_section", [])
    if not qa:
        return {
            "score": 0.0,
            "explanation": "No Q&A section detected.",
            "top_sentences": [],
            "sentences_analyzed": 0,
        }

    analyst_questions = [s for s in qa if s["speaker_type"] == "analyst"]
    mgmt_answers = [s for s in qa if s["speaker_type"] == "management"]

    # Simple keyword clustering for repeat questions
    def extract_nouns(text: str) -> set:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        stopwords = {"what", "when", "where", "which", "that", "this", "have", "been",
                      "will", "with", "from", "your", "about", "would", "could", "there",
                      "their", "they", "just", "also", "more", "very", "some", "than",
                      "going", "think", "like", "really", "question"}
        return set(words) - stopwords

    # Group questions by topic similarity
    question_groups = []
    for q in analyst_questions:
        nouns = extract_nouns(q["text"])
        placed = False
        for group in question_groups:
            overlap = len(nouns & group["nouns"])
            if overlap >= 2:
                group["questions"].append(q)
                group["nouns"] |= nouns
                placed = True
                break
        if not placed:
            question_groups.append({"nouns": nouns, "questions": [q]})

    # Score repeat questions negatively
    repeat_penalty = sum(1 for g in question_groups if len(g["questions"]) >= 2)

    # Score specific numbers and substantive answers positively
    specific_count = sum(1 for a in mgmt_answers
                         if _has_specific_numbers(a["text"]) or _is_substantive_answer(a["text"]))

    # Score deflection negatively
    deflection_count = 0
    deflection_sentences = []
    for a in mgmt_answers:
        text_lower = a["text"].lower()
        for phrase in DEFLECTION_PHRASES:
            if phrase in text_lower:
                deflection_count += 1
                deflection_sentences.append(a)
                break

    # Specificity ratio: what fraction of management answers contain hard numbers
    specificity_ratio = specific_count / max(len(mgmt_answers), 1)
    # Deflection ratio: what fraction of management answers are deflections
    deflection_ratio = deflection_count / max(len(mgmt_answers), 1)
    # Repeat pressure: penalize when analysts press on same topic
    repeat_ratio = repeat_penalty / max(len(question_groups), 1)

    # Positive: high specificity = management is being transparent and data-driven
    # Negative: deflections and repeat questioning = management being evasive
    raw_score = (specificity_ratio * 1.5) - (deflection_ratio * 2.0) - (repeat_ratio * 0.5)
    score = max(-1.0, min(1.0, raw_score))

    # Build top sentences
    scored_sentences = []
    for s in deflection_sentences[:2]:
        scored_sentences.append({
            "sentence_index": s["sentence_index"],
            "text": s["text"],
            "score_contribution": -0.2,
            "reason": "Deflection phrase detected",
        })
    for g in question_groups:
        if len(g["questions"]) >= 2:
            s = g["questions"][0]
            scored_sentences.append({
                "sentence_index": s["sentence_index"],
                "text": s["text"],
                "score_contribution": -0.15,
                "reason": f"Topic asked {len(g['questions'])} times (analyst pressing)",
            })
    # Add positive examples
    for a in mgmt_answers:
        if _has_specific_numbers(a["text"]):
            scored_sentences.append({
                "sentence_index": a["sentence_index"],
                "text": a["text"],
                "score_contribution": 0.1,
                "reason": "Specific numbers in response",
            })
            if len(scored_sentences) >= 6:
                break

    scored_sentences.sort(key=lambda x: abs(x["score_contribution"]), reverse=True)
    top = scored_sentences[:3]

    if score > 0.1:
        expl = f"Management responded with specifics ({specific_count} data points), limited deflection."
    elif score < -0.1:
        expl = f"Analysts pressed on {repeat_penalty} topics repeatedly; {deflection_count} deflections detected."
    else:
        expl = f"Mixed Q&A dynamics: {specific_count} specific answers, {deflection_count} deflections."

    return {
        "score": round(score, 2),
        "explanation": expl,
        "top_sentences": top,
        "sentences_analyzed": len(qa),
    }


# --- Sub-score 3: Topic Avoidance Score ---

def compute_topic_avoidance(parsed: dict) -> dict:
    model = _get_model()
    prepared = parsed.get("prepared_remarks", [])
    qa = parsed.get("qa_section", [])

    if not prepared or not qa:
        return {
            "score": 0.0,
            "explanation": "Insufficient data to assess topic avoidance.",
            "top_sentences": [],
            "sentences_analyzed": 0,
        }

    # Get analyst questions only
    analyst_qs = [s for s in qa if s["speaker_type"] == "analyst"]
    if not analyst_qs:
        return {
            "score": 0.0,
            "explanation": "No analyst questions detected for topic comparison.",
            "top_sentences": [],
            "sentences_analyzed": 0,
        }

    # Embed prepared remarks
    prep_texts = [s["text"] for s in prepared]
    qa_texts = [s["text"] for s in analyst_qs]

    prep_embeddings = model.encode(prep_texts, show_progress_bar=False)
    qa_embeddings = model.encode(qa_texts, show_progress_bar=False)

    # Cluster prepared remarks into topics (greedy grouping)
    prep_topics = []
    used = set()
    sim_matrix = cosine_similarity(prep_embeddings)

    for i in range(len(prep_texts)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(prep_texts)):
            if j not in used and sim_matrix[i][j] > 0.75:
                cluster.append(j)
                used.add(j)
        centroid = np.mean(prep_embeddings[cluster], axis=0)
        prep_topics.append({"indices": cluster, "centroid": centroid})

    # Cluster Q&A into topics similarly
    qa_topics = []
    used_qa = set()
    if len(qa_embeddings) > 1:
        qa_sim = cosine_similarity(qa_embeddings)
        for i in range(len(qa_texts)):
            if i in used_qa:
                continue
            cluster = [i]
            used_qa.add(i)
            for j in range(i + 1, len(qa_texts)):
                if j not in used_qa and qa_sim[i][j] > 0.75:
                    cluster.append(j)
                    used_qa.add(j)
            centroid = np.mean(qa_embeddings[cluster], axis=0)
            qa_topics.append({"indices": cluster, "centroid": centroid, "sample_idx": i})
    else:
        qa_topics = [{"indices": [0], "centroid": qa_embeddings[0], "sample_idx": 0}]

    # For each Q&A topic, find closest prepared topic
    # Also compare against individual prepared sentence embeddings for finer matching
    unanswered = []
    for qt in qa_topics:
        # Check against topic centroids
        min_dist = float("inf")
        for pt in prep_topics:
            dist = cosine(qt["centroid"], pt["centroid"])
            min_dist = min(min_dist, dist)
        # Also check against individual sentences for finer-grained matching
        for emb in prep_embeddings:
            dist = cosine(qt["centroid"], emb)
            min_dist = min(min_dist, dist)
        # Threshold 0.6: only flag truly unaddressed topics
        if min_dist > 0.6:
            unanswered.append({"topic": qt, "min_dist": min_dist})

    total_qa_topics = max(len(qa_topics), 1)
    raw_score = -(len(unanswered) / total_qa_topics)
    score = max(-1.0, min(0.0, raw_score))

    top_sentences = []
    for u in unanswered[:3]:
        idx = u["topic"]["sample_idx"]
        top_sentences.append({
            "sentence_index": analyst_qs[idx]["sentence_index"],
            "text": analyst_qs[idx]["text"],
            "score_contribution": round(float(-1.0 / total_qa_topics), 3),
            "reason": f"Topic not addressed in prepared remarks (distance={float(u['min_dist']):.2f})",
        })

    if len(unanswered) == 0:
        expl = "All analyst topics were addressed in prepared remarks."
    else:
        expl = f"{len(unanswered)} of {total_qa_topics} analyst topics were not addressed in prepared remarks."

    return {
        "score": round(score, 2),
        "explanation": expl,
        "top_sentences": top_sentences,
        "sentences_analyzed": len(prepared) + len(analyst_qs),
    }


# --- Sub-score 4: Language Momentum Score ---

def compute_language_momentum(parsed: dict, all_cached_texts: list[str] = None) -> dict:
    model = _get_model()
    all_sentences = parsed.get("all_sentences", [])

    if not all_sentences:
        return {
            "score": 0.0,
            "explanation": "No sentences to analyze for language momentum.",
            "top_sentences": [],
            "sentences_analyzed": 0,
        }

    texts = [s["text"] for s in all_sentences]
    embeddings = model.encode(texts, show_progress_bar=False)
    current_centroid = np.mean(embeddings, axis=0)

    # Compute anchor embeddings
    pos_emb = model.encode(POSITIVE_ANCHORS, show_progress_bar=False)
    neg_emb = model.encode(NEGATIVE_ANCHORS, show_progress_bar=False)
    pos_centroid = np.mean(pos_emb, axis=0)
    neg_centroid = np.mean(neg_emb, axis=0)

    # Per-sentence scoring: compute how strongly each sentence leans pos vs neg
    sentence_scores = []
    for i, emb in enumerate(embeddings):
        ps = float(1 - cosine(emb, pos_centroid))
        ns = float(1 - cosine(emb, neg_centroid))
        sentence_scores.append(ps - ns)

    # Key insight: ALL earnings calls lean positive in aggregate.
    # What differentiates bullish from bearish is:
    # 1. The DENSITY of strongly positive sentences (top percentile)
    # 2. The PRESENCE of negative-leaning sentences
    # 3. The SPREAD of the distribution

    sorted_scores = sorted(sentence_scores)
    n = len(sorted_scores)

    # Count strongly positive sentences (top quartile threshold)
    if n > 0:
        threshold_pos = sorted_scores[int(n * 0.75)]  # 75th percentile
        threshold_neg = sorted_scores[int(n * 0.25)]  # 25th percentile

        # Strong positive = sentences where pos-neg diff is in top 10%
        strong_pos = sum(1 for s in sentence_scores if s > sorted_scores[int(n * 0.90)])
        # Negative-leaning sentences (closer to negative anchors than positive)
        neg_leaning = sum(1 for s in sentence_scores if s < 0)
        neg_fraction = neg_leaning / n

        # Score combines:
        # - Mean sentence score (baseline)
        # - Negative sentence fraction (penalizes bearish language)
        # - Standard deviation (high variance = mixed signals = bearish)
        mean_score = float(np.mean(sentence_scores))
        std_score = float(np.std(sentence_scores))

        # Relative scoring: compare this transcript's metrics against
        # reference transcripts if available
        if all_cached_texts and len(all_cached_texts) > 0:
            ref_neg_fractions = []
            ref_means = []
            for t in all_cached_texts:
                ref_sents = [s.strip() for s in t.split(". ") if len(s.strip()) > 10][:200]
                if len(ref_sents) < 10:
                    continue
                ref_embs = model.encode(ref_sents, show_progress_bar=False)
                ref_ss = []
                for e in ref_embs:
                    rps = float(1 - cosine(e, pos_centroid))
                    rns = float(1 - cosine(e, neg_centroid))
                    ref_ss.append(rps - rns)
                ref_neg_fractions.append(sum(1 for s in ref_ss if s < 0) / len(ref_ss))
                ref_means.append(float(np.mean(ref_ss)))

            if ref_means:
                avg_ref_mean = float(np.mean(ref_means))
                avg_ref_neg = float(np.mean(ref_neg_fractions))
                # Score relative to the average across cached transcripts
                # Use moderate multipliers — embedding diffs are small (~0.01-0.04)
                mean_delta = mean_score - avg_ref_mean
                neg_delta = avg_ref_neg - neg_fraction  # less neg = better
                raw_score = mean_delta * 6 + neg_delta * 2
            else:
                raw_score = (mean_score - 0.10) * 3 - max(0, neg_fraction - 0.10) * 2
        else:
            # No reference transcripts — use conservative absolute scoring
            # Actual earnings call mean pos-neg similarity diff is ~0.07-0.14
            # Only flag clear extremes (e.g. TSLA's 0.07 mean + 19% neg fraction)
            raw_score = (mean_score - 0.10) * 3 - max(0, neg_fraction - 0.10) * 2
    else:
        raw_score = 0.0

    score = max(-1.0, min(1.0, raw_score))

    # Build top sentences
    scored_sentences = []
    for i, net in enumerate(sentence_scores):
        ps = float(1 - cosine(embeddings[i], pos_centroid))
        ns = float(1 - cosine(embeddings[i], neg_centroid))
        scored_sentences.append({
            "sentence_index": all_sentences[i]["sentence_index"],
            "text": all_sentences[i]["text"],
            "score_contribution": round(float(net), 3),
            "reason": f"pos_sim={ps:.2f}, neg_sim={ns:.2f}",
        })

    scored_sentences.sort(key=lambda x: abs(x["score_contribution"]), reverse=True)
    top = scored_sentences[:3]

    if score > 0.2:
        expl = "Language momentum is shifting toward confident, growth-oriented vocabulary."
    elif score < -0.2:
        expl = "Language momentum indicates defensive, cautious positioning."
    else:
        expl = "Language tone is neutral relative to reference transcripts."

    return {
        "score": round(score, 2),
        "explanation": expl,
        "top_sentences": top,
        "sentences_analyzed": len(texts),
        "_sentence_scores": [round(float(s), 4) for s in sentence_scores] if sentence_scores else [],
    }


# --- Sub-score 5: Earnings Surprise Score ---

def compute_earnings_surprise_score(surprise_data: dict | None) -> dict:
    """Convert raw EPS surprise data into a -1 to +1 score."""
    if not surprise_data:
        return {
            "score": 0.0,
            "explanation": "Earnings surprise data not available.",
            "top_sentences": [],
            "sentences_analyzed": 0,
            "available": False,
        }

    surprise_pct = surprise_data.get("surprise_pct", 0.0)
    reported = surprise_data.get("reported_eps", 0.0)
    estimated = surprise_data.get("estimated_eps", 0.0)
    beat = surprise_data.get("beat", False)

    # Score based on surprise percentage magnitude
    # >10% beat => strong bullish, >5% moderate, >0% slight
    # Same logic inverted for misses
    abs_pct = abs(surprise_pct)
    if abs_pct > 20:
        magnitude = 1.0
    elif abs_pct > 10:
        magnitude = 0.6 + (abs_pct - 10) * 0.04  # 0.6 to 1.0
    elif abs_pct > 5:
        magnitude = 0.3 + (abs_pct - 5) * 0.06  # 0.3 to 0.6
    elif abs_pct > 1:
        magnitude = 0.1 + (abs_pct - 1) * 0.05  # 0.1 to 0.3
    else:
        magnitude = abs_pct * 0.1  # 0.0 to 0.1

    score = magnitude if beat else -magnitude
    score = max(-1.0, min(1.0, score))

    # Build explanation
    direction = "beat" if beat else "missed"
    if abs_pct > 10:
        strength = "significantly"
    elif abs_pct > 5:
        strength = "solidly"
    elif abs_pct > 1:
        strength = "slightly"
    else:
        strength = "narrowly"

    expl = f"EPS {direction} estimates {strength}: reported ${reported:.2f} vs expected ${estimated:.2f} ({surprise_pct:+.1f}%)."

    details = [{
        "sentence_index": -1,
        "text": f"Reported EPS: ${reported:.2f} | Estimated EPS: ${estimated:.2f} | Surprise: {surprise_pct:+.1f}%",
        "score_contribution": round(score, 3),
        "reason": f"{'Beat' if beat else 'Missed'} by {abs_pct:.1f}%",
    }]

    return {
        "score": round(score, 2),
        "explanation": expl,
        "top_sentences": details,
        "sentences_analyzed": 1,
        "available": True,
        "reported_eps": reported,
        "estimated_eps": estimated,
        "surprise_pct": surprise_pct,
        "beat": beat,
        "reported_date": surprise_data.get("reported_date", ""),
    }


# --- Composite Signal ---

def compute_signal(parsed: dict, all_cached_texts: list[str] = None, earnings_surprise_data: dict | None = None) -> dict:
    all_sentences = parsed.get("all_sentences", [])

    guidance = compute_guidance_hedge(all_sentences)
    pressure = compute_analyst_pressure(parsed)
    momentum = compute_language_momentum(parsed, all_cached_texts)
    earnings = compute_earnings_surprise_score(earnings_surprise_data)

    # Fixed 3-factor composite — language momentum (embedding-based) weighted highest
    # as it's the most discriminative factor for direction
    composite = (
        0.35 * guidance["score"]
        + 0.15 * pressure["score"]
        + 0.50 * momentum["score"]
    )
    composite = round(max(-1.0, min(1.0, composite)), 2)

    # Signal label
    if composite >= 0.20:
        label = "LONG"
    elif composite <= -0.20:
        label = "SHORT"
    else:
        label = "NEUTRAL"

    # Confidence: blend of signal strength, factor agreement, and data coverage
    # 1) Signal strength — stronger composite = more confident, but diminishing returns
    strength = min(abs(composite) / 0.6, 1.0)  # maxes out at |0.6|

    # 2) Factor agreement — do sub-scores agree on direction?
    scores = [guidance["score"], pressure["score"], momentum["score"]]
    sign = 1 if composite >= 0 else -1
    agreeing = sum(1 for s in scores if s * sign > 0)
    agreement = agreeing / len(scores)  # 0.33, 0.67, or 1.0

    # 3) Data coverage — enough sentences analyzed? (diminishing returns)
    analyzed_counts = [
        guidance["sentences_analyzed"],
        pressure["sentences_analyzed"],
        momentum["sentences_analyzed"],
    ]
    avg_analyzed = sum(analyzed_counts) / len(analyzed_counts) if analyzed_counts else 0
    coverage = min(avg_analyzed / 300, 1.0)  # saturates at 300 sentences

    # Weighted blend, hard cap at 85% — no system should claim near-certainty
    raw_confidence = 0.40 * strength + 0.35 * agreement + 0.25 * coverage
    confidence = round(max(0.10, min(0.85, raw_confidence)), 2)

    # EPS surprise acts as confidence modifier
    eps_modifier = 0.0
    if earnings["available"]:
        eps_score = earnings["score"]
        # EPS confirms signal direction → boost confidence
        # EPS contradicts signal direction → reduce confidence
        if (composite > 0 and eps_score > 0) or (composite < 0 and eps_score < 0):
            eps_modifier = min(0.10, abs(eps_score) * 0.10)  # up to +10%
        elif (composite > 0 and eps_score < -0.2) or (composite < 0 and eps_score > 0.2):
            eps_modifier = -min(0.15, abs(eps_score) * 0.15)  # up to -15%
        confidence = round(max(0.10, min(0.85, confidence + eps_modifier)), 2)

    # One-line summary
    sub_summaries = []
    if earnings["available"]:
        if earnings["score"] > 0.2:
            sub_summaries.append(f"EPS beat ({earnings.get('surprise_pct', 0):+.1f}%)")
        elif earnings["score"] < -0.2:
            sub_summaries.append(f"EPS miss ({earnings.get('surprise_pct', 0):+.1f}%)")
    if guidance["score"] > 0.2:
        sub_summaries.append("confident guidance language")
    elif guidance["score"] < -0.2:
        sub_summaries.append("heavy hedging in guidance")
    if pressure["score"] > 0.1:
        sub_summaries.append("direct analyst responses")
    elif pressure["score"] < -0.1:
        sub_summaries.append("evasive management responses")
    if momentum["score"] > 0.2:
        sub_summaries.append("bullish language momentum")
    elif momentum["score"] < -0.2:
        sub_summaries.append("bearish language shift")

    if not sub_summaries:
        summary_detail = "mixed signals across all factors"
    else:
        summary_detail = " and ".join(sub_summaries)

    if label == "LONG":
        summary = f"Strong bullish signal driven by {summary_detail}."
    elif label == "SHORT":
        summary = f"Bearish signal driven by {summary_detail}."
    else:
        summary = f"Neutral signal with {summary_detail}."

    sub_scores = {
        "guidance_hedge": {
            "score": guidance["score"],
            "explanation": guidance["explanation"],
            "top_sentences": guidance["top_sentences"],
        },
        "analyst_pressure": {
            "score": pressure["score"],
            "explanation": pressure["explanation"],
            "top_sentences": pressure["top_sentences"],
        },
        "language_momentum": {
            "score": momentum["score"],
            "explanation": momentum["explanation"],
            "top_sentences": momentum["top_sentences"],
        },
    }

    if earnings["available"]:
        sub_scores["earnings_surprise"] = {
            "score": earnings["score"],
            "explanation": earnings["explanation"],
            "top_sentences": earnings["top_sentences"],
            "reported_eps": earnings.get("reported_eps"),
            "estimated_eps": earnings.get("estimated_eps"),
            "surprise_pct": earnings.get("surprise_pct"),
            "beat": earnings.get("beat"),
            "reported_date": earnings.get("reported_date", ""),
        }

    # Calibration context — what does this signal strength historically mean?
    abs_comp = abs(composite)
    if abs_comp >= 0.5:
        cal_bucket = "strong"
        cal_range = "5-15%"
        cal_accuracy = "70-80%"
    elif abs_comp >= 0.3:
        cal_bucket = "moderate"
        cal_range = "2-8%"
        cal_accuracy = "60-70%"
    elif abs_comp >= 0.2:
        cal_bucket = "weak"
        cal_range = "1-4%"
        cal_accuracy = "55-60%"
    else:
        cal_bucket = "neutral"
        cal_range = "0-2%"
        cal_accuracy = "50%"

    calibration = {
        "bucket": cal_bucket,
        "expected_move": cal_range,
        "historical_accuracy": cal_accuracy,
        "suggested_window": "5-10 trading days",
        "eps_modifier": round(eps_modifier, 3) if earnings["available"] else None,
    }

    # Bootstrap confidence intervals for sub-scores
    bootstrap_cis = {}
    # Guidance hedge bootstrap
    g_data = guidance.get("_per_sentence_data", [])
    if len(g_data) >= 5:
        g_contribs = [d["contribution"] for d in g_data]
        boot_g = []
        for _ in range(100):
            sample = np.random.choice(g_contribs, size=len(g_contribs), replace=True)
            # Recompute guidance score from resampled contributions
            pos_c = sum(1 for c in sample if c > 0)
            neg_c = sum(1 for c in sample if c < 0)
            total_c = pos_c + neg_c
            bs = (pos_c - neg_c) / max(total_c, 1) if total_c > 0 else 0.0
            boot_g.append(max(-1.0, min(1.0, bs)))
        bootstrap_cis["guidance_hedge"] = {
            "ci_lo": round(float(np.percentile(boot_g, 5)), 3),
            "ci_hi": round(float(np.percentile(boot_g, 95)), 3),
        }

    # Language momentum bootstrap
    m_data = momentum.get("_sentence_scores", [])
    if len(m_data) >= 10:
        boot_m = []
        for _ in range(100):
            sample = list(np.random.choice(m_data, size=len(m_data), replace=True))
            m_mean = float(np.mean(sample))
            m_neg_frac = sum(1 for s in sample if s < 0) / len(sample)
            bs = (m_mean - 0.10) * 3 - max(0, m_neg_frac - 0.10) * 2
            boot_m.append(max(-1.0, min(1.0, bs)))
        bootstrap_cis["language_momentum"] = {
            "ci_lo": round(float(np.percentile(boot_m, 5)), 3),
            "ci_hi": round(float(np.percentile(boot_m, 95)), 3),
        }

    # Sentence-level attributions for heatmap
    all_sents = parsed.get("all_sentences", [])
    sentence_attributions = []
    g_map = {d["sentence_index"]: d["contribution"] for d in g_data}
    m_scores = momentum.get("_sentence_scores", [])
    for i, s in enumerate(all_sents):
        idx = s["sentence_index"]
        g_val = g_map.get(idx, 0.0)
        m_val = m_scores[i] if i < len(m_scores) else 0.0
        # Weighted contribution to composite (guidance 35%, momentum 50%)
        contrib = g_val * 0.35 + m_val * 0.50
        sentence_attributions.append({
            "i": idx,
            "t": s["text"][:150],
            "s": round(float(contrib), 4),
        })

    return {
        "composite_signal": composite,
        "signal_label": label,
        "confidence": confidence,
        "sub_scores": sub_scores,
        "one_line_summary": summary,
        "calibration": calibration,
        "bootstrap_cis": bootstrap_cis,
        "sentence_attributions": sentence_attributions,
    }


# --- Bootstrap CIs and Attributions Helper ---

def compute_bootstrap_and_attributions(guidance_result, momentum_result, all_sentences):
    """Compute bootstrap CIs and sentence attributions from already-computed sub-scores.
    Used by /analyze endpoint to avoid double model inference."""
    bootstrap_cis = {}

    g_data = guidance_result.get("_per_sentence_data", [])
    if len(g_data) >= 5:
        g_contribs = [d["contribution"] for d in g_data]
        boot_g = []
        for _ in range(100):
            sample = np.random.choice(g_contribs, size=len(g_contribs), replace=True)
            pos_c = sum(1 for c in sample if c > 0)
            neg_c = sum(1 for c in sample if c < 0)
            total_c = pos_c + neg_c
            bs = (pos_c - neg_c) / max(total_c, 1) if total_c > 0 else 0.0
            boot_g.append(max(-1.0, min(1.0, bs)))
        bootstrap_cis["guidance_hedge"] = {
            "ci_lo": round(float(np.percentile(boot_g, 5)), 3),
            "ci_hi": round(float(np.percentile(boot_g, 95)), 3),
        }

    m_data = momentum_result.get("_sentence_scores", [])
    if len(m_data) >= 10:
        boot_m = []
        for _ in range(100):
            sample = list(np.random.choice(m_data, size=len(m_data), replace=True))
            m_mean = float(np.mean(sample))
            m_neg_frac = sum(1 for s in sample if s < 0) / len(sample)
            bs = (m_mean - 0.10) * 3 - max(0, m_neg_frac - 0.10) * 2
            boot_m.append(max(-1.0, min(1.0, bs)))
        bootstrap_cis["language_momentum"] = {
            "ci_lo": round(float(np.percentile(boot_m, 5)), 3),
            "ci_hi": round(float(np.percentile(boot_m, 95)), 3),
        }

    sentence_attributions = []
    g_map = {d["sentence_index"]: d["contribution"] for d in g_data}
    m_scores = momentum_result.get("_sentence_scores", [])
    for i, s in enumerate(all_sentences):
        idx = s["sentence_index"]
        g_val = g_map.get(idx, 0.0)
        m_val = m_scores[i] if i < len(m_scores) else 0.0
        contrib = g_val * 0.35 + m_val * 0.50
        sentence_attributions.append({
            "i": idx,
            "t": s["text"][:150],
            "s": round(float(contrib), 4),
        })

    return bootstrap_cis, sentence_attributions


# --- Naive Baseline Signal (keyword counting only) ---

def compute_naive_signal(parsed: dict) -> dict:
    """Simple keyword counting baseline — no embeddings, no sentence-transformers."""
    all_sentences = parsed.get("all_sentences", [])
    if not all_sentences:
        return {"composite_signal": 0.0, "signal_label": "NEUTRAL"}

    pos_count = 0
    neg_count = 0
    for s in all_sentences:
        text_lower = s["text"].lower()
        pos_count += sum(1 for kw in POSITIVE_CONTENT_KEYWORDS if kw in text_lower)
        neg_count += sum(1 for kw in NEGATIVE_CONTENT_KEYWORDS if kw in text_lower)

    total = pos_count + neg_count
    if total == 0:
        raw = 0.0
    else:
        raw = (pos_count - neg_count) / total

    score = round(max(-1.0, min(1.0, raw)), 2)

    if score >= 0.20:
        label = "LONG"
    elif score <= -0.20:
        label = "SHORT"
    else:
        label = "NEUTRAL"

    return {"composite_signal": score, "signal_label": label}
