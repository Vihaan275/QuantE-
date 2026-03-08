import re
from typing import Optional


# Patterns that indicate the ACTUAL start of Q&A (not just a mention of it)
# These must be action phrases where the operator is transitioning to Q&A
QA_START_PATTERNS = [
    re.compile(r"we\s+will\s+now\s+(open|take|begin)", re.IGNORECASE),
    re.compile(r"open\s+(the\s+)?(line|call|floor)\s+(for|to)\s+questions", re.IGNORECASE),
    re.compile(r"open\s+it\s+up\s+for\s+questions", re.IGNORECASE),
    re.compile(r"take\s+your\s+(first\s+)?questions", re.IGNORECASE),
    re.compile(r"begin\s+(the|our)\s+question", re.IGNORECASE),
    re.compile(r"(let'?s|we'?ll)\s+go\s+ahead\s+and\s+(open|take|start)", re.IGNORECASE),
    re.compile(r"please\s+poll\s+for\s+questions", re.IGNORECASE),
    re.compile(r"first\s+question\s+(comes?|is)\s+from", re.IGNORECASE),
    re.compile(r"operator\s+instructions", re.IGNORECASE),
]

# Speaker detection patterns
SPEAKER_PATTERNS = [
    re.compile(r"^([A-Z][a-zA-Z\s\.\-']+)\s*[-\u2013\u2014]\s*(.+?)\s*[-\u2013\u2014]\s*(.+?)$"),  # FIRST LAST - Title - Company
    re.compile(r"^([A-Z][a-zA-Z\s\.\-']+)\s*\(([^)]+)\)\s*:"),  # Name (Title):
    re.compile(r"^([A-Z][a-zA-Z\s\.\-']+)\s*:"),  # SPEAKER NAME:
]

# Keywords suggesting analyst role
ANALYST_KEYWORDS = [
    "analyst", "research", "capital", "securities", "investment",
    "partners", "advisors", "bank", "credit suisse",
    "goldman", "morgan stanley", "jpmorgan", "barclays", "citi",
    "bernstein", "wolfe", "oppenheimer", "piper", "wedbush",
    "keybanc", "cowen", "needham", "evercore", "jefferies",
    "raymond james", "stifel", "mizuho", "ubs", "hsbc",
    "baird", "susquehanna", "truist", "btig",
]

OPERATOR_KEYWORDS = ["operator", "moderator", "conference"]

MANAGEMENT_KEYWORDS = [
    "ceo", "cfo", "coo", "cto", "president", "chief",
    "vice president", "vp", "director", "head of",
    "founder", "chairman", "executive",
]


def _classify_speaker(speaker: str, context: str = "") -> str:
    combined = (speaker + " " + context).lower()
    for kw in OPERATOR_KEYWORDS:
        if kw in combined:
            return "operator"
    for kw in MANAGEMENT_KEYWORDS:
        if kw in combined:
            return "management"
    for kw in ANALYST_KEYWORDS:
        if kw in combined:
            return "analyst"
    return "management"  # default to management


def _detect_speaker(line: str) -> Optional[tuple[str, str]]:
    for pattern in SPEAKER_PATTERNS:
        match = pattern.match(line.strip())
        if match:
            speaker = match.group(1).strip()
            context = match.group(2).strip() if match.lastindex >= 2 else ""
            return speaker, context
    return None


def _split_sentences(text: str) -> list[str]:
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def _is_qa_start(line: str) -> bool:
    """Check if a line signals the actual start of Q&A, not just a mention."""
    for pattern in QA_START_PATTERNS:
        if pattern.search(line):
            return True
    return False


def parse_transcript(raw_text: str) -> dict:
    lines = raw_text.split("\n")

    # Find Q&A section boundary — scan for the ACTUAL transition
    # Skip early mentions like "there will be a Q&A session"
    qa_start_idx = None

    # First pass: find speaker-tagged lines to understand structure
    # The Q&A starts when: (a) operator says "we will now open/take questions"
    # or (b) the first analyst speaks after the prepared remarks
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_qa_start(stripped):
            qa_start_idx = i
            break

    # Fallback: if no explicit Q&A marker, find where the first analyst speaks
    # (after at least 20% of the transcript as prepared remarks)
    if qa_start_idx is None:
        min_prepared = int(len(lines) * 0.2)
        for i, line in enumerate(lines):
            if i < min_prepared:
                continue
            speaker_match = _detect_speaker(line.strip())
            if speaker_match:
                speaker, context = speaker_match
                if _classify_speaker(speaker, context) == "analyst":
                    qa_start_idx = i
                    break

    # Final fallback: assume last 40% is Q&A
    if qa_start_idx is None:
        qa_start_idx = int(len(lines) * 0.6)

    prepared_lines = lines[:qa_start_idx]
    qa_lines = lines[qa_start_idx:]

    # Parse into tagged sentences
    all_sentences = []
    sentence_idx = 0
    current_speaker = "Unknown"
    current_speaker_type = "management"

    for section_name, section_lines in [("prepared", prepared_lines), ("qa", qa_lines)]:
        for line in section_lines:
            line = line.strip()
            if not line:
                continue

            # Check for new speaker
            speaker_match = _detect_speaker(line)
            if speaker_match:
                current_speaker, context = speaker_match
                current_speaker_type = _classify_speaker(current_speaker, context)
                # Remove speaker prefix from line for sentence extraction
                for pattern in SPEAKER_PATTERNS:
                    line = pattern.sub("", line).strip()
                if not line:
                    continue

            sentences = _split_sentences(line)
            for sent in sentences:
                all_sentences.append({
                    "sentence_index": sentence_idx,
                    "text": sent,
                    "speaker": current_speaker,
                    "speaker_type": current_speaker_type,
                    "section": section_name,
                })
                sentence_idx += 1

    # Separate sections
    prepared_sentences = [s for s in all_sentences if s["section"] == "prepared"]
    qa_sentences = [s for s in all_sentences if s["section"] == "qa"]

    return {
        "prepared_remarks": prepared_sentences,
        "qa_section": qa_sentences,
        "all_sentences": all_sentences,
        "stats": {
            "total_sentences": len(all_sentences),
            "prepared_count": len(prepared_sentences),
            "qa_count": len(qa_sentences),
            "speakers": list(set(s["speaker"] for s in all_sentences)),
        },
    }
