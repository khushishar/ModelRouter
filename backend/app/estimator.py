import math
import re
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SIMPLE_PROMPT_PATTERN = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|good morning|good evening)[!. ]*\s*$",
IMAGE_GENERATION_PATTERN = re.compile(
    r"\b(?:generate|create|make|draw|paint|render|illustrate|design)\b"
    r".{0,80}\b(?:image|picture|illustration|artwork|photo|logo|poster|icon|wallpaper)\b"
    r"|\b(?:draw|paint|illustrate)\s+(?:me\s+)?(?:a|an)\s+\w+"
    r"|^\s*(?:an?\s+)?(?:image|picture|illustration|artwork|photo)\s+of\b",
    re.IGNORECASE,
)
COMPLEXITY_CUES = {
    "reasoning": (
        "analyze", "analyse", "compare", "evaluate", "prove", "debug",
        "design", "step by step", "reason", "trade-off", "tradeoff","explain", "justify", "assess", "synthesize", "deduce", "infer"
    ),
    "long_context": (
        "entire document", "whole document", "long document", "all of this",
        "contract", "transcript", "multiple files","entire conversation", "full context", "full conversation", "full transcript",
        "full document", "full text", "entire text", "entire conversation"
    ),
}
COMPLEXITY_CAPABILITY_FLOORS = {
    "low": None,
    "medium": 30.0,
    "high": 40.0,
}


def complexity_capability_floor(complexity: str) -> float | None:
    """Return the conservative benchmark floor for an estimated complexity."""
    return COMPLEXITY_CAPABILITY_FLOORS[complexity]


def estimate_tokens(text: str) -> int:
    """Return a conservative, model-independent estimate, not an exact count."""
    words = TOKEN_PATTERN.findall(text)
    if not words:
        return 0

    # Punctuation, code, and uncommon words can split into multiple tokens.
    return max(len(words), math.ceil(len(text.encode("utf-8")) / 3.5))


def analyze_prompt(
    prompt: str,
    *,
    file_token_estimate: int = 0,
    file_count: int = 0,
) -> dict[str, Any]:
    prompt_tokens = estimate_tokens(prompt)
    normalized = prompt.casefold()
    reasoning_signals = [
        cue for cue in COMPLEXITY_CUES["reasoning"] if cue in normalized
    ]
    long_context_signals = [
        cue for cue in COMPLEXITY_CUES["long_context"] if cue in normalized
    ]
    input_tokens = prompt_tokens + file_token_estimate
    complexity_signals = []
    if input_tokens > 12_000:
        complexity_signals.append("input_tokens_over_12000")
    if long_context_signals:
        complexity_signals.extend(long_context_signals)
    if reasoning_signals:
        complexity_signals.extend(reasoning_signals)
    if file_count:
        complexity_signals.append("attached_files")

    if input_tokens > 12_000 or long_context_signals:
        complexity = "high"
        complexity_confidence = "high"
    elif input_tokens > 2_500 or reasoning_signals or file_count:
        complexity = "medium"
        complexity_confidence = "medium"
    elif SIMPLE_PROMPT_PATTERN.match(normalized):
        complexity = "low"
        complexity_confidence = "high"
    else:
        complexity = "medium"
        complexity_confidence = "low"
        complexity_signals.append("ambiguous_prompt_defaulted_to_medium")

    task = "general"
    output_modality = "text"
    if IMAGE_GENERATION_PATTERN.search(normalized):
        task = "image_generation"
        output_modality = "image"
    elif any(cue in normalized for cue in ("code", "program", "debug", "function")):
        task = "coding"
    elif any(cue in normalized for cue in ("summarize", "summarise", "summary")):
        task = "summarization"
    elif any(cue in normalized for cue in ("translate", "translation")):
        task = "translation"
    elif any(cue in normalized for cue in ("classify", "classification")):
        task = "classification"
    elif reasoning_signals:
        task = "reasoning"

    return {
        "estimated_input_tokens": input_tokens,
        "prompt_tokens": prompt_tokens,
        "file_tokens": file_token_estimate,
        "complexity": complexity,
        "complexity_confidence": complexity_confidence,
        "complexity_signals": list(dict.fromkeys(complexity_signals)),
        "automatic_min_capability": complexity_capability_floor(complexity),
        "task": task,
        "output_modality": output_modality,
        "reasoning_signals": reasoning_signals,
        "long_context_signals": long_context_signals,
        "file_count": file_count,
        "estimation_note": (
            "Approximate, model-independent token estimate; actual provider "
            "tokenization may differ."
        ),
    }
