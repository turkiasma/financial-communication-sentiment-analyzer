"""An original, frozen keyword baseline; no model, external lexicon or fitting."""

import re

BASELINE_VERSION = "keyword_v1"
# Fixed before benchmark inference; these are ordinary words, not an imported lexicon.
POSITIVE_WORDS = frozenset({
    "growth", "grow", "grew", "growing", "increase", "increased", "increases",
    "improve", "improved", "improves", "improvement", "profit", "profits", "profitable",
    "gain", "gains", "gained", "strong", "record", "beat", "beats", "upgrade", "upgraded",
    "raise", "raised", "up", "rise", "rises", "rose", "positive", "robust", "expansion", "recovery",
})
NEGATIVE_WORDS = frozenset({
    "loss", "losses", "decline", "declined", "declines", "decrease", "decreased", "decreases",
    "fall", "falls", "fell", "down", "drop", "dropped", "drops", "weak", "weakness", "cut", "cuts",
    "reduced", "reduction", "downgrade", "downgraded", "risk", "risks", "challenging", "negative",
    "lower", "slump", "slumped",
})
NEGATIONS = frozenset({"no", "not", "never", "without", "neither", "nor"})


def predict_baseline(text: str) -> str:
    """Signed word counts; negation in the previous three words flips polarity.

    Punctuation ends the scope, contractions ending in n't count as negation,
    and ties/absence of matches yield neutral. No probabilities are invented.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Baseline text must be a non-empty string.")
    previous = []
    score = 0
    for token in re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?;,]", text.lower()):
        if token in ".!?;,":
            previous.clear()
            continue
        value = int(token in POSITIVE_WORDS) - int(token in NEGATIVE_WORDS)
        negated = sum(word in NEGATIONS or word.endswith("n't") for word in previous[-3:]) % 2
        score += -value if negated else value
        previous.append(token)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"
