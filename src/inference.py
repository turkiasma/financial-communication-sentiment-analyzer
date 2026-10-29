"""Reusable, CPU-only FinBERT inference and a terminal demonstration."""

import argparse
from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model_config import MODEL_ID, MODEL_REVISION

SENTIMENT_LABELS = frozenset({"positive", "negative", "neutral"})


class Prediction(TypedDict):
    sentence: str
    label: str
    probabilities: dict[str, float]


class FinBertAnalyzer:
    """Load once, then call predict for each English financial sentence.

    Missing files are downloaded to the Hugging Face cache unless
    local_files_only=True. No GPU is used even if one is available.
    """

    def __init__(self, *, local_files_only: bool = False) -> None:
        options = {
            "revision": MODEL_REVISION,
            "local_files_only": local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, use_safetensors=True, **options
        )
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.labels = {
            int(index): label.lower()
            for index, label in self.model.config.id2label.items()
        }
        if set(self.labels) != {0, 1, 2} or set(self.labels.values()) != SENTIMENT_LABELS:
            raise ValueError("FinBERT must define positive, negative and neutral labels.")

    def predict(self, sentence: str) -> Prediction:
        """Return the winning label and all softmax probabilities.

        Blank input raises ValueError; non-string input raises TypeError.
        Inputs longer than the model context are truncated to its token limit.
        """
        if not isinstance(sentence, str):
            raise TypeError("sentence must be a string")
        if not sentence.strip():
            raise ValueError("sentence must not be empty or whitespace-only")

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            scores = torch.softmax(logits, dim=-1)[0].tolist()
        return self._prediction(sentence, scores)

    def _prediction(self, sentence: str, scores: list[float]) -> Prediction:
        probabilities = {self.labels[i]: score for i, score in enumerate(scores)}
        return {
            "sentence": sentence,
            "label": max(probabilities, key=probabilities.__getitem__),
            "probabilities": probabilities,
        }

    def predict_batch(self, sentences: list[str]) -> list[Prediction]:
        """Classify one CPU batch without truncation; reject overlong inputs.

        Unlike the original single-sentence interface, this method never discards
        tokens. The caller controls batch size; padding is only to this batch's
        longest sentence, including special tokens.
        """
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list of strings")
        for sentence in sentences:
            if not isinstance(sentence, str):
                raise TypeError("each sentence must be a string")
            if not sentence.strip():
                raise ValueError("sentences must not be empty or whitespace-only")
        if not sentences:
            return []
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=False
        )
        if inputs["input_ids"].shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("Batch contains a sentence exceeding the model token limit; no truncation applied.")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        self.model.eval()
        with torch.inference_mode():
            scores = torch.softmax(self.model(**inputs).logits, dim=-1).tolist()
        if len(scores) != len(sentences):
            raise ValueError("Model returned an unexpected number of predictions.")
        return [self._prediction(sentence, row) for sentence, row in zip(sentences, scores)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-files-only", action="store_true", help="Use cached files without network access."
    )
    args = parser.parse_args()
    analyzer = FinBertAnalyzer(local_files_only=args.local_files_only)
    sentences = [
        "The company reported strong revenue growth.",
        "Operating expenses increased significantly.",
        "The board approved the annual financial statements.",
        "Profit margins improved compared with the previous quarter.",
        "The company expects challenging market conditions.",
    ]
    print(f"Model: {MODEL_ID} | Device: {analyzer.device}")
    for sentence in sentences:
        result = analyzer.predict(sentence)
        print(f"\n{result['sentence']}\nSentiment: {result['label']}")
        print(" | ".join(
            f"{label}: {result['probabilities'][label]:.6f}"
            for label in ("positive", "negative", "neutral")
        ))


if __name__ == "__main__":
    main()
