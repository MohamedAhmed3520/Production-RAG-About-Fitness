from __future__ import annotations

from typing import List, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class BGEReranker:
    """Use the BAAI/bge-reranker-v2-m3 reranker to prioritize retrieved chunks."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

    def rerank(self, query: str, candidates: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
        if not candidates:
            return []
        pairs = [(query, c[0].get("text", "")) for c in candidates]
        results = self.pipe(pairs, batch_size=8)
        scored: List[Tuple[dict, float]] = []
        for (candidate, _), result in zip(candidates, results):
            score = float(result["score"])
            scored.append((candidate[0], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:5]
