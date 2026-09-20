"""KURE-v2(ColBERT 계열 late-interaction) 기반 재순위(rerank) 모듈. 2단계 전용.

1단계(Chroma)에서 넓게 추린 후보들을 받아서, 토큰 단위 MaxSim으로 정밀 재정렬한다.
"""

from __future__ import annotations

import os

from sentence_transformers import MultiVectorEncoder

from src.rag.mps_guard import MPS_INFERENCE_LOCK

_MODEL_NAME = os.getenv("RERANK_MODEL", "nlpai-lab/KURE-v2")
_model: MultiVectorEncoder | None = None


def _get_model() -> MultiVectorEncoder:
    global _model
    with MPS_INFERENCE_LOCK:
        if _model is None:
            # embeddings.py와 동일한 이유로 CPU에 고정 (MPS 크래시 근본 회피).
            _model = MultiVectorEncoder(_MODEL_NAME, device="cpu")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """candidates: retriever.search()가 반환한 [{"text": ..., "source": ..., ...}, ...] 형태.

    반환된 각 dict엔 "rerank_score"(MaxSim 점수, 높을수록 관련도 높음)가 추가된다.
    """
    if not candidates:
        return []

    model = _get_model()
    texts = [c["text"] for c in candidates]

    with MPS_INFERENCE_LOCK:
        query_emb = model.encode([query])
        doc_embs = model.encode(texts)
        scores = model.similarity(query_emb, doc_embs)[0]

    ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)

    result = []
    for candidate, score in ranked[:top_k]:
        item = dict(candidate)
        item["rerank_score"] = float(score)
        result.append(item)
    return result