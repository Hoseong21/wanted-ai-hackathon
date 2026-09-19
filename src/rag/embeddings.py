"""임베딩 함수.

기본값은 KURE-v1(nlpai-lab, 한국어 특화 dense encoder)이다. Chroma는 문서 1개당 벡터 1개만
지원하는 구조라, 1단계(Chroma 검색)에는 이런 단일 벡터(dense) 임베딩이 맞다. 한국어 문서를
다루기 때문에 Chroma 기본 내장 임베딩(all-MiniLM-L6-v2, 영어 전용)이나 OpenAI 대신
KURE-v1을 골랐다 (MTEB-ko-retrieval 벤치마크 기준 한국어 검색 성능이 더 높고, 무료/로컬 실행).

- KUREEmbeddingFunction: KURE-v1로 텍스트를 1024차원 dense 벡터로 인코딩한다. 기본 백엔드.
- HashingEmbeddingFunction: 네트워크/모델 다운로드가 전혀 안 되는 환경(오프라인 개발, CI 등)에서
  파이프라인만 빠르게 검증하고 싶을 때 쓰는 대체용. 텍스트를 토큰화 -> 해시로 고정 차원 벡터에
  매핑하는 bag-of-hashed-words 방식이라 의미 기반 검색은 아니고 정확도가 낮다.
- OpenAI 임베딩(text-embedding-3-small)도 옵션으로 남겨뒀다. .env에서
  EMBEDDING_BACKEND=openai + OPENAI_API_KEY 설정하면 전환된다.

.env의 EMBEDDING_BACKEND 값으로 전환: kure(기본) | hash | openai
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter

from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

_TOKEN_RE = re.compile(r"[\w가-힣]+")
_DIM = 256


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_embed(text: str, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    counts = Counter(_tokenize(text))
    for token, count in counts.items():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += count
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbeddingFunction(EmbeddingFunction):
    """네트워크/모델 다운로드 없이 동작하는 로컬 해싱 기반 임베딩 함수 (오프라인 대체용)."""

    def __call__(self, input: Documents) -> Embeddings:
        return [_hash_embed(text) for text in input]


_kure_model: SentenceTransformer | None = None


def _get_kure_model() -> SentenceTransformer:
    global _kure_model
    if _kure_model is None:
        _kure_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "nlpai-lab/KURE-v1"))
    return _kure_model


class KUREEmbeddingFunction(EmbeddingFunction):
    """KURE-v1(한국어 특화 dense encoder) 기반 임베딩 함수. 1단계(Chroma) 검색용, 기본값."""

    def __init__(self):
        self._model = _get_kure_model()

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), normalize_embeddings=True).tolist()


def get_embedding_function():
    backend = os.getenv("EMBEDDING_BACKEND", "kure")
    if backend == "kure":
        return KUREEmbeddingFunction()
    if backend == "openai":
        from chromadb.utils import embedding_functions

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
    return HashingEmbeddingFunction()