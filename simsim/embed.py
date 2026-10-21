"""Text embeddings for semantic memory retrieval.

    emb = OpenAIEmbedder("text-embedding-3-small")                       # OpenAI
    emb = OpenAIEmbedder("BAAI/bge-m3", base_url="http://gpu012:8002/v1")  # vLLM --task embed
    emb = HashEmbedder()                                                  # no model; bag of words

    memory = ScoredMemory(embedder=emb)
"""

from __future__ import annotations

import hashlib
import math
import re
import threading


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Embedder:
    """Base class: `embed(texts) -> list of vectors`. Results are cached per text."""

    def __init__(self):
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self._embed(missing)
            with self._lock:
                self._cache.update(zip(missing, vectors, strict=True))
        with self._lock:
            return [self._cache[t] for t in texts]

    def __call__(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity mapped to [0, 1]."""
        va, vb = self.embed([a, b])
        return (cosine(va, vb) + 1) / 2

    def _embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


class OpenAIEmbedder(Embedder):
    """Any OpenAI-compatible /embeddings endpoint (OpenAI, vLLM with an embedding model, Ollama)."""

    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None,
                 api_key: str | None = None, batch_size: int = 128):
        super().__init__()
        self.model, self.base_url, self.api_key, self.batch_size = model, base_url, api_key, batch_size
        self._connect()

    def _connect(self):
        import os

        from openai import OpenAI

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def _embed(self, texts):
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            out += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        return out

    def __setstate__(self, state):
        super().__setstate__(state)
        self._connect()


class HashEmbedder(Embedder):
    """Deterministic bag-of-words vectors (feature hashing). No model, no network.

    Good for tests and as a cheap baseline; it captures word overlap, not meaning.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim

    def _embed(self, texts):
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9']+", text.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
            out.append(v)
        return out
