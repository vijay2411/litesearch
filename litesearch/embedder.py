from __future__ import annotations

import math
import struct
from typing import Iterable, List, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]: ...

    def embed_query(self, query: str) -> List[float]: ...


def normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def to_blob(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def from_blob(blob: bytes, dim: int) -> List[float]:
    return list(struct.unpack(f"{dim}f", blob))


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "embeddinggemma",
        ollama_url: str = "http://localhost:11434",
        dims: int = 768,
    ):
        self._model = model
        self._url = ollama_url
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def _client(self):
        import ollama
        return ollama.Client(host=self._url)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        client = self._client()
        out: List[List[float]] = []
        for text in texts:
            prompt = f"title: none | text: {text}"
            resp = client.embeddings(
                model=self._model, prompt=prompt, keep_alive="30m"
            )
            out.append(normalize(list(resp["embedding"])))
        return out

    def embed_query(self, query: str) -> List[float]:
        client = self._client()
        prompt = f"task: search result | query: {query}"
        resp = client.embeddings(
            model=self._model, prompt=prompt, keep_alive="30m"
        )
        return normalize(list(resp["embedding"]))
