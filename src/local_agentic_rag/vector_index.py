from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    score: float
    rank: int


class VectorIndex:
    def __init__(self, *, backend: str, index_path: Path, metadata_path: Path) -> None:
        self.backend = backend
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunk_ids: list[str] = []
        self.embedding_model: str | None = None
        self._matrix: np.ndarray | None = None
        self._faiss_index = None

    def build(self, vectors: list[tuple[str, list[float]]], *, embedding_model: str) -> None:
        self.chunk_ids = [chunk_id for chunk_id, _ in vectors]
        self.embedding_model = embedding_model
        if not vectors:
            self._matrix = np.empty((0, 0), dtype=np.float32)
            self._save_metadata(dim=0)
            return
        matrix = np.asarray([vector for _, vector in vectors], dtype=np.float32)
        matrix = _normalize(matrix)
        if self.backend == "faiss":
            try:
                import faiss
            except ImportError as exc:  # pragma: no cover - depends on local environment
                raise RuntimeError(
                    "FAISS backend requested but faiss-cpu is not installed. "
                    "Install the faiss extra or switch retrieval.vector_backend to 'numpy'."
                ) from exc
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            faiss.write_index(index, str(self.index_path))
            self._faiss_index = index
        else:
            with self.index_path.open("wb") as handle:
                np.save(handle, matrix)
            self._matrix = matrix
        self._save_metadata(dim=matrix.shape[1])

    def load(self) -> None:
        if not self.metadata_path.exists():
            self.chunk_ids = []
            self.embedding_model = None
            self._matrix = None
            self._faiss_index = None
            return
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.chunk_ids = metadata.get("chunk_ids", [])
        self.embedding_model = metadata.get("embedding_model")
        if metadata.get("dim", 0) == 0 or not self.index_path.exists():
            self._matrix = np.empty((0, 0), dtype=np.float32)
            self._faiss_index = None
            return
        if self.backend == "faiss":
            try:
                import faiss
            except ImportError as exc:  # pragma: no cover - depends on local environment
                raise RuntimeError("FAISS backend requested but faiss-cpu is not installed.") from exc
            self._faiss_index = faiss.read_index(str(self.index_path))
        else:
            with self.index_path.open("rb") as handle:
                self._matrix = np.load(handle)

    def search(self, query_vector: list[float], *, limit: int) -> list[VectorHit]:
        if not self.chunk_ids:
            return []
        vector = _normalize(np.asarray([query_vector], dtype=np.float32))
        if self.backend == "faiss":
            if self._faiss_index is None:
                self.load()
            distances, indices = self._faiss_index.search(vector, limit)
            return [
                VectorHit(chunk_id=self.chunk_ids[index], score=float(score), rank=rank)
                for rank, (score, index) in enumerate(zip(distances[0], indices[0]), start=1)
                if index != -1
            ]
        if self._matrix is None:
            self.load()
        assert self._matrix is not None
        if self._matrix.size == 0:
            return []
        scores = self._matrix @ vector[0]
        ranked_indices = np.argsort(scores)[::-1][:limit]
        return [
            VectorHit(chunk_id=self.chunk_ids[index], score=float(scores[index]), rank=rank)
            for rank, index in enumerate(ranked_indices, start=1)
        ]

    def _save_metadata(self, *, dim: int) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "chunk_ids": self.chunk_ids,
            "embedding_model": self.embedding_model,
            "dim": dim,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
