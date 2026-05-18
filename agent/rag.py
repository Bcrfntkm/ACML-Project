"""RAG (Retrieval-Augmented Generation) module for the chemical annotation agent.

Provides:
* :class:`RAGRetriever` — indexes documents, embeds chunks using
  ``sentence-transformers`` (preferred) or a TF-IDF fallback, persists the
  index to disk, and retrieves the top-k most relevant chunks for a query.
* :func:`rag_retrieve_node` — LangGraph node function that wraps
  :class:`RAGRetriever` and integrates with :class:`~state.AgentState`.

CLI usage::

    python -m Science.agent.rag add --file path/to/doc.txt
    python -m Science.agent.rag add --text "Some chemical text..."
    python -m Science.agent.rag retrieve --query "aspirin synthesis" --top-k 5
    python -m Science.agent.rag clear
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .config import settings as _default_settings
from .state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional embedding backends
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False

try:  # pragma: no cover
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


DEFAULT_INDEX_DIR: str = "Science/rag_index"
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split *text* into overlapping chunks of approximately *chunk_size* words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """Retrieval-Augmented Generation index over text chunks.

    Parameters
    ----------
    index_dir:
        Directory where the index files (``embeddings.npz``, ``chunks.json``)
        are stored.
    model_name:
        Name of the sentence-transformers model to use.  Ignored when the
        ``sentence-transformers`` package is not available; in that case a
        TF-IDF fallback is used.
    chunk_size:
        Approximate number of words per chunk.
    chunk_overlap:
        Number of overlapping words between adjacent chunks.
    """

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.index_dir: Path = Path(index_dir)
        self.model_name: str = model_name
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D), L2-normalised

        # Backend selection
        self._backend: str
        self._st_model = None
        self._tfidf: Optional["TfidfVectorizer"] = None
        if _HAS_ST:
            self._backend = "sentence-transformers"
        elif _HAS_SKLEARN:
            self._backend = "tfidf"
        else:
            raise ImportError(
                "Neither `sentence-transformers` nor `scikit-learn` is installed. "
                "Install at least one to use RAGRetriever."
            )

        # Auto-load existing index if present
        if self._index_files_exist():
            try:
                self.load()
            except Exception as exc:  # pragma: no cover
                logger.warning("RAGRetriever: failed to auto-load index: %s", exc)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def _emb_path(self) -> Path:
        return self.index_dir / "embeddings.npz"

    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def _tfidf_path(self) -> Path:
        return self.index_dir / "tfidf.npz"

    def _index_files_exist(self) -> bool:
        return self._chunks_path.exists() and (
            self._emb_path.exists() or self._tfidf_path.exists()
        )

    # ------------------------------------------------------------------
    # Backend / embedding helpers
    # ------------------------------------------------------------------

    def _load_st_model(self) -> None:
        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model_name)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def _embed_with_st(self, texts: list[str]) -> np.ndarray:
        self._load_st_model()
        vecs = self._st_model.encode(  # type: ignore[union-attr]
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return self._l2_normalize(np.asarray(vecs, dtype=np.float32))

    def _refit_tfidf(self) -> None:
        """(Re)fit the TF-IDF vectorizer over all current chunks."""
        if not self.chunks:
            self._tfidf = None
            self.embeddings = None
            return
        self._tfidf = TfidfVectorizer()
        mat = self._tfidf.fit_transform(self.chunks).toarray().astype(np.float32)
        self.embeddings = self._l2_normalize(mat)

    def _embed_query(self, query: str) -> np.ndarray:
        if self._backend == "sentence-transformers":
            return self._embed_with_st([query])[0]
        if self._tfidf is None:
            raise RuntimeError("TF-IDF vectorizer is not initialised.")
        vec = self._tfidf.transform([query]).toarray().astype(np.float32)
        return self._l2_normalize(vec)[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: list[str],
        source_names: Optional[list[str]] = None,
    ) -> int:
        """Add raw texts to the index.  Returns number of chunks added."""
        if source_names is not None and len(source_names) != len(texts):
            raise ValueError("source_names length must match texts length")

        new_chunks: list[str] = []
        new_meta: list[dict] = []
        for i, text in enumerate(texts):
            src = source_names[i] if source_names else f"text_{len(self.chunks) + i}"
            pieces = _chunk_text(text, self.chunk_size, self.chunk_overlap)
            for j, piece in enumerate(pieces):
                new_chunks.append(piece)
                new_meta.append({"source": src, "chunk_id": j})

        if not new_chunks:
            return 0

        if self._backend == "sentence-transformers":
            new_emb = self._embed_with_st(new_chunks)
            if self.embeddings is None or self.embeddings.size == 0:
                self.embeddings = new_emb
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb])
            self.chunks.extend(new_chunks)
            self.metadata.extend(new_meta)
        else:
            self.chunks.extend(new_chunks)
            self.metadata.extend(new_meta)
            self._refit_tfidf()

        return len(new_chunks)

    def add_file(self, path: str) -> int:
        """Read a ``.txt`` file and add its content to the index."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if p.suffix.lower() != ".txt":
            raise ValueError(f"Only .txt files are supported, got: {p.suffix}")
        text = p.read_text(encoding="utf-8")
        return self.add_texts([text], source_names=[str(p)])

    def save(self) -> None:
        """Persist the index to disk."""
        if not self.chunks:
            raise RuntimeError("Cannot save an empty index. Add documents first.")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        with self._chunks_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "backend": self._backend,
                    "model_name": self.model_name,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunks": self.chunks,
                    "metadata": self.metadata,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

        if self._backend == "sentence-transformers":
            assert self.embeddings is not None
            np.savez_compressed(self._emb_path, embeddings=self.embeddings)
            if self._tfidf_path.exists():
                self._tfidf_path.unlink()
        else:
            assert self.embeddings is not None
            np.savez_compressed(self._tfidf_path, embeddings=self.embeddings)
            if self._emb_path.exists():
                self._emb_path.unlink()

    def load(self) -> None:
        """Load the index from disk."""
        if not self._chunks_path.exists():
            raise FileNotFoundError(f"No chunks file at {self._chunks_path}")
        with self._chunks_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.chunks = list(data.get("chunks", []))
        self.metadata = list(data.get("metadata", []))
        saved_backend = data.get("backend", self._backend)
        self.chunk_size = int(data.get("chunk_size", self.chunk_size))
        self.chunk_overlap = int(data.get("chunk_overlap", self.chunk_overlap))
        saved_model_name = data.get("model_name", self.model_name)

        if saved_backend == "sentence-transformers" and self._backend == "sentence-transformers":
            self.model_name = saved_model_name
            if not self._emb_path.exists():
                raise FileNotFoundError(f"Embeddings file missing at {self._emb_path}")
            with np.load(self._emb_path) as npz:
                self.embeddings = npz["embeddings"].astype(np.float32)
        elif saved_backend == "tfidf" and self._backend == "tfidf":
            self._refit_tfidf()
        else:
            logger.warning(
                "RAGRetriever: backend mismatch (saved=%s, current=%s); "
                "re-embedding chunks with current backend.",
                saved_backend,
                self._backend,
            )
            if self._backend == "sentence-transformers":
                self.embeddings = self._embed_with_st(self.chunks) if self.chunks else None
            else:
                self._refit_tfidf()

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return the top-k most relevant chunks for *query*."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not self.chunks or self.embeddings is None or self.embeddings.size == 0:
            raise RuntimeError(
                "Index is empty. Add documents with `add_texts`/`add_file` "
                "before calling `retrieve`."
            )
        if top_k <= 0:
            return []

        q_vec = self._embed_query(query)
        sims = self.embeddings @ q_vec
        k = min(top_k, len(self.chunks))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [self.chunks[i] for i in top_idx]

    def clear(self) -> None:
        """Wipe the index in memory and on disk."""
        self.chunks = []
        self.metadata = []
        self.embeddings = None
        self._tfidf = None
        for p in (self._emb_path, self._chunks_path, self._tfidf_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:  # pragma: no cover
                    pass

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RAGRetriever(backend={self._backend!r}, chunks={len(self.chunks)}, "
            f"index_dir={str(self.index_dir)!r})"
        )


# ---------------------------------------------------------------------------
# LangGraph node function (also used by nodes.py)
# ---------------------------------------------------------------------------

def rag_retrieve_node(state: AgentState) -> dict:
    """LangGraph node: retrieve relevant chunks from the RAG index.

    This is the canonical implementation; :mod:`nodes` imports it from here.

    Returns
    -------
    dict
        ``{"rag_context": list[str]}``
    """
    if not _default_settings.rag_enabled:
        logger.debug("rag_retrieve_node: RAG disabled — skipping.")
        return {"rag_context": []}

    index_dir = Path(_default_settings.rag_index_dir)
    chunks_file = index_dir / "chunks.json"
    if not chunks_file.exists():
        logger.debug("rag_retrieve_node: index not found at %s — skipping.", index_dir)
        return {"rag_context": []}

    try:
        retriever = RAGRetriever(index_dir=str(index_dir))
        if len(retriever) == 0:
            return {"rag_context": []}

        query: str = state.get("query", "")
        top_k: int = _default_settings.rag_top_k
        chunks: list[str] = retriever.retrieve(query, top_k=top_k)
        logger.info("rag_retrieve_node: retrieved %d chunks.", len(chunks))
        return {"rag_context": chunks}

    except Exception as exc:  # pragma: no cover
        logger.warning("rag_retrieve_node failed (non-fatal): %s", exc)
        return {"rag_context": []}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG retriever for the chemical annotation agent."
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Directory for the index (default: {DEFAULT_INDEX_DIR}).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Sentence-transformers model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Word overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add a text or file to the index.")
    add_grp = add_p.add_mutually_exclusive_group(required=True)
    add_grp.add_argument("--file", help="Path to a .txt file to ingest.")
    add_grp.add_argument("--text", help="Raw text to ingest.")

    ret_p = sub.add_parser("retrieve", help="Retrieve top-k chunks for a query.")
    ret_p.add_argument("--query", required=True, help="Query string.")
    ret_p.add_argument("--top-k", type=int, default=5, help="Number of chunks (default: 5).")

    sub.add_parser("clear", help="Wipe the index.")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    retriever = RAGRetriever(
        index_dir=args.index_dir,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.command == "add":
        if args.file:
            n = retriever.add_file(args.file)
            print(f"Added {n} chunk(s) from file: {args.file}")
        else:
            n = retriever.add_texts([args.text])
            print(f"Added {n} chunk(s) from --text input.")
        retriever.save()
        print(f"Index saved to: {retriever.index_dir}")
        return 0

    if args.command == "retrieve":
        try:
            results = retriever.retrieve(args.query, top_k=args.top_k)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if not results:
            print("(no results)")
            return 0
        for i, chunk in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(chunk)
            print()
        return 0

    if args.command == "clear":
        retriever.clear()
        print(f"Cleared index at: {retriever.index_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
