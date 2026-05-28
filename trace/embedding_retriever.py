"""Lightweight embedding retriever used by ``RobustAgenticMemorySystem``.

Provides:
  - ``simple_tokenize`` — NLTK word_tokenize wrapper (used by BM25 helpers in
    ``memory_layer_robust``).
  - ``SimpleEmbeddingRetriever`` — SentenceTransformer-backed dense retriever
    with on-disk pickle + .npy persistence.

Originally extracted from the upstream A-Mem ``memory_layer`` module (which
included a large amount of unused-by-TRACE classes); this module keeps only
the symbols the release pipeline imports. The release uses the
``all-MiniLM-L6-v2`` SentenceTransformer model exclusively.
"""

import os
from typing import Dict, List

import numpy as np
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def simple_tokenize(text):
    return word_tokenize(text)


class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the simple embedding retriever."""
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index

    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5):
        """Search for similar documents using cosine similarity.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Top-k indices into the corpus, sorted by similarity descending.
        """
        if not self.corpus:
            return []
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk."""
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)

        import pickle
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids,
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk."""
        if os.path.exists(retriever_cache_embeddings_file):
            self.embeddings = np.load(retriever_cache_embeddings_file)

        if os.path.exists(retriever_cache_file):
            import pickle
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']

        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """Build a retriever from an existing memory dict."""
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)

        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever
