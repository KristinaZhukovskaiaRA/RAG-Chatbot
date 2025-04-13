from typing import List, Tuple

import faiss
import numpy as np
import torch
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class VectorDBManager:
    def __init__(self):
        self.index = None
        self.retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.metadata = []

    def __del__(self):
        """Cleanup method to ensure resources are properly released"""
        if hasattr(self, "index") and self.index is not None:
            self.index = None
        if hasattr(self, "retrieval_model"):
            del self.retrieval_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def compute_embeddings(
        self, documents: List[Document]
    ) -> Tuple[np.ndarray, List[dict]]:
        texts = [doc.page_content for doc in documents]
        embeddings = self.retrieval_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        metadata = []
        for doc in documents:
            meta = {
                "source": doc.metadata.get("source_file")
                or doc.metadata.get("source_url", "unknown"),
                "text_snippet": doc.page_content[:200] + "...",
            }
            metadata.append(meta)

        return embeddings, metadata

    def build_faiss_index(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index
        return index

    def search_index(self, query: str, top_k: int = 3) -> List[dict]:
        if self.index is None:
            raise ValueError(
                "FAISS index not initialized. Call build_faiss_index() first."
            )

        query_embedding = self.retrieval_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = {
                    "rank": i + 1,
                    "source": self.metadata[idx]["source"],
                    "text_snippet": self.metadata[idx]["text_snippet"],
                    "distance": float(distances[0][i]),
                }
                results.append(result)

        return results
