# app/services/vector_db_manager.py

from typing import List, Tuple

import faiss
import numpy as np
import torch
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from utils.logger import logger


class VectorDBManager:
    def __init__(self):
        logger.info("Initializing VectorDBManager...")
        self.index = None
        self.retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.metadata = []
        logger.info("VectorDBManager initialized successfully")

    def __del__(self):
        logger.info("Cleaning up VectorDBManager resources...")
        if hasattr(self, "index") and self.index is not None:
            self.index = None
        if hasattr(self, "retrieval_model"):
            del self.retrieval_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("VectorDBManager cleanup completed")

    def compute_embeddings(
        self, documents: List[Document]
    ) -> Tuple[np.ndarray, List[dict]]:
        logger.info(f"Computing embeddings for {len(documents)} documents...")
        texts = [doc.page_content for doc in documents]
        embeddings = self.retrieval_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
        logger.info(f"Successfully computed embeddings of shape {embeddings.shape}")

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
        logger.info(
            f"Building FAISS index for embeddings of shape {embeddings.shape}..."
        )
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index
        logger.info("FAISS index built successfully")
        return index

    def search_index(self, query: str, top_k: int = 3) -> List[dict]:
        if self.index is None:
            logger.error("FAISS index not initialized")
            raise ValueError(
                "FAISS index not initialized. Call build_faiss_index() first."
            )

        logger.info(f"Searching index for query: {query[:50]}...")
        query_embedding = self.retrieval_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        logger.info(f"Found {len(indices[0])} results")

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
