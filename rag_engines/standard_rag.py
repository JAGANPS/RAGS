"""
Standard Vector-based RAG Engine using ChromaDB
"""
import time
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .base import BaseRAGEngine, RetrievalResult, RAGResponse
from utils.bedrock_client import get_bedrock_client
from utils.latency_tracker import get_latency_tracker, track_latency
from config.settings import settings


class StandardRAGEngine(BaseRAGEngine):
    """
    Standard RAG implementation using:
    - ChromaDB for vector storage
    - Sentence Transformers for embeddings
    - AWS Bedrock Claude for generation
    """

    def __init__(self):
        super().__init__("standard_rag")
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.bedrock = None
        self.tracker = get_latency_tracker()

    async def initialize(self) -> None:
        """Initialize embedding model and vector store"""
        if self._initialized:
            return

        # Initialize embedding model
        with track_latency("standard_rag.init_embeddings"):
            self.embedding_model = SentenceTransformer(settings.embedding_model)

        # Initialize ChromaDB
        with track_latency("standard_rag.init_chromadb"):
            persist_dir = Path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            self.collection = self.chroma_client.get_or_create_collection(
                name="flexera_knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )

        # Initialize Bedrock client
        self.bedrock = get_bedrock_client()
        self._initialized = True

    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add document chunks to vector store"""
        if not self._initialized:
            await self.initialize()

        added_count = 0

        with track_latency("standard_rag.add_documents", {"count": len(documents)}):
            for doc in documents:
                chunks = doc.get("chunks", [])
                if not chunks:
                    continue

                # Batch process embeddings
                texts = [chunk["text"] for chunk in chunks]
                embeddings = self.embedding_model.encode(texts).tolist()

                # Prepare for ChromaDB
                ids = [chunk["chunk_id"] for chunk in chunks]
                metadatas = [
                    {
                        "doc_id": chunk["metadata"]["doc_id"],
                        "filename": chunk["metadata"]["filename"],
                        "category": chunk["metadata"]["category"],
                        "chunk_index": chunk["metadata"]["chunk_index"]
                    }
                    for chunk in chunks
                ]

                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

                added_count += len(chunks)

        return added_count

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks using vector similarity"""
        if not self._initialized:
            await self.initialize()

        results = []

        with track_latency("standard_rag.retrieve", {"query_length": len(query), "top_k": top_k}):
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Build where clause for filters
            where = None
            if filters:
                where = {}
                if "category" in filters:
                    where["category"] = filters["category"]
                if "filename" in filters:
                    where["filename"] = filters["filename"]

            # Query ChromaDB
            query_result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )

            # Convert to RetrievalResult
            if query_result["documents"] and query_result["documents"][0]:
                for i, doc in enumerate(query_result["documents"][0]):
                    metadata = query_result["metadatas"][0][i]
                    distance = query_result["distances"][0][i]

                    # Convert distance to similarity score (cosine)
                    score = 1 - distance

                    results.append(RetrievalResult(
                        chunk_id=query_result["ids"][0][i],
                        text=doc,
                        score=score,
                        metadata=metadata,
                        source_file=metadata.get("filename", "unknown"),
                        category=metadata.get("category", "general")
                    ))

        return results

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """Full RAG pipeline: retrieve + generate"""
        latency = {}

        # Retrieval phase
        start = time.perf_counter()
        sources = await self.retrieve(query, top_k, filters)
        latency["retrieval_ms"] = (time.perf_counter() - start) * 1000

        # Build context
        context_parts = []
        for i, source in enumerate(sources):
            context_parts.append(
                f"[Source {i+1}: {source.source_file} ({source.category})] "
                f"(Relevance: {source.score:.2%})\n{source.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Generation phase
        start = time.perf_counter()
        with track_latency("standard_rag.generate"):
            response = self.bedrock.invoke_with_context(query, context)
        latency["generation_ms"] = (time.perf_counter() - start) * 1000
        latency["total_ms"] = latency["retrieval_ms"] + latency["generation_ms"]

        return RAGResponse(
            answer=response.content,
            sources=sources,
            rag_type="standard_rag",
            query=query,
            latency=latency,
            metadata={
                "model": response.model_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "chunks_retrieved": len(sources)
            }
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "engine": self.name,
            "total_documents": self.collection.count(),
            "embedding_model": settings.embedding_model,
            "vector_dimensions": self.embedding_model.get_sentence_embedding_dimension(),
            "storage_path": settings.chroma_persist_dir
        }
