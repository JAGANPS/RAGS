"""
Base RAG Engine interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source_file: str
    category: str


@dataclass
class RAGResponse:
    """Complete RAG response with metrics"""
    answer: str
    sources: List[RetrievalResult]
    rag_type: str
    query: str
    latency: Dict[str, float]  # Breakdown of latencies
    metadata: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "text": s.text[:200] + "..." if len(s.text) > 200 else s.text,
                    "score": round(s.score, 4),
                    "source_file": s.source_file,
                    "category": s.category
                }
                for s in self.sources
            ],
            "rag_type": self.rag_type,
            "query": self.query,
            "latency": {k: round(v, 2) for k, v in self.latency.items()},
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class BaseRAGEngine(ABC):
    """Abstract base class for RAG engines"""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the RAG engine"""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents to the knowledge base"""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """Full RAG pipeline: retrieve + generate"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized
