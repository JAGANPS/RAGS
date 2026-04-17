"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class RAGType(str, Enum):
    STANDARD = "standard"
    VECTORLESS = "vectorless"
    AGENTIC = "agentic"
    AUTO = "auto"  # Let the system choose


class CategoryType(str, Enum):
    SAM = "sam"
    ITAM = "itam"
    ITV = "itv"
    SRE = "sre"
    ALL = "all"


# Request Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to process", min_length=1)
    rag_type: RAGType = Field(default=RAGType.AUTO, description="RAG engine to use")
    category: CategoryType = Field(default=CategoryType.ALL, description="Filter by category")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    enable_reflection: bool = Field(default=True, description="Enable reflection for agentic RAG")


class DocumentUploadRequest(BaseModel):
    category: CategoryType = Field(..., description="Document category")


class CompareRequest(BaseModel):
    query: str = Field(..., description="Query to compare across RAG types")
    top_k: int = Field(default=5, ge=1, le=20)
    category: CategoryType = Field(default=CategoryType.ALL)


# Response Models
class SourceInfo(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_file: str
    category: str
    metadata: Dict[str, Any] = {}


class LatencyInfo(BaseModel):
    retrieval_ms: float = 0
    generation_ms: float = 0
    routing_ms: Optional[float] = None
    synthesis_ms: Optional[float] = None
    reflection_ms: Optional[float] = None
    total_ms: float


class QueryResponse(BaseModel):
    success: bool
    answer: str
    sources: List[SourceInfo]
    rag_type: str
    query: str
    latency: LatencyInfo
    metadata: Dict[str, Any] = {}
    timestamp: datetime


class CompareResponse(BaseModel):
    query: str
    results: Dict[str, QueryResponse]
    comparison: Dict[str, Any]
    recommendation: str


class EngineStats(BaseModel):
    engine: str
    status: str
    details: Dict[str, Any]


class SystemStats(BaseModel):
    engines: Dict[str, EngineStats]
    total_documents: int
    categories: Dict[str, int]
    latency_metrics: Dict[str, Any]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    category: str
    file_type: str
    chunks_count: int
    status: str


class UploadResponse(BaseModel):
    success: bool
    documents_processed: int
    documents: List[DocumentInfo]
    errors: List[str] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    engines: Dict[str, str]
    aws_region: str
    model_id: str
