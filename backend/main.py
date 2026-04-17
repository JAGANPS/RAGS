"""
Multi-RAG System - FastAPI Backend
Supports Standard RAG, Vectorless RAG, and Agentic RAG
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import (
    QueryRequest, QueryResponse, CompareRequest, CompareResponse,
    DocumentUploadRequest, UploadResponse, DocumentInfo,
    EngineStats, SystemStats, HealthResponse, LatencyInfo, SourceInfo,
    RAGType, CategoryType
)
from rag_engines import StandardRAGEngine, VectorlessRAGEngine, AgenticRAGEngine
from utils import DocumentProcessor, get_latency_tracker, track_latency
from config import settings


# Global RAG engines
engines: Dict[str, Any] = {}
document_processor = DocumentProcessor(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engines on startup"""
    print("Initializing RAG engines...")

    # Initialize all engines
    print("Creating engine instances...")
    engines["standard"] = StandardRAGEngine()
    engines["vectorless"] = VectorlessRAGEngine()
    engines["agentic"] = AgenticRAGEngine()

    # Initialize in parallel
    print("Starting parallel initialization...")
    await asyncio.gather(
        engines["standard"].initialize(),
        engines["vectorless"].initialize(),
        engines["agentic"].initialize()
    )

    print("RAG engines initialized successfully!")

    # Skip document loading on startup for faster boot
    # Documents can be uploaded via the UI
    print("Skipping auto-load of documents (use UI to upload)")
    # Uncomment the following to auto-load documents:
    # await load_existing_documents()

    print("Server ready!")

    yield

    # Cleanup
    print("Shutting down RAG engines...")


async def load_existing_documents():
    """Load documents from the documents directory"""
    docs_dir = Path(settings.documents_dir)
    if not docs_dir.exists():
        print("No documents directory found")
        return

    for category in ["sam", "itam", "itv", "sre"]:
        category_dir = docs_dir / category
        if category_dir.exists():
            try:
                documents = document_processor.process_directory(str(category_dir), category)
                if documents:
                    doc_dicts = [
                        {
                            "id": doc.id,
                            "filename": doc.filename,
                            "content": doc.content,
                            "category": doc.category,
                            "chunks": doc.chunks
                        }
                        for doc in documents
                    ]
                    # Load to standard RAG (always works)
                    await engines["standard"].add_documents(doc_dicts)
                    print(f"Loaded {len(documents)} documents from {category} to standard RAG")

                    # Try to load to vectorless RAG (may need Bedrock)
                    try:
                        await engines["vectorless"].add_documents(doc_dicts)
                        print(f"Loaded {len(documents)} documents from {category} to vectorless RAG")
                    except Exception as e:
                        print(f"Warning: Could not load to vectorless RAG (may need AWS credentials): {e}")
            except Exception as e:
                print(f"Error processing {category}: {e}")


# Create FastAPI app
app = FastAPI(
    title="Multi-RAG System",
    description="Advanced RAG system with Standard, Vectorless, and Agentic capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")
templates = Jinja2Templates(directory=str(frontend_path / "templates"))


# ============== Routes ==============

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        engines={
            "standard": "ready" if engines.get("standard") and engines["standard"].is_initialized else "not_ready",
            "vectorless": "ready" if engines.get("vectorless") and engines["vectorless"].is_initialized else "not_ready",
            "agentic": "ready" if engines.get("agentic") and engines["agentic"].is_initialized else "not_ready"
        },
        aws_region=settings.aws_region,
        model_id=settings.bedrock_model_id
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using the specified RAG engine"""
    tracker = get_latency_tracker()

    with track_latency("api.query", {"rag_type": request.rag_type.value}):
        # Select engine
        if request.rag_type == RAGType.AUTO:
            # Use agentic for complex queries, standard for simple ones
            engine = engines["agentic"]
            rag_type_used = "agentic"
        elif request.rag_type == RAGType.STANDARD:
            engine = engines["standard"]
            rag_type_used = "standard"
        elif request.rag_type == RAGType.VECTORLESS:
            engine = engines["vectorless"]
            rag_type_used = "vectorless"
        else:
            engine = engines["agentic"]
            rag_type_used = "agentic"

        # Build filters
        filters = None
        if request.category != CategoryType.ALL:
            filters = {"category": request.category.value}

        try:
            # Execute query
            if rag_type_used == "agentic":
                result = await engine.query(
                    query=request.query,
                    top_k=request.top_k,
                    filters=filters,
                    enable_reflection=request.enable_reflection
                )
            else:
                result = await engine.query(
                    query=request.query,
                    top_k=request.top_k,
                    filters=filters
                )

            # Build response
            return QueryResponse(
                success=True,
                answer=result.answer,
                sources=[
                    SourceInfo(
                        chunk_id=s.chunk_id,
                        text=s.text[:500] + "..." if len(s.text) > 500 else s.text,
                        score=s.score,
                        source_file=s.source_file,
                        category=s.category,
                        metadata=s.metadata
                    )
                    for s in result.sources
                ],
                rag_type=result.rag_type,
                query=result.query,
                latency=LatencyInfo(**result.latency),
                metadata=result.metadata,
                timestamp=result.timestamp
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare", response_model=CompareResponse)
async def compare_rag_engines(request: CompareRequest):
    """Compare results from all RAG engines"""
    results = {}
    filters = None
    if request.category != CategoryType.ALL:
        filters = {"category": request.category.value}

    # Run all engines in parallel
    async def run_engine(name: str, engine) -> tuple:
        try:
            if name == "agentic":
                result = await engine.query(
                    query=request.query,
                    top_k=request.top_k,
                    filters=filters,
                    enable_reflection=True
                )
            else:
                result = await engine.query(
                    query=request.query,
                    top_k=request.top_k,
                    filters=filters
                )
            return name, result
        except Exception as e:
            return name, None

    tasks = [
        run_engine("standard", engines["standard"]),
        run_engine("vectorless", engines["vectorless"]),
        run_engine("agentic", engines["agentic"])
    ]

    engine_results = await asyncio.gather(*tasks)

    for name, result in engine_results:
        if result:
            results[name] = QueryResponse(
                success=True,
                answer=result.answer,
                sources=[
                    SourceInfo(
                        chunk_id=s.chunk_id,
                        text=s.text[:300] + "..." if len(s.text) > 300 else s.text,
                        score=s.score,
                        source_file=s.source_file,
                        category=s.category,
                        metadata=s.metadata
                    )
                    for s in result.sources
                ],
                rag_type=result.rag_type,
                query=result.query,
                latency=LatencyInfo(**result.latency),
                metadata=result.metadata,
                timestamp=result.timestamp
            )

    # Generate comparison
    comparison = {
        "latencies": {
            name: resp.latency.total_ms for name, resp in results.items()
        },
        "source_counts": {
            name: len(resp.sources) for name, resp in results.items()
        },
        "avg_scores": {
            name: sum(s.score for s in resp.sources) / len(resp.sources) if resp.sources else 0
            for name, resp in results.items()
        }
    }

    # Determine recommendation
    fastest = min(comparison["latencies"], key=comparison["latencies"].get)
    highest_score = max(comparison["avg_scores"], key=comparison["avg_scores"].get)

    if fastest == highest_score:
        recommendation = f"Use {fastest} RAG - fastest and highest relevance scores"
    else:
        recommendation = f"Use {highest_score} RAG for accuracy or {fastest} RAG for speed"

    return CompareResponse(
        query=request.query,
        results=results,
        comparison=comparison,
        recommendation=recommendation
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    category: str = Form(...)
):
    """Upload and process documents"""
    processed_docs = []
    errors = []

    # Create temp directory for uploads
    upload_dir = Path(settings.documents_dir) / category
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        try:
            # Save file
            file_path = upload_dir / file.filename
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)

            # Process document
            doc = document_processor.process_file(str(file_path), category)

            # Add to all engines
            doc_dict = {
                "id": doc.id,
                "filename": doc.filename,
                "content": doc.content,
                "category": doc.category,
                "chunks": doc.chunks
            }

            await asyncio.gather(
                engines["standard"].add_documents([doc_dict]),
                engines["vectorless"].add_documents([doc_dict])
            )

            processed_docs.append(DocumentInfo(
                id=doc.id,
                filename=doc.filename,
                category=doc.category,
                file_type=doc.file_type,
                chunks_count=len(doc.chunks),
                status="processed"
            ))

        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")

    return UploadResponse(
        success=len(errors) == 0,
        documents_processed=len(processed_docs),
        documents=processed_docs,
        errors=errors
    )


@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    tracker = get_latency_tracker()

    engine_stats = {}
    total_docs = 0
    categories = {"sam": 0, "itam": 0, "itv": 0, "sre": 0}

    for name, engine in engines.items():
        stats = await engine.get_stats()
        engine_stats[name] = EngineStats(
            engine=name,
            status=stats.get("status", "unknown"),
            details=stats
        )
        if name == "standard" and "total_documents" in stats:
            total_docs = stats["total_documents"]

    return SystemStats(
        engines=engine_stats,
        total_documents=total_docs,
        categories=categories,
        latency_metrics=tracker.to_dict()
    )


@app.get("/api/metrics")
async def get_metrics():
    """Get detailed latency metrics"""
    tracker = get_latency_tracker()
    return JSONResponse(content=tracker.to_dict())


@app.delete("/api/clear")
async def clear_data():
    """Clear all documents and reinitialize engines"""
    global engines

    # Reinitialize engines
    engines["standard"] = StandardRAGEngine()
    engines["vectorless"] = VectorlessRAGEngine()
    engines["agentic"] = AgenticRAGEngine()

    await asyncio.gather(
        engines["standard"].initialize(),
        engines["vectorless"].initialize(),
        engines["agentic"].initialize()
    )

    return {"status": "cleared", "message": "All data cleared and engines reinitialized"}


@app.get("/api/graph")
async def get_knowledge_graph():
    """Get knowledge graph data for visualization"""
    nodes = []
    edges = []
    node_id_counter = 0

    # Get stats from standard RAG engine
    if engines.get("standard") and engines["standard"].is_initialized:
        stats = await engines["standard"].get_stats()

        # Create category nodes
        categories = {"sam": "Software Asset Management", "itam": "IT Asset Management",
                     "itv": "IT Visibility", "sre": "Site Reliability Engineering"}

        category_node_ids = {}
        for cat_key, cat_name in categories.items():
            node_id = f"cat_{cat_key}"
            nodes.append({
                "id": node_id,
                "label": cat_name,
                "title": f"Category: {cat_name}",
                "type": "category"
            })
            category_node_ids[cat_key] = node_id
            node_id_counter += 1

        # Get document info from vectorless engine which has structured data
        if engines.get("vectorless") and engines["vectorless"].is_initialized:
            vectorless_stats = await engines["vectorless"].get_stats()

            if "documents" in vectorless_stats:
                for doc in vectorless_stats["documents"]:
                    doc_id = f"doc_{doc.get('id', node_id_counter)}"
                    nodes.append({
                        "id": doc_id,
                        "label": doc.get("filename", "Document"),
                        "title": f"Document: {doc.get('filename', 'Unknown')}",
                        "type": "document"
                    })
                    node_id_counter += 1

                    # Connect to category
                    cat = doc.get("category", "sam")
                    if cat in category_node_ids:
                        edges.append({
                            "from": category_node_ids[cat],
                            "to": doc_id,
                            "label": "contains"
                        })

                    # Add section nodes
                    sections = doc.get("sections", [])
                    for i, section in enumerate(sections[:5]):  # Limit sections
                        section_id = f"sec_{doc_id}_{i}"
                        nodes.append({
                            "id": section_id,
                            "label": section.get("title", f"Section {i+1}"),
                            "title": section.get("title", "Section"),
                            "type": "section"
                        })
                        edges.append({
                            "from": doc_id,
                            "to": section_id,
                            "label": "has section"
                        })
                        node_id_counter += 1

        # If no documents, create sample structure
        if len(nodes) == len(categories):
            # Add sample nodes for demonstration
            for i, (cat_key, cat_name) in enumerate(categories.items()):
                sample_doc_id = f"sample_doc_{i}"
                nodes.append({
                    "id": sample_doc_id,
                    "label": f"{cat_key.upper()} Guide",
                    "title": f"Sample document for {cat_name}",
                    "type": "document"
                })
                edges.append({
                    "from": category_node_ids[cat_key],
                    "to": sample_doc_id,
                    "label": "contains"
                })

    return {"nodes": nodes, "edges": edges}


@app.get("/api/tree")
async def get_document_tree():
    """Get hierarchical document tree for vectorless RAG visualization"""
    tree = []

    categories = {"sam": "Software Asset Management", "itam": "IT Asset Management",
                 "itv": "IT Visibility", "sre": "Site Reliability Engineering"}

    if engines.get("vectorless") and engines["vectorless"].is_initialized:
        vectorless_stats = await engines["vectorless"].get_stats()

        # Build tree from vectorless engine data
        for cat_key, cat_name in categories.items():
            cat_node = {
                "id": f"cat_{cat_key}",
                "title": cat_name,
                "type": "category",
                "children": []
            }

            # Get documents for this category
            if "documents" in vectorless_stats:
                for doc in vectorless_stats["documents"]:
                    if doc.get("category") == cat_key:
                        doc_node = {
                            "id": doc.get("id", ""),
                            "title": doc.get("filename", "Document"),
                            "type": "document",
                            "children": []
                        }

                        # Add sections as children
                        sections = doc.get("sections", [])
                        for section in sections:
                            section_node = {
                                "id": section.get("id", ""),
                                "title": section.get("title", "Section"),
                                "type": "section",
                                "children": []
                            }

                            # Add chunks as children
                            chunks = section.get("chunks", [])
                            for chunk in chunks[:3]:  # Limit chunks shown
                                chunk_node = {
                                    "id": chunk.get("id", ""),
                                    "title": chunk.get("text", "")[:50] + "...",
                                    "type": "chunk"
                                }
                                section_node["children"].append(chunk_node)

                            doc_node["children"].append(section_node)

                        cat_node["children"].append(doc_node)

            tree.append(cat_node)
    else:
        # Return sample tree structure
        for cat_key, cat_name in categories.items():
            tree.append({
                "id": f"cat_{cat_key}",
                "title": cat_name,
                "type": "category",
                "children": [{
                    "id": f"sample_{cat_key}",
                    "title": f"Sample {cat_key.upper()} Document",
                    "type": "document",
                    "children": []
                }]
            })

    return {"tree": tree}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
