# Multi-RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system featuring three distinct RAG architectures:
- **Standard RAG** - Vector-based semantic similarity search
- **Vectorless RAG** - Reasoning-based hierarchical document navigation
- **Agentic RAG** - Multi-agent system with intelligent query routing

Built for **Flexera** domain knowledge: SAM (Software Asset Management), ITAM (IT Asset Management), ITV (IT Value), and SRE (Site Reliability Engineering).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                             │
│                    (Modern Web UI with Loaders)                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                             │
│              (Query Processing, Document Management)                │
└─────────────────────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   STANDARD RAG   │  │  VECTORLESS RAG  │  │   AGENTIC RAG    │
│                  │  │                  │  │                  │
│ • ChromaDB       │  │ • Hierarchical   │  │ • Query Router   │
│ • Embeddings     │  │   ToC Index      │  │ • Multi-Agent    │
│ • Cosine         │  │ • LLM Reasoning  │  │ • Reflection     │
│   Similarity     │  │ • No Vectors     │  │ • Synthesis      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AWS BEDROCK (Claude Sonnet 4)                  │
│                    Model: us.anthropic.claude-sonnet-4-6            │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### RAG Engines
- **Standard RAG**: Fast vector similarity search using ChromaDB and sentence transformers
- **Vectorless RAG**: PageIndex-inspired reasoning over hierarchical document structure
- **Agentic RAG**: Intelligent routing, query decomposition, synthesis, and reflection

### UI Features
- Modern, professional dark theme
- Beautiful animated loaders (DNA helix, neural network, orbit)
- Real-time latency metrics visualization
- Source document display with relevance scores
- Compare mode for side-by-side RAG evaluation
- Drag-and-drop document upload

### Document Support
- PDF, DOCX, DOC
- TXT
- CSV, XLSX, XLS

### Latency Tracking
- Per-operation timing breakdown
- Retrieval vs generation latency
- Historical metrics aggregation
- P50, P95, P99 percentiles

## Prerequisites

- Python 3.10+
- AWS Account with Bedrock access
- AWS credentials configured

## Installation

1. **Clone and navigate to the project**:
```bash
cd /Users/Downloads/Demo-RAG
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your AWS credentials if needed
```

5. **Configure AWS credentials** (if not already done):
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1

# Option 3: Use AWS profile in .env
AWS_PROFILE=your_profile_name
```

## Running the Application

```bash
python run.py
```

Or directly with uvicorn:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Access the UI at: **http://localhost:8000**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/health` | GET | Health check |
| `/api/query` | POST | Process a query |
| `/api/compare` | POST | Compare all RAG engines |
| `/api/upload` | POST | Upload documents |
| `/api/stats` | GET | System statistics |
| `/api/metrics` | GET | Latency metrics |

### Query Request Example
```json
{
  "query": "What are the best practices for software license compliance?",
  "rag_type": "auto",
  "category": "sam",
  "top_k": 5,
  "enable_reflection": true
}
```

### Response Example
```json
{
  "success": true,
  "answer": "Based on the SAM best practices guide...",
  "sources": [
    {
      "chunk_id": "abc123_0",
      "text": "License compliance requires...",
      "score": 0.89,
      "source_file": "sam_best_practices.txt",
      "category": "sam"
    }
  ],
  "rag_type": "agentic_rag",
  "latency": {
    "routing_ms": 245.5,
    "retrieval_ms": 523.2,
    "synthesis_ms": 1234.5,
    "reflection_ms": 892.1,
    "total_ms": 2895.3
  }
}
```

## Project Structure

```
Demo-RAG/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   └── models.py        # Pydantic models
├── config/
│   ├── __init__.py
│   └── settings.py      # Configuration management
├── rag_engines/
│   ├── __init__.py
│   ├── base.py          # Base RAG interface
│   ├── standard_rag.py  # Vector-based RAG
│   ├── vectorless_rag.py # Reasoning-based RAG
│   └── agentic_rag.py   # Multi-agent RAG
├── utils/
│   ├── __init__.py
│   ├── document_processor.py  # Document parsing
│   ├── bedrock_client.py      # AWS Bedrock integration
│   └── latency_tracker.py     # Metrics tracking
├── frontend/
│   ├── static/
│   │   ├── css/styles.css
│   │   └── js/app.js
│   └── templates/
│       └── index.html
├── documents/
│   ├── sam/             # Software Asset Management docs
│   ├── itam/            # IT Asset Management docs
│   ├── itv/             # IT Value docs
│   └── sre/             # Site Reliability Engineering docs
├── data/                # Generated at runtime
│   ├── chroma_db/
│   ├── faiss_index/
│   └── vectorless_index/
├── .env.example
├── requirements.txt
├── run.py
└── README.md
```

## Sample Documents Included

- **SAM**: Software inventory, best practices guide
- **ITAM**: Hardware inventory, lifecycle management guide
- **ITV**: IT value optimization strategies
- **SRE**: Site reliability engineering handbook

## RAG Engine Comparison

| Feature | Standard | Vectorless | Agentic |
|---------|----------|------------|---------|
| Speed | Fastest | Medium | Slowest |
| Accuracy | Good | Best for structured docs | Best overall |
| Complex queries | Limited | Good | Excellent |
| Multi-hop reasoning | No | Limited | Yes |
| Query decomposition | No | No | Yes |
| Self-reflection | No | No | Yes |

## Troubleshooting

### AWS Credentials Error
```
botocore.exceptions.NoCredentialsError
```
Solution: Configure AWS credentials using one of the methods above.

### Model Access Error
```
AccessDeniedException: You don't have access to the model
```
Solution: Ensure Bedrock model access is enabled in your AWS account for `us.anthropic.claude-sonnet-4-6`.

### ChromaDB Error
```
sqlite3.OperationalError: database is locked
```
Solution: Stop any other processes using the database, or delete `data/chroma_db/` and restart.

## License

MIT

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request
