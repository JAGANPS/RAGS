#!/usr/bin/env python3
"""
Multi-RAG System - Main Entry Point
Run this script to start the application
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress verbose logging and warnings before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check if required environment variables are set"""
    required_vars = []
    missing = []

    # AWS credentials can come from environment, profile, or IAM role
    # We'll just check that the region is accessible

    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these variables or create a .env file")
        print("See .env.example for reference")
        return False

    return True


def create_directories():
    """Create necessary directories"""
    dirs = [
        "data/chroma_db",
        "data/faiss_index",
        "data/vectorless_index",
        "documents/sam",
        "documents/itam",
        "documents/itv",
        "documents/sre",
    ]

    for dir_path in dirs:
        Path(project_root / dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point"""
    print("=" * 60)
    print("  Multi-RAG System")
    print("  Flexera Knowledge Base with Standard, Vectorless & Agentic RAG")
    print("=" * 60)
    print()

    # Create directories
    create_directories()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Copy .env.example to .env if .env doesn't exist
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("Created .env file from .env.example")
        print("Please update with your AWS credentials if needed")
        print()

    # Start the server
    print("Starting server...")
    print("Access the UI at: http://localhost:8000")
    print()

    import uvicorn
    from config.settings import settings

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
