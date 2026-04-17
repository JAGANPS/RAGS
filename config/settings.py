"""
Configuration settings for the Multi-RAG System
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # AWS Configuration
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_profile: Optional[str] = Field(default=None, alias="AWS_PROFILE")
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")

    # Bedrock Model Configuration
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-6",
        alias="BEDROCK_MODEL_ID"
    )
    bedrock_max_tokens: int = Field(default=4096, alias="BEDROCK_MAX_TOKENS")
    bedrock_temperature: float = Field(default=0.7, alias="BEDROCK_TEMPERATURE")

    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )

    # Vector Store Configuration
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    faiss_index_path: str = Field(default="./data/faiss_index", alias="FAISS_INDEX_PATH")

    # Document Configuration
    documents_dir: str = Field(default="./documents", alias="DOCUMENTS_DIR")

    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    # RAG Configuration
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")

    # Metrics
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings"""
    return settings
