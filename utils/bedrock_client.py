"""
AWS Bedrock client for Claude Sonnet model integration
"""
import boto3
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from config.settings import settings


@dataclass
class BedrockResponse:
    """Response from Bedrock model"""
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    stop_reason: str


class BedrockClient:
    """AWS Bedrock client for Claude Sonnet"""

    def __init__(
        self,
        model_id: str = None,
        region: str = None,
        max_tokens: int = None,
        temperature: float = None
    ):
        self.model_id = model_id or settings.bedrock_model_id
        self.region = region or settings.aws_region
        self.max_tokens = max_tokens or settings.bedrock_max_tokens
        self.temperature = temperature or settings.bedrock_temperature
        self.client = None
        self._initialized = False

        # Try to initialize Bedrock client
        try:
            session_kwargs = {"region_name": self.region}
            if settings.aws_profile:
                session_kwargs["profile_name"] = settings.aws_profile

            session = boto3.Session(**session_kwargs)
            self.client = session.client("bedrock-runtime")
            self._initialized = True
            print(f"Bedrock client initialized for region: {self.region}")
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock client: {e}")
            print("RAG engines will work but LLM queries will fail until AWS credentials are configured.")

    @property
    def is_available(self) -> bool:
        """Check if Bedrock client is available"""
        return self._initialized and self.client is not None

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> BedrockResponse:
        """Invoke Claude model with a prompt"""
        if not self.is_available:
            # Return a mock response when Bedrock isn't available
            return BedrockResponse(
                content="[AWS Bedrock not configured. Please set up AWS credentials to enable LLM responses.]",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                model_id=self.model_id,
                stop_reason="no_client"
            )

        messages = [{"role": "user", "content": prompt}]

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages
        }

        if system_prompt:
            request_body["system"] = system_prompt

        if stop_sequences:
            request_body["stop_sequences"] = stop_sequences

        start_time = time.perf_counter()

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        response_body = json.loads(response["body"].read())

        return BedrockResponse(
            content=response_body["content"][0]["text"],
            input_tokens=response_body["usage"]["input_tokens"],
            output_tokens=response_body["usage"]["output_tokens"],
            latency_ms=latency_ms,
            model_id=self.model_id,
            stop_reason=response_body.get("stop_reason", "end_turn")
        )

    def invoke_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> BedrockResponse:
        """Invoke with RAG context"""

        default_system = """You are a helpful AI assistant specialized in IT Asset Management (ITAM),
Software Asset Management (SAM), IT Value (ITV), and Site Reliability Engineering (SRE) domains.

Use the provided context to answer questions accurately. If the context doesn't contain
relevant information, say so clearly. Always cite the source documents when possible."""

        rag_prompt = f"""Context Information:
---
{context}
---

Based on the above context, please answer the following question:
{query}

If the context doesn't contain enough information to fully answer the question,
acknowledge what information is available and what is missing."""

        return self.invoke(
            prompt=rag_prompt,
            system_prompt=system_prompt or default_system
        )

    def stream_invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """Stream response from Claude model"""
        if not self.is_available:
            yield "[AWS Bedrock not configured]"
            return

        messages = [{"role": "user", "content": prompt}]

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages
        }

        if system_prompt:
            request_body["system"] = system_prompt

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")
            elif chunk["type"] == "message_stop":
                break


# Singleton instance
_bedrock_client: Optional[BedrockClient] = None


def get_bedrock_client() -> BedrockClient:
    """Get or create Bedrock client instance"""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = BedrockClient()
    return _bedrock_client
