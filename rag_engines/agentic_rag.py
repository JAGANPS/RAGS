"""
Agentic RAG Engine using Agno Framework
Multi-agent system with planning, routing, and reflection capabilities
"""
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from agno.agent import Agent
from agno.models.aws import Claude

from .base import BaseRAGEngine, RetrievalResult, RAGResponse
from .standard_rag import StandardRAGEngine
from .vectorless_rag import VectorlessRAGEngine
from utils.bedrock_client import get_bedrock_client
from utils.latency_tracker import get_latency_tracker, track_latency
from config.settings import settings


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class RetrievalStrategy(Enum):
    VECTOR = "vector"
    REASONING = "reasoning"
    HYBRID = "hybrid"


@dataclass
class AgentDecision:
    """Decision made by the routing agent"""
    complexity: QueryComplexity
    strategy: RetrievalStrategy
    reasoning: str
    sub_queries: List[str]
    requires_iteration: bool


class AgenticRAGEngine(BaseRAGEngine):
    """
    Agentic RAG with intelligent query routing and multi-step reasoning

    Architecture:
    1. Router Agent - Analyzes query complexity and routes to appropriate strategy
    2. Retrieval Agents - Standard RAG and Vectorless RAG engines
    3. Synthesis Agent - Combines results and generates final answer
    4. Reflection Agent - Validates and improves answers
    """

    def __init__(self):
        super().__init__("agentic_rag")
        self.standard_rag: Optional[StandardRAGEngine] = None
        self.vectorless_rag: Optional[VectorlessRAGEngine] = None
        self.bedrock = None
        self.tracker = get_latency_tracker()

        # Agno agents
        self.router_agent: Optional[Agent] = None
        self.synthesis_agent: Optional[Agent] = None
        self.reflection_agent: Optional[Agent] = None

    async def initialize(self) -> None:
        """Initialize all components and agents"""
        if self._initialized:
            return

        # Initialize underlying RAG engines
        self.standard_rag = StandardRAGEngine()
        self.vectorless_rag = VectorlessRAGEngine()

        await asyncio.gather(
            self.standard_rag.initialize(),
            self.vectorless_rag.initialize()
        )

        self.bedrock = get_bedrock_client()

        # Skip Agno agent initialization at startup - will initialize lazily on first use
        # This prevents blocking during server startup
        print("Agentic RAG initialized (agents will be created on first use)")
        self._initialized = True

    def _ensure_agents(self):
        """Lazily initialize Agno agents on first use"""
        if self.router_agent is not None:
            return  # Already initialized

        try:
            model = Claude(id=settings.bedrock_model_id)

            self.router_agent = Agent(
                name="QueryRouter",
                model=model,
                instructions="""You are a query routing expert for IT Asset Management systems.
                Analyze queries and determine:
                1. Query complexity (simple/moderate/complex)
                2. Best retrieval strategy (vector for semantic, reasoning for structured docs, hybrid for complex)
                3. Whether query needs decomposition into sub-queries

                Consider the domain context: SAM (Software Asset Management), ITAM (IT Asset Management),
                ITV (IT Value), and SRE (Site Reliability Engineering).""",
                markdown=True
            )

            self.synthesis_agent = Agent(
                name="Synthesizer",
                model=model,
                instructions="""You are an expert at synthesizing information from multiple sources.
                Combine retrieved information into coherent, accurate answers.
                Always cite sources and highlight confidence levels.
                Focus on ITAM, SAM, ITV, and SRE domain knowledge.""",
                markdown=True
            )

            self.reflection_agent = Agent(
                name="Reflector",
                model=model,
                instructions="""You are a quality assurance agent.
                Review answers for accuracy, completeness, and relevance.
                Identify gaps or potential improvements.
                Suggest if additional retrieval is needed.""",
                markdown=True
            )
            print("Agno agents initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize Agno agents: {e}")
            # Continue without agents - will fall back to direct RAG

    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents to both underlying engines"""
        if not self._initialized:
            await self.initialize()

        # Add to both engines in parallel
        results = await asyncio.gather(
            self.standard_rag.add_documents(documents),
            self.vectorless_rag.add_documents(documents)
        )

        return max(results)  # Return the count from the engine that processed more

    async def _route_query(self, query: str) -> AgentDecision:
        """Use router agent to analyze and route query"""
        # Ensure agents are initialized
        self._ensure_agents()

        routing_prompt = f"""Analyze this query for an IT Asset Management knowledge base:

Query: {query}

Determine:
1. Complexity: simple (direct lookup), moderate (multi-fact), complex (multi-hop reasoning)
2. Strategy: vector (semantic similarity), reasoning (structured document navigation), hybrid (both)
3. Sub-queries: If complex, break into simpler sub-queries

Return JSON:
{{
    "complexity": "simple|moderate|complex",
    "strategy": "vector|reasoning|hybrid",
    "reasoning": "explanation of your decision",
    "sub_queries": ["list", "of", "sub-queries"] or [],
    "requires_iteration": true/false
}}"""

        with track_latency("agentic_rag.route_query"):
            response = self.bedrock.invoke(
                prompt=routing_prompt,
                system_prompt="You are a query routing expert. Return only valid JSON.",
                temperature=0.1
            )

        try:
            json_str = response.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            return AgentDecision(
                complexity=QueryComplexity(data.get("complexity", "moderate")),
                strategy=RetrievalStrategy(data.get("strategy", "hybrid")),
                reasoning=data.get("reasoning", ""),
                sub_queries=data.get("sub_queries", []),
                requires_iteration=data.get("requires_iteration", False)
            )
        except (json.JSONDecodeError, ValueError):
            # Default to hybrid strategy
            return AgentDecision(
                complexity=QueryComplexity.MODERATE,
                strategy=RetrievalStrategy.HYBRID,
                reasoning="Defaulting to hybrid strategy",
                sub_queries=[],
                requires_iteration=False
            )

    async def _execute_retrieval(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Execute retrieval based on strategy"""

        if strategy == RetrievalStrategy.VECTOR:
            return await self.standard_rag.retrieve(query, top_k)

        elif strategy == RetrievalStrategy.REASONING:
            return await self.vectorless_rag.retrieve(query, top_k)

        else:  # HYBRID
            # Execute both in parallel
            vector_results, reasoning_results = await asyncio.gather(
                self.standard_rag.retrieve(query, top_k // 2 + 1),
                self.vectorless_rag.retrieve(query, top_k // 2 + 1)
            )

            # Merge and deduplicate results
            seen_texts = set()
            merged = []

            for result in vector_results + reasoning_results:
                text_hash = hash(result.text[:100])
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    merged.append(result)

            # Sort by score and return top_k
            merged.sort(key=lambda x: x.score, reverse=True)
            return merged[:top_k]

    async def _synthesize_answer(
        self,
        query: str,
        sources: List[RetrievalResult],
        decision: AgentDecision
    ) -> str:
        """Use synthesis agent to generate answer"""

        context_parts = []
        for i, source in enumerate(sources):
            context_parts.append(
                f"[Source {i+1}: {source.source_file} | Category: {source.category} | Score: {source.score:.2%}]\n"
                f"{source.text}"
            )

        synthesis_prompt = f"""Query: {query}

Routing Decision:
- Complexity: {decision.complexity.value}
- Strategy: {decision.strategy.value}
- Reasoning: {decision.reasoning}

Retrieved Information:
---
{chr(10).join(context_parts)}
---

Synthesize a comprehensive answer based on the retrieved information.
- Cite sources by number [Source N]
- Indicate confidence level
- Note any gaps in the available information
- Focus on ITAM/SAM/ITV/SRE domain context"""

        with track_latency("agentic_rag.synthesize"):
            response = self.bedrock.invoke(
                prompt=synthesis_prompt,
                system_prompt="You are an expert synthesizer for IT Asset Management knowledge.",
                temperature=0.3
            )

        return response.content

    async def _reflect_and_improve(
        self,
        query: str,
        answer: str,
        sources: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """Use reflection agent to validate and potentially improve answer"""

        reflection_prompt = f"""Review this answer for quality and completeness:

Original Query: {query}

Generated Answer:
{answer}

Number of sources used: {len(sources)}

Evaluate:
1. Accuracy - Does the answer accurately reflect the sources?
2. Completeness - Are there obvious gaps?
3. Relevance - Does it address the query?
4. Quality - Is it well-structured and clear?

Return JSON:
{{
    "quality_score": 0.0-1.0,
    "is_complete": true/false,
    "gaps_identified": ["list of gaps"],
    "suggested_improvements": ["list of improvements"],
    "needs_additional_retrieval": true/false
}}"""

        with track_latency("agentic_rag.reflect"):
            response = self.bedrock.invoke(
                prompt=reflection_prompt,
                system_prompt="You are a quality assurance expert. Return only valid JSON.",
                temperature=0.1
            )

        try:
            json_str = response.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            return {
                "quality_score": 0.7,
                "is_complete": True,
                "gaps_identified": [],
                "suggested_improvements": [],
                "needs_additional_retrieval": False
            }

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Agentic retrieval with routing"""
        if not self._initialized:
            await self.initialize()

        # Route the query
        decision = await self._route_query(query)

        # Execute retrieval based on decision
        return await self._execute_retrieval(query, decision.strategy, top_k)

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        enable_reflection: bool = True
    ) -> RAGResponse:
        """Full agentic RAG pipeline with routing, retrieval, synthesis, and reflection"""
        if not self._initialized:
            await self.initialize()

        latency = {}
        total_start = time.perf_counter()

        # Step 1: Route query
        start = time.perf_counter()
        decision = await self._route_query(query)
        latency["routing_ms"] = (time.perf_counter() - start) * 1000

        # Step 2: Handle sub-queries if complex
        all_sources = []
        if decision.sub_queries:
            start = time.perf_counter()
            sub_results = await asyncio.gather(*[
                self._execute_retrieval(sq, decision.strategy, top_k // len(decision.sub_queries) + 1)
                for sq in decision.sub_queries
            ])
            for results in sub_results:
                all_sources.extend(results)
            latency["sub_query_retrieval_ms"] = (time.perf_counter() - start) * 1000

        # Step 3: Main retrieval
        start = time.perf_counter()
        main_sources = await self._execute_retrieval(query, decision.strategy, top_k)
        all_sources.extend(main_sources)
        latency["retrieval_ms"] = (time.perf_counter() - start) * 1000

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = hash(s.text[:100])
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        unique_sources = sorted(unique_sources, key=lambda x: x.score, reverse=True)[:top_k]

        # Step 4: Synthesize answer
        start = time.perf_counter()
        answer = await self._synthesize_answer(query, unique_sources, decision)
        latency["synthesis_ms"] = (time.perf_counter() - start) * 1000

        # Step 5: Reflect and validate (optional)
        reflection = {}
        if enable_reflection:
            start = time.perf_counter()
            reflection = await self._reflect_and_improve(query, answer, unique_sources)
            latency["reflection_ms"] = (time.perf_counter() - start) * 1000

        latency["total_ms"] = (time.perf_counter() - total_start) * 1000

        return RAGResponse(
            answer=answer,
            sources=unique_sources,
            rag_type="agentic_rag",
            query=query,
            latency=latency,
            metadata={
                "routing_decision": {
                    "complexity": decision.complexity.value,
                    "strategy": decision.strategy.value,
                    "reasoning": decision.reasoning,
                    "sub_queries": decision.sub_queries
                },
                "reflection": reflection,
                "engines_used": ["standard_rag", "vectorless_rag"] if decision.strategy == RetrievalStrategy.HYBRID else [
                    "standard_rag" if decision.strategy == RetrievalStrategy.VECTOR else "vectorless_rag"
                ]
            }
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all components"""
        if not self._initialized:
            return {"status": "not_initialized"}

        standard_stats = await self.standard_rag.get_stats()
        vectorless_stats = await self.vectorless_rag.get_stats()

        return {
            "status": "initialized",
            "engine": self.name,
            "components": {
                "standard_rag": standard_stats,
                "vectorless_rag": vectorless_stats
            },
            "agents": ["router", "synthesis", "reflection"],
            "capabilities": [
                "query_routing",
                "multi_strategy_retrieval",
                "query_decomposition",
                "answer_synthesis",
                "quality_reflection"
            ]
        }
