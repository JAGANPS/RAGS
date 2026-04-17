"""
Vectorless RAG Engine - PageIndex-style reasoning-based retrieval
No embeddings, no vector DB - uses hierarchical document structure and LLM reasoning
"""
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .base import BaseRAGEngine, RetrievalResult, RAGResponse
from utils.bedrock_client import get_bedrock_client
from utils.latency_tracker import get_latency_tracker, track_latency
from config.settings import settings


@dataclass
class DocumentNode:
    """Node in the document hierarchy tree"""
    id: str
    title: str
    summary: str
    content: str
    level: int
    page_range: str
    children: List['DocumentNode']
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "level": self.level,
            "page_range": self.page_range,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata
        }


class VectorlessRAGEngine(BaseRAGEngine):
    """
    Vectorless RAG using hierarchical document structure
    - Builds Table of Contents (ToC) tree for each document
    - LLM reasons through structure to find relevant sections
    - No embeddings or vector similarity required
    """

    def __init__(self):
        super().__init__("vectorless_rag")
        self.document_trees: Dict[str, DocumentNode] = {}
        self.document_contents: Dict[str, str] = {}
        self.bedrock = None
        self.tracker = get_latency_tracker()
        self.index_path = Path("./data/vectorless_index")

    async def initialize(self) -> None:
        """Initialize the vectorless RAG engine"""
        if self._initialized:
            return

        self.index_path.mkdir(parents=True, exist_ok=True)
        self.bedrock = get_bedrock_client()

        # Load existing index if available
        await self._load_index()
        self._initialized = True

    async def _load_index(self) -> None:
        """Load existing document index"""
        index_file = self.index_path / "document_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                data = json.load(f)
                for doc_id, tree_data in data.get("trees", {}).items():
                    self.document_trees[doc_id] = self._dict_to_node(tree_data)
                self.document_contents = data.get("contents", {})

    async def _save_index(self) -> None:
        """Save document index to disk"""
        index_file = self.index_path / "document_index.json"
        data = {
            "trees": {doc_id: tree.to_dict() for doc_id, tree in self.document_trees.items()},
            "contents": self.document_contents
        }
        with open(index_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _dict_to_node(self, data: Dict) -> DocumentNode:
        """Convert dictionary to DocumentNode"""
        return DocumentNode(
            id=data["id"],
            title=data["title"],
            summary=data.get("summary", ""),
            content=data.get("content", ""),
            level=data["level"],
            page_range=data.get("page_range", ""),
            children=[self._dict_to_node(c) for c in data.get("children", [])],
            metadata=data.get("metadata", {})
        )

    async def _build_document_tree(self, doc_id: str, content: str, filename: str, category: str) -> DocumentNode:
        """Build hierarchical structure for a document using LLM"""

        build_prompt = f"""Analyze this document and create a hierarchical Table of Contents structure.
For each section, provide:
1. A clear title
2. A brief summary (1-2 sentences)
3. The approximate content location

Document content:
---
{content[:8000]}  # Limit for context
---

Return a JSON structure like this:
{{
    "title": "Document Title",
    "summary": "Overall document summary",
    "sections": [
        {{
            "title": "Section Title",
            "summary": "Section summary",
            "subsections": [...]
        }}
    ]
}}

Only return valid JSON, no other text."""

        with track_latency("vectorless_rag.build_tree"):
            response = self.bedrock.invoke(
                prompt=build_prompt,
                system_prompt="You are a document structure analyzer. Return only valid JSON.",
                temperature=0.1
            )

        # Parse the response
        try:
            # Extract JSON from response
            json_str = response.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            structure = json.loads(json_str.strip())
        except json.JSONDecodeError:
            # Fallback to simple structure
            structure = {
                "title": filename,
                "summary": content[:200],
                "sections": []
            }

        # Convert to DocumentNode tree
        root = self._create_node_from_structure(
            structure, doc_id, 0, content, filename, category
        )

        return root

    def _create_node_from_structure(
        self,
        structure: Dict,
        doc_id: str,
        level: int,
        full_content: str,
        filename: str,
        category: str
    ) -> DocumentNode:
        """Create DocumentNode from parsed structure"""
        node_id = f"{doc_id}_L{level}_{structure.get('title', 'root')[:20].replace(' ', '_')}"

        children = []
        for i, section in enumerate(structure.get("sections", structure.get("subsections", []))):
            child = self._create_node_from_structure(
                section, doc_id, level + 1, full_content, filename, category
            )
            children.append(child)

        return DocumentNode(
            id=node_id,
            title=structure.get("title", "Untitled"),
            summary=structure.get("summary", ""),
            content="",  # Content loaded on demand
            level=level,
            page_range=structure.get("page_range", ""),
            children=children,
            metadata={
                "filename": filename,
                "category": category,
                "doc_id": doc_id
            }
        )

    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents and build hierarchical indexes"""
        if not self._initialized:
            await self.initialize()

        added_count = 0

        for doc in documents:
            doc_id = doc.get("id", "")
            content = doc.get("content", "")
            filename = doc.get("filename", "unknown")
            category = doc.get("category", "general")

            if not content:
                continue

            with track_latency("vectorless_rag.add_document", {"filename": filename}):
                # Store full content
                self.document_contents[doc_id] = content

                # Build hierarchical tree
                tree = await self._build_document_tree(doc_id, content, filename, category)
                self.document_trees[doc_id] = tree

                added_count += 1

        # Persist index
        await self._save_index()
        return added_count

    def _generate_toc(self) -> str:
        """Generate readable Table of Contents from all documents"""
        toc_parts = []

        for doc_id, tree in self.document_trees.items():
            toc_parts.append(self._node_to_toc(tree, ""))

        return "\n\n".join(toc_parts)

    def _node_to_toc(self, node: DocumentNode, indent: str) -> str:
        """Convert node to TOC string representation"""
        lines = [f"{indent}[{node.id}] {node.title}"]
        if node.summary:
            lines.append(f"{indent}  Summary: {node.summary}")

        for child in node.children:
            lines.append(self._node_to_toc(child, indent + "  "))

        return "\n".join(lines)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using LLM reasoning over document structure"""
        if not self._initialized:
            await self.initialize()

        results = []

        # Generate TOC for LLM to reason over
        toc = self._generate_toc()

        # LLM reasons about which sections are relevant
        reasoning_prompt = f"""You are analyzing a document knowledge base to find relevant sections for a query.

Table of Contents:
---
{toc}
---

Query: {query}

Based on the document structure and summaries, identify the {top_k} most relevant sections that would contain information to answer this query.

For each relevant section, explain why it's relevant.

Return as JSON:
{{
    "reasoning": "Your reasoning process",
    "relevant_sections": [
        {{
            "section_id": "the section ID from the TOC",
            "relevance_score": 0.0-1.0,
            "reason": "why this section is relevant"
        }}
    ]
}}

Only return valid JSON."""

        with track_latency("vectorless_rag.reason_retrieval"):
            response = self.bedrock.invoke(
                prompt=reasoning_prompt,
                system_prompt="You are a document retrieval expert. Analyze structure and find relevant sections.",
                temperature=0.2
            )

        # Parse reasoning response
        try:
            json_str = response.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            reasoning = json.loads(json_str.strip())
            relevant_sections = reasoning.get("relevant_sections", [])
        except json.JSONDecodeError:
            relevant_sections = []

        # Convert to RetrievalResults
        for section in relevant_sections[:top_k]:
            section_id = section.get("section_id", "")
            node = self._find_node_by_id(section_id)

            if node:
                # Get actual content from the document
                doc_id = node.metadata.get("doc_id", "")
                content = self.document_contents.get(doc_id, "")

                # Extract relevant portion (simplified - in production would be smarter)
                section_content = f"{node.title}\n{node.summary}\n\n{content[:1000]}"

                results.append(RetrievalResult(
                    chunk_id=section_id,
                    text=section_content,
                    score=section.get("relevance_score", 0.5),
                    metadata={
                        "reason": section.get("reason", ""),
                        "level": node.level
                    },
                    source_file=node.metadata.get("filename", "unknown"),
                    category=node.metadata.get("category", "general")
                ))

        return results

    def _find_node_by_id(self, node_id: str) -> Optional[DocumentNode]:
        """Find node by ID in all trees"""
        for tree in self.document_trees.values():
            found = self._search_node(tree, node_id)
            if found:
                return found
        return None

    def _search_node(self, node: DocumentNode, target_id: str) -> Optional[DocumentNode]:
        """Recursively search for node by ID"""
        if node.id == target_id:
            return node
        for child in node.children:
            found = self._search_node(child, target_id)
            if found:
                return found
        return None

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """Full vectorless RAG pipeline"""
        latency = {}

        # Reasoning-based retrieval
        start = time.perf_counter()
        sources = await self.retrieve(query, top_k, filters)
        latency["retrieval_ms"] = (time.perf_counter() - start) * 1000

        # Build context from retrieved sections
        context_parts = []
        for i, source in enumerate(sources):
            reason = source.metadata.get("reason", "")
            context_parts.append(
                f"[Source {i+1}: {source.source_file}]\n"
                f"Relevance reason: {reason}\n\n{source.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        start = time.perf_counter()
        with track_latency("vectorless_rag.generate"):
            response = self.bedrock.invoke_with_context(query, context)
        latency["generation_ms"] = (time.perf_counter() - start) * 1000
        latency["total_ms"] = latency["retrieval_ms"] + latency["generation_ms"]

        return RAGResponse(
            answer=response.content,
            sources=sources,
            rag_type="vectorless_rag",
            query=query,
            latency=latency,
            metadata={
                "model": response.model_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "reasoning_based": True,
                "documents_indexed": len(self.document_trees)
            }
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}

        total_nodes = sum(
            self._count_nodes(tree) for tree in self.document_trees.values()
        )

        # Build document list with tree structure for visualization
        documents = []
        for doc_id, tree in self.document_trees.items():
            doc_info = {
                "id": doc_id,
                "filename": tree.metadata.get("filename", "unknown"),
                "category": tree.metadata.get("category", "general"),
                "title": tree.title,
                "summary": tree.summary,
                "sections": self._tree_to_sections(tree)
            }
            documents.append(doc_info)

        return {
            "status": "initialized",
            "engine": self.name,
            "total_documents": len(self.document_trees),
            "total_nodes": total_nodes,
            "index_path": str(self.index_path),
            "uses_vectors": False,
            "uses_reasoning": True,
            "documents": documents
        }

    def _tree_to_sections(self, node: DocumentNode) -> List[Dict[str, Any]]:
        """Convert tree node children to sections list"""
        sections = []
        for child in node.children:
            section = {
                "id": child.id,
                "title": child.title,
                "summary": child.summary,
                "level": child.level,
                "chunks": [{"id": f"{child.id}_chunk", "text": child.summary}],
                "subsections": self._tree_to_sections(child)
            }
            sections.append(section)
        return sections

    def _count_nodes(self, node: DocumentNode) -> int:
        """Count total nodes in tree"""
        return 1 + sum(self._count_nodes(child) for child in node.children)
