"""
Retrieval-Augmented Generation (RAG) pattern implementation.

This module implements the RAG pattern, which uses vector databases as external 
knowledge sources to enhance LLM outputs with dynamic context injection.
"""

from vector_llm_integration.rag.base import RAGPipeline
from vector_llm_integration.rag.chroma import ChromaRAGPipeline
from vector_llm_integration.rag.pinecone import PineconeRAGPipeline
from vector_llm_integration.rag.weaviate import WeaviateRAGPipeline
from vector_llm_integration.rag.milvus import MilvusRAGPipeline

__all__ = [
    "RAGPipeline", 
    "ChromaRAGPipeline",
    "PineconeRAGPipeline",
    "WeaviateRAGPipeline",
    "MilvusRAGPipeline",
]
