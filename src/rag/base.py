"""
Base implementation of the Retrieval-Augmented Generation (RAG) pipeline.

This module defines the abstract base class for all RAG implementations,
establishing the common interface and shared functionality.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGPipeline(ABC):
    """
    Abstract base class for Retrieval-Augmented Generation (RAG) pipelines.
    
    This class defines the interface for RAG implementations and provides
    shared functionality for document processing, retrieval, and generation.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo", 
        embedding_model_name: str = "text-embedding-ada-002",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.0,
        max_tokens: int = 500,
        top_k: int = 5
    ) -> None:
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: Name of the LLM model to use for generation
            embedding_model_name: Name of the embedding model
            chunk_size: Maximum size of document chunks for processing
            chunk_overlap: Overlap size between consecutive chunks
            temperature: Temperature parameter for LLM generation
            max_tokens: Maximum number of tokens to generate
            top_k: Number of top documents to retrieve
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        
        logger.info(f"Initializing RAG pipeline with {model_name} and {embedding_model_name}")
        
        # Initialize components in the implementation
        self._initialize_components()
    
    @abstractmethod
    def _initialize_components(self) -> None:
        """
        Initialize vector store, embeddings model, and LLM.
        This method must be implemented by derived classes.
        """
        pass
    
    @abstractmethod
    def add_documents(
        self, 
        documents: Union[str, List[str], Path, List[Path]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Add documents to the retrieval database.
        
        Args:
            documents: Document(s) to add - can be text strings, file paths,
                      or a list of either
            metadata: Optional metadata for each document
            
        Returns:
            Boolean indicating success
        """
        pass
    
    @abstractmethod
    def get_relevant_documents(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides instance setting)
            filters: Optional filters to apply
            
        Returns:
            List of documents with text and metadata
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query the RAG system to get a response.
        
        Args:
            query: User query
            temperature: Temperature for generation (overrides instance setting)
            max_tokens: Max tokens to generate (overrides instance setting)
            filters: Optional filters for retrieval
            
        Returns:
            Generated response
        """
        pass

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.get("metadata", {}).get("source", f"Document {i+1}")
            context_parts.append(f"[{source}]:\n{doc['text']}\n")
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM with the query and retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        return f"""
Answer the question based on the following context. If the context doesn't contain 
relevant information, just say that you don't have enough information, but try your best 
to provide a helpful response.

Context:
{context}

Question: {query}

Answer:
"""
