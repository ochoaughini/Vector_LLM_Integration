#!/usr/bin/env python3
"""
Example demonstrating RAG pattern integration with ChromaDB.

This example shows how to use the ChromaRAGPipeline to implement
retrieval-augmented generation with a vector database.
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import ChromaRAGPipeline
from src.observability import setup_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample documents for the knowledge base
SAMPLE_DOCS = [
    """Vector databases are specialized database systems designed to store and query 
    high-dimensional vectors, which are mathematical representations of features or 
    content. These vectors are typically created using machine learning models, 
    particularly embedding models, which transform data (text, images, audio, etc.) 
    into numerical vectors that capture semantic meaning.""",
    
    """Retrieval-Augmented Generation (RAG) is an AI framework that enhances 
    large language models by retrieving relevant information from external knowledge 
    sources before generating responses. This approach combines the strengths of 
    retrieval-based and generation-based AI systems to produce more accurate, 
    up-to-date, and contextually relevant outputs.""",
    
    """Large Language Models (LLMs) are advanced AI systems trained on vast amounts 
    of text data to understand and generate human-like language. These models learn 
    patterns, relationships, and structures in language, enabling them to perform 
    a wide range of natural language processing tasks.""",
    
    """The Hierarchical Navigable Small World (HNSW) algorithm is a graph-based 
    approach for approximate nearest neighbor search in high-dimensional spaces. 
    It builds a multi-layered graph structure that enables efficient navigation 
    through the vector space, significantly speeding up similarity searches.""",
    
    """Product quantization is a technique for efficient similarity search in 
    high-dimensional spaces. It works by decomposing the high-dimensional vector 
    space into multiple lower-dimensional subspaces and quantizing each subspace 
    separately. This approach reduces memory requirements and accelerates search."""
]

def main():
    """Run the RAG integration example."""
    logger.info("Starting RAG integration example")
    
    # Set up monitoring
    setup_monitoring("chroma", dimensions=1536, collection="example_kb")
    
    # Initialize the RAG pipeline
    pipeline = ChromaRAGPipeline(
        model_name="gpt-3.5-turbo",
        embedding_model_name="text-embedding-ada-002",
        top_k=3
    )
    
    # Add documents to the knowledge base
    logger.info("Adding documents to the knowledge base")
    pipeline.add_documents(
        SAMPLE_DOCS,
        metadata=[
            {"source": "vector_db_definition.txt", "topic": "databases"},
            {"source": "rag_definition.txt", "topic": "architectures"},
            {"source": "llm_definition.txt", "topic": "models"},
            {"source": "hnsw_definition.txt", "topic": "algorithms"},
            {"source": "product_quantization.txt", "topic": "algorithms"}
        ]
    )
    
    # Example queries to test the system
    queries = [
        "What is a vector database and why is it useful?",
        "How does RAG improve LLM outputs?",
        "Explain how HNSW algorithm works for vector search.",
        "What are the benefits of product quantization?"
    ]
    
    # Process each query
    for i, query in enumerate(queries):
        logger.info(f"\nQuery {i+1}: {query}")
        
        # Get relevant documents
        logger.info("Retrieving relevant documents...")
        docs = pipeline.get_relevant_documents(query)
        
        # Print retrieved documents
        logger.info(f"Retrieved {len(docs)} relevant documents:")
        for j, doc in enumerate(docs):
            source = doc.get("metadata", {}).get("source", f"Document {j+1}")
            logger.info(f"  - [{source}] {doc['text'][:100]}...")
        
        # Query the RAG system
        logger.info("Generating response...")
        response = pipeline.query(query)
        
        logger.info(f"Response: {response}")
    
    logger.info("\nExample completed successfully")

if __name__ == "__main__":
    main()
