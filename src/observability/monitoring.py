"""
Monitoring and telemetry for vector database operations with LLMs.

This module implements the OpenLLMetry framework for monitoring vector database
operations, including tracing, metrics collection, and observability.
"""
import logging
import time
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from datetime import datetime

# Conditional import to handle cases where traceloop isn't installed
try:
    from traceloop.sdk import Traceloop
    TRACELOOP_AVAILABLE = True
except ImportError:
    TRACELOOP_AVAILABLE = False
    logging.warning("Traceloop SDK not available. Using fallback monitoring.")

logger = logging.getLogger(__name__)

# Define a type variable for function return types
T = TypeVar('T')

def setup_monitoring(
    vector_db: str,
    dimensions: int,
    collection: Optional[str] = None,
    enable_tracing: bool = True,
    sample_rate: float = 1.0,
    api_key: Optional[str] = None
) -> bool:
    """
    Set up monitoring for vector database operations.
    
    Args:
        vector_db: The vector database name (e.g., 'chroma', 'pinecone')
        dimensions: Number of dimensions in embeddings
        collection: Optional collection/index name
        enable_tracing: Whether to enable tracing
        sample_rate: Sampling rate for telemetry
        api_key: Optional API key for telemetry service
        
    Returns:
        Boolean indicating success
    """
    logger.info(f"Setting up monitoring for {vector_db} with {dimensions} dimensions")
    
    if TRACELOOP_AVAILABLE:
        try:
            # Initialize Traceloop
            Traceloop.init(
                app_name="vector-llm-integration",
                api_key=api_key,
                sample_rate=sample_rate,
                disable_batch=True
            )
            
            # Monitor vector database operations
            Traceloop.monitor_vector_db(
                vector_db,
                dimensions=dimensions,
                collection=collection
            )
            
            logger.info("Successfully set up OpenLLMetry monitoring")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Traceloop monitoring: {e}")
            return False
    else:
        logger.info("Using fallback monitoring (Traceloop not available)")
        return True

class MetricsCollector:
    """
    Collects and stores metrics for vector database operations.
    
    This is a fallback for when Traceloop is not available.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance.operations = {}
            cls._instance.latencies = {}
            cls._instance.error_counts = {}
            cls._instance.last_reset_time = datetime.now()
        return cls._instance
    
    def record_operation(self, db_type: str, operation: str, latency_ms: float, success: bool) -> None:
        """
        Record a vector database operation.
        
        Args:
            db_type: Type of vector database
            operation: Operation name
            latency_ms: Operation latency in milliseconds
            success: Whether the operation succeeded
        """
        key = f"{db_type}_{operation}"
        
        # Initialize counters if needed
        if key not in self.operations:
            self.operations[key] = 0
            self.latencies[key] = []
            self.error_counts[key] = 0
            
        # Update metrics
        self.operations[key] += 1
        self.latencies[key].append(latency_ms)
        
        if not success:
            self.error_counts[key] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics.
        
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        for key in self.operations:
            count = self.operations[key]
            latencies = self.latencies[key]
            errors = self.error_counts[key]
            
            if count > 0:
                avg_latency = sum(latencies) / count
                max_latency = max(latencies) if latencies else 0
                error_rate = errors / count if count > 0 else 0
                
                metrics[key] = {
                    "count": count,
                    "avg_latency_ms": avg_latency,
                    "max_latency_ms": max_latency,
                    "error_count": errors,
                    "error_rate": error_rate
                }
        
        metrics["uptime_seconds"] = (datetime.now() - self.last_reset_time).total_seconds()
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.operations = {}
        self.latencies = {}
        self.error_counts = {}
        self.last_reset_time = datetime.now()

# Global metrics collector
metrics_collector = MetricsCollector()

def trace_vectordb_operation(db_type: str, operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for tracing vector database operations.
    
    Args:
        db_type: Type of vector database
        operation: Operation name
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            success = True
            
            # Start span if traceloop is available
            if TRACELOOP_AVAILABLE:
                span = Traceloop.start_span(f"{db_type}.{operation}")
                
                try:
                    # Record operation parameters
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("db.type", db_type)
                        span.set_attribute("db.operation", operation)
                        
                        # Record query parameters if available
                        if "query" in kwargs:
                            span.set_attribute("db.query", kwargs["query"])
                        if "top_k" in kwargs:
                            span.set_attribute("db.top_k", kwargs.get("top_k"))
                        if "filters" in kwargs and kwargs["filters"]:
                            span.set_attribute("db.has_filters", True)
                        
                        # If this is an embedding operation, record document count
                        if operation == "add_documents" and "documents" in kwargs:
                            if isinstance(kwargs["documents"], list):
                                span.set_attribute("document_count", len(kwargs["documents"]))
            else:
                span = None
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                success = False
                if span and hasattr(span, "record_exception"):
                    span.record_exception(e)
                raise
                
            finally:
                # Calculate latency
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Record metrics
                metrics_collector.record_operation(db_type, operation, latency_ms, success)
                
                # End span if traceloop is available
                if span and hasattr(span, "end"):
                    span.set_attribute("latency_ms", latency_ms)
                    span.set_attribute("success", success)
                    span.end()
                
                # Log the operation
                log_level = logging.INFO if success else logging.ERROR
                logger.log(
                    log_level,
                    f"{db_type}.{operation} completed in {latency_ms:.2f}ms (success={success})"
                )
                
        return cast(Callable[..., T], wrapper)
    return decorator

def log_embedding_metrics(
    embedding_name: str, 
    dimensions: int,
    vectors: Any,
    **kwargs: Any
) -> None:
    """
    Log metrics about embedding vectors for quality monitoring.
    
    Args:
        embedding_name: Name of the embedding model
        dimensions: Number of dimensions
        vectors: The embedding vectors
        **kwargs: Additional metadata
    """
    try:
        import numpy as np
        
        # Convert to numpy if it's not already
        if not isinstance(vectors, np.ndarray):
            if hasattr(vectors, "numpy"):
                # Handle torch tensors
                vectors = vectors.numpy()
            else:
                # Try to convert list-like objects
                vectors = np.array(vectors)
        
        # Calculate basic metrics
        vector_count = len(vectors)
        avg_norm = float(np.mean(np.linalg.norm(vectors, axis=1)))
        
        # Check for NaN values
        nan_count = int(np.isnan(vectors).sum())
        
        # Calculate variance of vectors
        avg_vector = np.mean(vectors, axis=0)
        avg_distance = float(np.mean(np.linalg.norm(vectors - avg_vector, axis=1)))
        
        # Log metrics
        metrics = {
            "embedding_name": embedding_name,
            "dimensions": dimensions,
            "vector_count": vector_count,
            "avg_norm": avg_norm,
            "nan_count": nan_count,
            "avg_distance_from_mean": avg_distance
        }
        
        # Add any additional metadata
        metrics.update(kwargs)
        
        # Log to appropriate systems
        if TRACELOOP_AVAILABLE:
            span = Traceloop.start_span("embedding.metrics")
            for key, value in metrics.items():
                span.set_attribute(key, value)
            span.end()
        
        logger.info(f"Embedding metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error logging embedding metrics: {e}")
