"""
Parameter Injection pattern for integrating vector databases with LLMs.

This module implements the more advanced Parameter Injection pattern, which
directly modifies LLM parameters using vector-derived features for improved
inference performance.
"""

from vector_llm_integration.parameter_injection.base import ParameterInjectionModel
from vector_llm_integration.parameter_injection.vector_context_layer import VectorContextLayer

__all__ = [
    "ParameterInjectionModel",
    "VectorContextLayer",
]
