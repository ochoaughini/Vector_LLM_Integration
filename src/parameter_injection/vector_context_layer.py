"""
Vector Context Layer implementation for the Parameter Injection pattern.

This module provides the VectorContextLayer class, which modifies attention layers
in transformer models to incorporate vector-derived features for improved performance.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class VectorContextLayer(nn.Module):
    """
    Enhanced attention layer that incorporates vector database knowledge.
    
    This layer modifies the standard attention mechanism by injecting vector-derived
    features from external knowledge sources directly into model parameters.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        vector_data: torch.Tensor,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        is_causal: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        injection_factor: float = 0.3
    ) -> None:
        """
        Initialize the Vector Context Layer.
        
        Args:
            base_layer: Original attention layer to wrap
            vector_data: Tensor containing vector knowledge base embeddings
            embedding_dim: Hidden dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias terms
            is_causal: Whether to apply causal masking
            device: Computation device
            dtype: Data type for computations
            injection_factor: Weight factor for vector context injection (0-1)
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.is_causal = is_causal
        self.dropout = dropout
        self.injection_factor = injection_factor
        
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}")
            
        # Store vector knowledge
        self.register_buffer("vector_data", vector_data)
        
        # Create vector key and value projections
        self.vector_k_proj = nn.Linear(vector_data.shape[-1], embedding_dim, bias=bias, device=device, dtype=dtype)
        self.vector_v_proj = nn.Linear(vector_data.shape[-1], embedding_dim, bias=bias, device=device, dtype=dtype)
        
        # Create context attention score scaling
        self.context_scale = nn.Parameter(torch.tensor(1.0))
        
        logger.info(f"Initialized VectorContextLayer with {vector_data.shape[0]} knowledge vectors")
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = False,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Vector Context Layer.
        
        This method extends the standard self-attention with vector context injection,
        enhancing the attention patterns with external knowledge.
        
        Args:
            query: Query embeddings
            key: Key embeddings
            value: Value embeddings
            key_padding_mask: Mask for keys
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor and optional attention weights
        """
        # First, compute standard attention with the base layer
        base_output, base_attn_weights = self.base_layer(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,  # Always need weights for context injection
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal or self.is_causal
        )
        
        # Project vector knowledge to key and value spaces
        batch_size, seq_len = query.shape[0], query.shape[1]
        vector_k = self.vector_k_proj(self.vector_data)  # [num_vectors, embedding_dim]
        vector_v = self.vector_v_proj(self.vector_data)  # [num_vectors, embedding_dim]
        
        # Reshape for multi-head attention
        vector_k = vector_k.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, num_vectors, head_dim]
        vector_v = vector_v.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, num_vectors, head_dim]
        
        # Reshape query for context attention computation
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(2, 0, 1, 3)  # [num_heads, batch_size, seq_len, head_dim]
        
        # Compute attention scores with vector knowledge
        # Scale query for numerical stability
        q = q / math.sqrt(self.head_dim)
        
        # Compute context attention scores
        context_attn_scores = torch.matmul(q, vector_k.transpose(-2, -1))  # [num_heads, batch_size, seq_len, num_vectors]
        context_attn_scores = context_attn_scores * self.context_scale
        
        # Apply softmax to get context attention weights
        context_attn_weights = F.softmax(context_attn_scores, dim=-1)
        context_attn_weights = F.dropout(context_attn_weights, p=self.dropout, training=self.training)
        
        # Compute context-based output
        context_output = torch.matmul(context_attn_weights, vector_v)  # [num_heads, batch_size, seq_len, head_dim]
        context_output = context_output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, self.embedding_dim)
        
        # Combine base output with context-enriched output
        combined_output = (1 - self.injection_factor) * base_output + self.injection_factor * context_output
        
        # Return combined output and optionally attention weights
        if need_weights:
            # Combine attention weights for visualization/analysis
            combined_attn = {"base_attn": base_attn_weights, "context_attn": context_attn_weights}
            return combined_output, combined_attn
        else:
            return combined_output, None
    
    @classmethod
    def from_layer(
        cls,
        layer: nn.Module,
        vector_data: torch.Tensor,
        injection_factor: float = 0.3
    ) -> "VectorContextLayer":
        """
        Create a VectorContextLayer by wrapping an existing attention layer.
        
        Args:
            layer: Existing attention layer to wrap
            vector_data: Vector embeddings to inject
            injection_factor: Weight for vector context injection
            
        Returns:
            VectorContextLayer instance
        """
        # Extract parameters from the layer
        embedding_dim = layer.embed_dim if hasattr(layer, "embed_dim") else layer.hidden_size
        num_heads = layer.num_heads if hasattr(layer, "num_heads") else layer.num_attention_heads
        dropout = layer.dropout if hasattr(layer, "dropout") else 0.1
        bias = layer.in_proj_bias is not None if hasattr(layer, "in_proj_bias") else True
        
        # Create new layer
        return cls(
            base_layer=layer,
            vector_data=vector_data,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            injection_factor=injection_factor,
            device=next(layer.parameters()).device,
            dtype=next(layer.parameters()).dtype
        )
