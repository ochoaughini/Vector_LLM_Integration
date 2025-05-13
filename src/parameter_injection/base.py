"""
Base implementation for Parameter Injection pattern with LLMs.

This module provides the ParameterInjectionModel class that directly modifies
LLM parameters using vector-derived features for improved inference speeds.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel

from vector_llm_integration.parameter_injection.vector_context_layer import VectorContextLayer
from vector_llm_integration.observability.monitoring import trace_vectordb_operation

logger = logging.getLogger(__name__)

class ParameterInjectionModel:
    """
    Parameter Injection model that integrates vector database knowledge into LLM parameters.
    
    This advanced approach achieves significant inference speed improvements by directly
    modifying the transformer's attention layers with vector-derived features.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        model_type: str = "encoder",  # "encoder" or "causal"
        embedding_dim: Optional[int] = None,
        layer_injection_indexes: Optional[List[int]] = None,
        injection_factor: float = 0.3,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the Parameter Injection model.
        
        Args:
            model_name: Name of the pretrained model to use
            model_type: Type of the model ('encoder' or 'causal')
            embedding_dim: Dimension of embeddings (if None, derived from model)
            layer_injection_indexes: Which layers to inject vectors into (if None, use all)
            injection_factor: Weight factor for vector context injection (0-1)
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.injection_factor = injection_factor
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load base model
        logger.info(f"Loading base model {model_name}")
        if model_type == "encoder":
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        elif model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Extract model structure information
        self.embedding_dim = embedding_dim if embedding_dim is not None else self._get_embedding_dim()
        self.layer_structure = self._analyze_model_structure()
        
        # Default to all attention layers if not specified
        if layer_injection_indexes is None:
            layer_injection_indexes = list(range(len(self.layer_structure["attention_layers"])))
        self.layer_injection_indexes = layer_injection_indexes
        
        # Initialize vector data
        self.vector_data = None
        
        logger.info(f"Initialized parameter injection model based on {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Target layers for injection: {layer_injection_indexes}")
        
    def _get_embedding_dim(self) -> int:
        """
        Get the embedding dimension from the model.
        
        Returns:
            Embedding dimension
        """
        # Check common attribute names for different model types
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, "d_model"):
            return self.model.config.d_model
        elif hasattr(self.model.config, "n_embd"):
            return self.model.config.n_embd
        else:
            # Default fallback
            logger.warning("Could not determine embedding dimension, defaulting to 768")
            return 768
    
    def _analyze_model_structure(self) -> Dict[str, Any]:
        """
        Analyze the model structure to identify attention layers.
        
        Returns:
            Dictionary with model structure information
        """
        structure = {
            "attention_layers": [],
            "model_type": self.model_type,
        }
        
        if self.model_type == "encoder":
            # BERT and similar models
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                for i, layer in enumerate(self.model.encoder.layer):
                    if hasattr(layer, "attention"):
                        if hasattr(layer.attention, "self"):
                            structure["attention_layers"].append({
                                "index": i,
                                "path": f"encoder.layer.{i}.attention.self",
                                "module": layer.attention.self
                            })
        elif self.model_type == "causal":
            # GPT-like models - structure varies significantly between models
            # This is a simplification - may need adaptation for specific architectures
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                # GPT-2 style
                for i, block in enumerate(self.model.transformer.h):
                    if hasattr(block, "attn"):
                        structure["attention_layers"].append({
                            "index": i,
                            "path": f"transformer.h.{i}.attn",
                            "module": block.attn
                        })
            elif hasattr(self.model, "layers"):
                # Some newer models
                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, "self_attn"):
                        structure["attention_layers"].append({
                            "index": i,
                            "path": f"layers.{i}.self_attn",
                            "module": layer.self_attn
                        })
        
        logger.info(f"Found {len(structure['attention_layers'])} attention layers")
        return structure
    
    @trace_vectordb_operation("parameter_injection", "load_vector_data")
    def load_vector_data(
        self,
        vector_data: Union[torch.Tensor, str, List[List[float]]],
        vector_dim: Optional[int] = None
    ) -> bool:
        """
        Load vector data for parameter injection.
        
        Args:
            vector_data: Vector embeddings as tensor, file path, or nested list
            vector_dim: Dimension of vectors (required if loading from file)
            
        Returns:
            Boolean indicating success
        """
        try:
            if isinstance(vector_data, torch.Tensor):
                # Already a tensor
                self.vector_data = vector_data.to(self.device)
                logger.info(f"Loaded vector data tensor with shape {self.vector_data.shape}")
                
            elif isinstance(vector_data, str) and os.path.exists(vector_data):
                # Path to a file
                if vector_data.endswith(".pt") or vector_data.endswith(".pth"):
                    # PyTorch tensor file
                    self.vector_data = torch.load(vector_data, map_location=self.device)
                    logger.info(f"Loaded vector data from {vector_data} with shape {self.vector_data.shape}")
                    
                elif vector_data.endswith(".npy"):
                    # NumPy array file
                    import numpy as np
                    np_data = np.load(vector_data)
                    self.vector_data = torch.tensor(np_data, device=self.device)
                    logger.info(f"Loaded vector data from NumPy file {vector_data} with shape {self.vector_data.shape}")
                    
                else:
                    # Assume text file with one vector per line
                    if vector_dim is None:
                        raise ValueError("vector_dim must be provided when loading from text file")
                        
                    vectors = []
                    with open(vector_data, "r") as f:
                        for line in f:
                            values = [float(x) for x in line.strip().split()]
                            if len(values) != vector_dim:
                                logger.warning(f"Expected {vector_dim} dimensions, got {len(values)}")
                                continue
                            vectors.append(values)
                    
                    self.vector_data = torch.tensor(vectors, device=self.device)
                    logger.info(f"Loaded {len(vectors)} vectors from text file {vector_data}")
                    
            elif isinstance(vector_data, list):
                # List of vectors
                self.vector_data = torch.tensor(vector_data, device=self.device)
                logger.info(f"Converted list to vector data tensor with shape {self.vector_data.shape}")
                
            else:
                logger.error(f"Unsupported vector data format: {type(vector_data)}")
                return False
                
            # Verify vector dimensions
            if self.vector_data.dim() != 2:
                raise ValueError(f"Expected 2D tensor, got {self.vector_data.dim()}D")
                
            logger.info(f"Successfully loaded vector data with shape {self.vector_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector data: {e}")
            return False
    
    @trace_vectordb_operation("parameter_injection", "inject_vectors")
    def inject_vectors(self) -> bool:
        """
        Inject vector knowledge into the model's attention layers.
        
        This is the core of the Parameter Injection pattern, directly modifying
        the model's attention mechanisms with vector knowledge.
        
        Returns:
            Boolean indicating success
        """
        if self.vector_data is None:
            logger.error("No vector data loaded. Call load_vector_data first.")
            return False
            
        # Count how many layers we've modified
        injected_count = 0
        
        try:
            for idx in self.layer_injection_indexes:
                if idx >= len(self.layer_structure["attention_layers"]):
                    logger.warning(f"Layer index {idx} out of range, skipping")
                    continue
                    
                layer_info = self.layer_structure["attention_layers"][idx]
                attn_module = layer_info["module"]
                
                # Create a VectorContextLayer wrapping the original attention layer
                vector_layer = VectorContextLayer.from_layer(
                    layer=attn_module,
                    vector_data=self.vector_data,
                    injection_factor=self.injection_factor
                )
                
                # Replace the original module with our enhanced version
                # This requires navigating the model's module hierarchy
                path_parts = layer_info["path"].split(".")
                parent_module = self.model
                for part in path_parts[:-1]:
                    if part.isdigit():
                        parent_module = parent_module[int(part)]
                    else:
                        parent_module = getattr(parent_module, part)
                
                last_part = path_parts[-1]
                if last_part.isdigit():
                    parent_module[int(last_part)] = vector_layer
                else:
                    setattr(parent_module, last_part, vector_layer)
                
                injected_count += 1
                logger.info(f"Injected vector context into layer {idx}")
            
            logger.info(f"Successfully injected vectors into {injected_count} layers")
            return injected_count > 0
            
        except Exception as e:
            logger.error(f"Error injecting vectors: {e}")
            return False
    
    @trace_vectordb_operation("parameter_injection", "save_model")
    def save_model(self, path: str) -> bool:
        """
        Save the enhanced model with injected vectors.
        
        Args:
            path: Path to save the model
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save model and configuration
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(path)
                logger.info(f"Saved enhanced model to {path}")
                
                # Save vector data alongside the model
                vector_path = os.path.join(path, "vector_data.pt")
                torch.save(self.vector_data, vector_path)
                logger.info(f"Saved vector data to {vector_path}")
                
                # Save injection configuration
                config_path = os.path.join(path, "injection_config.py")
                with open(config_path, "w") as f:
                    f.write(f"""
# Vector Injection Configuration
INJECTION_FACTOR = {self.injection_factor}
LAYER_INJECTION_INDEXES = {self.layer_injection_indexes}
EMBEDDING_DIM = {self.embedding_dim}
MODEL_TYPE = "{self.model_type}"
BASE_MODEL = "{self.model_name}"
                    """)
                logger.info(f"Saved injection configuration to {config_path}")
                
                return True
            else:
                logger.error("Model does not support save_pretrained method")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None) -> "ParameterInjectionModel":
        """
        Load a previously saved model with injected vectors.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            ParameterInjectionModel instance
        """
        try:
            # Load injection configuration
            config_path = os.path.join(path, "injection_config.py")
            config = {}
            with open(config_path, "r") as f:
                exec(f.read(), config)
            
            # Create model instance
            model = cls(
                model_name=path,  # Use the path directly for loading
                model_type=config.get("MODEL_TYPE", "encoder"),
                embedding_dim=config.get("EMBEDDING_DIM"),
                layer_injection_indexes=config.get("LAYER_INJECTION_INDEXES"),
                injection_factor=config.get("INJECTION_FACTOR", 0.3),
                device=device
            )
            
            # Load vector data
            vector_path = os.path.join(path, "vector_data.pt")
            model.load_vector_data(vector_path)
            
            logger.info(f"Successfully loaded model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise
    
    @trace_vectordb_operation("parameter_injection", "inference")
    def inference(
        self, 
        inputs: Union[str, List[str], torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Any:
        """
        Run inference with the enhanced model.
        
        Args:
            inputs: Input to the model (text, batch of text, or tensors)
            
        Returns:
            Model outputs
        """
        try:
            # Ensure model is in evaluation mode
            self.model.eval()
            
            # Process input based on type
            if isinstance(inputs, str):
                # Single text input - need to tokenize
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                encoded_inputs = tokenizer(inputs, return_tensors="pt").to(self.device)
                
            elif isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
                # Batch of text inputs - need to tokenize
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
                
            elif isinstance(inputs, torch.Tensor):
                # Already tensor input (assumed to be input_ids)
                encoded_inputs = {"input_ids": inputs.to(self.device)}
                
            elif isinstance(inputs, dict) and "input_ids" in inputs:
                # Dictionary with tensors
                encoded_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                
            return outputs
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
