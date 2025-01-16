import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn as nn
from scipy.sparse import random
from dataclasses import dataclass
from collections import deque

@dataclass
class LSMConfig:
    input_size: int = 64
    reservoir_size: int = 1000
    sparsity: float = 0.1
    spectral_radius: float = 0.9
    input_scaling: float = 1.0
    leak_rate: float = 0.3
    
class LiquidStateReservoir:
    def __init__(self, config: LSMConfig):
        self.config = config
        # Initialize sparse reservoir weights with specific spectral radius
        sparse_weights = random(config.reservoir_size, config.reservoir_size, 
                              density=config.sparsity)
        self.W = torch.from_numpy(sparse_weights.toarray()).float()
        
        # Scale weights to desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W.numpy())))
        self.W *= (config.spectral_radius / radius)
        
        # Input weights
        self.W_in = torch.randn(config.input_size, config.reservoir_size) * config.input_scaling
        
        # State initialization
        self.state = torch.zeros(config.reservoir_size)
        
    def step(self, input_signal: torch.Tensor) -> torch.Tensor:
        """Perform one step of reservoir computing."""
        # Update reservoir state
        preactivation = (torch.matmul(input_signal, self.W_in) + 
                        torch.matmul(self.state, self.W))
        new_state = torch.tanh(preactivation)
        
        # Apply leaky integration
        self.state = ((1 - self.config.leak_rate) * self.state + 
                     self.config.leak_rate * new_state)
        return self.state

class LSMInferenceEngine:
    def __init__(self, config: LSMConfig):
        self.reservoir = LiquidStateReservoir(config)
        self.state_buffer = deque(maxlen=100)  # Store recent states
        self.vector_store = {}  # Simple in-memory vector store
        
    def process_input(self, input_signal: torch.Tensor) -> Dict[str, Any]:
        """Process input through LSM and return inference results."""
        # Get reservoir state
        state = self.reservoir.step(input_signal)
        self.state_buffer.append(state)
        
        # Perform similarity search in vector store
        similarities = self._compute_similarities(state)
        
        # Return inference results
        return {
            'current_state': state,
            'top_matches': similarities[:5],  # Top 5 similar states
            'reservoir_activity': torch.mean(torch.abs(state)).item()
        }
    
    def _compute_similarities(self, state: torch.Tensor) -> List[Dict[str, float]]:
        """Compute similarities between current state and stored vectors."""
        similarities = []
        for label, stored_state in self.vector_store.items():
            similarity = torch.cosine_similarity(state.flatten(), 
                                              stored_state.flatten(), 
                                              dim=0)
            similarities.append({
                'label': label,
                'similarity': similarity.item()
            })
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    def store_reference_state(self, label: str, state: torch.Tensor):
        """Store a reference state in the vector database."""
        self.vector_store[label] = state

class RealTimeInferenceAPI:
    def __init__(self, config: LSMConfig):
        self.engine = LSMInferenceEngine(config)
        
    async def process_stream(self, input_stream: torch.Tensor) -> Dict[str, Any]:
        """Process streaming input and return real-time inference results."""
        results = self.engine.process_input(input_stream)
        return {
            'timestamp': torch.datetime.now(),
            'inference_results': results,
            'system_status': {
                'reservoir_size': self.engine.reservoir.state.shape[0],
                'stored_patterns': len(self.engine.vector_store)
            }
        }
    
    def train_reference_pattern(self, 
                              pattern: torch.Tensor, 
                              label: str):
        """Train the system on a reference pattern."""
        # Process pattern through reservoir
        state = self.engine.reservoir.step(pattern)
        # Store in vector database
        self.engine.store_reference_state(label, state)

# Example usage configuration
config = LSMConfig(
    input_size=64,
    reservoir_size=1000,
    sparsity=0.1,
    spectral_radius=0.9,
    input_scaling=1.0,
    leak_rate=0.3
)