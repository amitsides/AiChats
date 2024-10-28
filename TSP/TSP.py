import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class TSPEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        super(TSPEmbeddingNet, self).__init__()
        
        # City embedding network
        self.city_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2D city coordinates -> hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)  # hidden -> embedding space
        )
        
        # Attention mechanism for city selection
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Path decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def embed_cities(self, cities: torch.Tensor) -> torch.Tensor:
        """Convert city coordinates into learned embeddings."""
        return self.city_encoder(cities)
    
    def compute_attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute attention scores for each city."""
        return self.attention(embeddings).squeeze(-1)
    
    def forward(self, cities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_cities, _ = cities.shape
        
        # Embed cities
        embeddings = self.embed_cities(cities)  # [batch, cities, embedding_dim]
        
        # Calculate attention scores
        attention = self.compute_attention(embeddings)  # [batch, cities]
        attention = torch.softmax(attention, dim=1)
        
        # Generate path probabilities using attention
        path_probs = torch.zeros(batch_size, num_cities, num_cities)
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    combined = torch.cat([embeddings[:, i], embeddings[:, j]], dim=-1)
                    path_probs[:, i, j] = self.decoder(combined).squeeze(-1)
        
        return attention, path_probs

class NPHardSolver:
    def __init__(self, embedding_dim: int = 128):
        self.model = TSPEmbeddingNet(embedding_dim=embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        
    def generate_tsp_instance(self, num_cities: int) -> torch.Tensor:
        """Generate random TSP instance."""
        return torch.rand(1, num_cities, 2)  # [batch=1, cities, coordinates]
    
    def compute_tour_length(self, cities: torch.Tensor, tour: List[int]) -> float:
        """Calculate total tour length."""
        total_distance = 0
        for i in range(len(tour)):
            city1 = cities[0, tour[i]]
            city2 = cities[0, tour[(i + 1) % len(tour)]]
            distance = torch.sqrt(torch.sum((city1 - city2) ** 2))
            total_distance += distance
        return total_distance.item()
    
    def train_step(self, cities: torch.Tensor, num_steps: int = 100):
        """Train the model on a TSP instance."""
        for step in range(num_steps):
            self.optimizer.zero_grad()
            
            # Forward pass
            attention, path_probs = self.model(cities)
            
            # Compute loss (combination of tour length and validity constraints)
            # This is a simplified loss - in practice you'd want more sophisticated constraints
            tour_loss = torch.mean(path_probs)  # Encourage shorter tours
            constraint_loss = torch.abs(torch.sum(attention) - 1)  # Valid tour constraint
            
            loss = tour_loss + 10.0 * constraint_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
    
    def solve(self, cities: torch.Tensor) -> List[int]:
        """Generate solution for TSP instance."""
        self.model.eval()
        with torch.no_grad():
            attention, path_probs = self.model(cities)
            
            # Greedy path construction using attention and path probabilities
            current_city = 0
            tour = [current_city]
            unvisited = set(range(1, cities.shape[1]))
            
            while unvisited:
                next_probs = path_probs[0, current_city].clone()
                # Mask visited cities
                for visited in tour:
                    next_probs[visited] = -float('inf')
                
                current_city = torch.argmax(next_probs).item()
                tour.append(current_city)
                unvisited.remove(current_city)
                
        return tour

# Example usage
def solve_tsp_instance(num_cities: int = 10):
    solver = NPHardSolver(embedding_dim=128)
    
    # Generate random TSP instance
    cities = solver.generate_tsp_instance(num_cities)
    
    # Train model on this instance
    solver.train_step(cities, num_steps=200)
    
    # Get solution
    tour = solver.solve(cities)
    tour_length = solver.compute_tour_length(cities, tour)
    
    return tour, tour_length


# Solve a 10-city TSP instance
tour, length = solve_tsp_instance(num_cities=10)
print(f"Found tour: {tour}")
print(f"Tour length: {length:.2f}")