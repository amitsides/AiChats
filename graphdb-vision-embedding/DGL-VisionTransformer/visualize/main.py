import torch
import dgl
import networkx as nx
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import community
import pandas as pd
from umap import UMAP
import logging
import warnings
from tqdm import tqdm

class AdvancedGraphVisualizer:
    def __init__(self, dim_reduction_method: str = 'tsne'):
        """
        Initialize visualizer with dimension reduction method
        Args:
            dim_reduction_method: 'tsne' or 'umap'
        """
        self.dim_reduction_method = dim_reduction_method
        self.tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        self.umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def reduce_dimensions(self, 
                         embeddings: np.ndarray,
                         method: Optional[str] = None) -> np.ndarray:
        """
        Reduce dimensionality of embeddings using t-SNE or UMAP
        """
        method = method or self.dim_reduction_method
        
        try:
            if method == 'tsne':
                return self.tsne.fit_transform(embeddings)
            elif method == 'umap':
                return self.umap.fit_transform(embeddings)
            else:
                raise ValueError(f"Unknown dimension reduction method: {method}")
        except Exception as e:
            self.logger.error(f"Error in dimension reduction: {str(e)}")
            raise

    def create_interactive_embedding_plot(self,
                                       embeddings: np.ndarray,
                                       labels: Optional[List] = None,
                                       metadata: Optional[Dict] = None) -> go.Figure:
        """
        Create interactive plot of embeddings with metadata
        """
        # Reduce dimensions for visualization
        reduced_embeddings = self.reduce_dimensions(embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': labels if labels else [''] * len(reduced_embeddings)
        })
        
        if metadata:
            for key, values in metadata.items():
                df[key] = values
        
        # Create interactive scatter plot
        fig = px.scatter(df, x='x', y='y', color='label',
                        hover_data=list(metadata.keys()) if metadata else None,
                        title=f'Embedding Visualization ({self.dim_reduction_method.upper()})')
        
        return fig

class GraphMetricsAnalyzer:
    def __init__(self):
        """
        Initialize graph metrics analyzer
        """
        self.metrics_cache = {}
        self.logger = logging.getLogger(__name__)

    def compute_advanced_metrics(self, 
                               graph: dgl.DGLGraph,
                               embeddings: np.ndarray) -> Dict:
        """
        Compute advanced graph and embedding metrics
        """
        # Convert to networkx for some calculations
        nx_graph = dgl.to_networkx(graph)
        
        metrics = {
            'basic': self._compute_basic_metrics(nx_graph),
            'centrality': self._compute_centrality_metrics(nx_graph),
            'community': self._compute_community_metrics(nx_graph),
            'embedding': self._compute_embedding_metrics(embeddings)
        }
        
        return metrics

    def _compute_basic_metrics(self, nx_graph: nx.Graph) -> Dict:
        """
        Compute basic graph metrics
        """
        return {
            'density': nx.density(nx_graph),
            'transitivity': nx.transitivity(nx_graph),
            'average_clustering': nx.average_clustering(nx_graph),
            'diameter': nx.diameter(nx.Graph(nx_graph)),
            'average_shortest_path': nx.average_shortest_path_length(nx.Graph(nx_graph))
        }

    def _compute_centrality_metrics(self, nx_graph: nx.Graph) -> Dict:
        """
        Compute various centrality metrics
        """
        return {
            'degree_centrality': nx.degree_centrality(nx_graph),
            'betweenness_centrality': nx.betweenness_centrality(nx_graph),
            'eigenvector_centrality': nx.eigenvector_centrality(nx_graph, max_iter=1000),
            'pagerank': nx.pagerank(nx_graph)
        }

    def _compute_community_metrics(self, nx_graph: nx.Graph) -> Dict:
        """
        Compute community detection metrics
        """
        communities = community.best_partition(nx_graph)
        return {
            'communities': communities,
            'modularity': community.modularity(communities, nx_graph),
            'num_communities': len(set(communities.values()))
        }

    def _compute_embedding_metrics(self, embeddings: np.ndarray) -> Dict:
        """
        Compute embedding space metrics
        """
        # Compute pairwise distances
        distances = cdist(embeddings, embeddings)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'num_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0),
            'isolation_score': np.mean(clustering.labels_ == -1) if len(clustering.labels_) > 0 else 0
        }

class EnhancedGraphVisualizer:
    def __init__(self):
        """
        Initialize enhanced graph visualizer
        """
        self.layout_algorithms = {
            'spring': nx.spring_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout
        }
        self.logger = logging.getLogger(__name__)

    def create_interactive_graph_visualization(self,
                                            graph: dgl.DGLGraph,
                                            embeddings: np.ndarray,
                                            node_labels: Optional[List] = None,
                                            layout: str = 'spring') -> go.Figure:
        """
        Create interactive graph visualization with embeddings
        """
        # Convert to networkx
        nx_graph = dgl.to_networkx(graph)
        
        # Get layout
        layout_func = self.layout_algorithms.get(layout, nx.spring_layout)
        pos = layout_func(nx_graph)
        
        # Create edge trace
        edge_x, edge_y = [], []
        for edge in nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_x = [pos[node][0] for node in nx_graph.nodes()]
        node_y = [pos[node][1] for node in nx_graph.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        # Add node data
        node_adjacencies = []
        node_text = []
        for node in nx_graph.nodes():
            adjacencies = list(nx_graph.neighbors(node))
            node_adjacencies.append(len(adjacencies))
            node_text.append(f'Node {node}<br>Connections: {len(adjacencies)}')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           title=f'Interactive Graph Visualization ({layout} layout)',
                           annotations=[dict(
                               text="",
                               showarrow=False,
                               xref="paper", yref="paper"
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig

def visualize_graph_embeddings(graph: dgl.DGLGraph,
                             embeddings: np.ndarray,
                             labels: Optional[List] = None,
                             metadata: Optional[Dict] = None) -> Tuple[go.Figure, Dict]:
    """
    Create comprehensive visualization of graph and embeddings
    """
    # Initialize components
    emb_viz = AdvancedGraphVisualizer()
    graph_viz = EnhancedGraphVisualizer()
    metrics_analyzer = GraphMetricsAnalyzer()
    
    # Compute metrics
    metrics = metrics_analyzer.compute_advanced_metrics(graph, embeddings)
    
    # Create visualizations
    emb_plot = emb_viz.create_interactive_embedding_plot(embeddings, labels, metadata)
    graph_plot = graph_viz.create_interactive_graph_visualization(graph, embeddings)
    
    # Create combined visualization
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Embedding Space', 'Graph Structure'))
    
    # Add embedding plot
    for trace in emb_plot.data:
        fig.add_trace(trace, row=1, col=1)
        
    # Add graph plot
    for trace in graph_plot.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Update layout
    fig.update_layout(height=600, width=1200, title_text="Graph and Embedding Analysis")
    
    return fig, metrics

# Example usage
def main():
    processor = GraphImageProcessor()  # From previous code
    
    # Process images and get graph and embeddings
    images = [Image.open(f"image{i}.jpg") for i in range(5)]
    graph, predictions = processor.process_image_batch(images)
    
    # Extract embeddings from the model
    embeddings = processor.model.extract_vit_features(images).cpu().numpy()
    
    # Create visualizations and compute metrics
    fig, metrics = visualize_graph_embeddings(
        graph=graph,
        embeddings=embeddings,
        labels=[f"Image {i}" for i in range(len(images))],
        metadata={"filename": [f"image{i}.jpg" for i in range(len(images))]}
    )
    
    # Show interactive plot
    fig.show()
    
    # Print metrics
    print("\nGraph Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()