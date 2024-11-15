import torch
import shap
import lime
import lime.lime_image
from captum.attr import (
    IntegratedGradients, 
    DeepLift,
    GradientShap,
    Occlusion,
    LayerAttribution
)
from captum.insights import AttributionVisualizer
import networkx as nx
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F

@dataclass
class ExplainabilityResult:
    """Container for explainability results"""
    feature_importance: np.ndarray
    attribution_scores: np.ndarray
    relationship_metrics: Dict
    visualization_data: Dict
    interpretation: str

class GraphEmbeddingExplainer:
    def __init__(self, model, graph_processor):
        """
        Initialize explainer with model and graph processor
        Args:
            model: DGLViTGraphNet model
            graph_processor: GraphImageProcessor instance
        """
        self.model = model
        self.graph_processor = graph_processor
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(model)
        self.deep_lift = DeepLift(model)
        self.gradient_shap = GradientShap(model)
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.DeepExplainer(model, torch.randn(1, 3, 224, 224))
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_image.LimeImageExplainer()

    def explain_node_relationships(self,
                                 graph: dgl.DGLGraph,
                                 embeddings: torch.Tensor,
                                 node_idx: int) -> ExplainabilityResult:
        """
        Explain relationships for a specific node
        """
        # Get node features and neighbors
        node_embedding = embeddings[node_idx]
        neighbor_indices = graph.predecessors(node_idx).numpy()
        
        # Compute attribution scores
        attr_scores = self._compute_attributions(node_embedding, neighbor_indices)
        
        # Analyze feature interactions
        feature_importance = self._analyze_feature_interactions(
            embeddings, graph, node_idx
        )
        
        # Compute relationship metrics
        rel_metrics = self._compute_relationship_metrics(
            embeddings, graph, node_idx
        )
        
        # Generate visualization data
        viz_data = self._prepare_visualization_data(
            attr_scores, feature_importance, rel_metrics
        )
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            attr_scores, feature_importance, rel_metrics
        )
        
        return ExplainabilityResult(
            feature_importance=feature_importance,
            attribution_scores=attr_scores,
            relationship_metrics=rel_metrics,
            visualization_data=viz_data,
            interpretation=interpretation
        )

    def explain_subgraph_structure(self,
                                 graph: dgl.DGLGraph,
                                 embeddings: torch.Tensor,
                                 node_indices: List[int]) -> Dict[str, ExplainabilityResult]:
        """
        Explain subgraph structure and relationships
        """
        results = {}
        
        # Get subgraph
        subgraph = dgl.node_subgraph(graph, node_indices)
        sub_embeddings = embeddings[node_indices]
        
        # Analyze subgraph structure
        for idx in node_indices:
            results[f"node_{idx}"] = self.explain_node_relationships(
                subgraph, sub_embeddings, idx
            )
            
        # Add subgraph-level explanations
        results["subgraph"] = self._explain_subgraph_patterns(
            subgraph, sub_embeddings
        )
        
        return results

    def _compute_attributions(self,
                            node_embedding: torch.Tensor,
                            neighbor_indices: np.ndarray) -> np.ndarray:
        """
        Compute attribution scores using multiple methods
        """
        # Integrated Gradients
        ig_attrs = self.integrated_gradients.attribute(
            node_embedding.unsqueeze(0),
            target=0,
            n_steps=50
        )
        
        # DeepLift
        dl_attrs = self.deep_lift.attribute(
            node_embedding.unsqueeze(0),
            target=0
        )
        
        # GradientSHAP
        gs_attrs = self.gradient_shap.attribute(
            node_embedding.unsqueeze(0),
            n_samples=50,
            stdevs=0.0001
        )
        
        # Combine attributions
        combined_attrs = (
            ig_attrs.detach().numpy() +
            dl_attrs.detach().numpy() +
            gs_attrs.detach().numpy()
        ) / 3
        
        return combined_attrs

    def _analyze_feature_interactions(self,
                                   embeddings: torch.Tensor,
                                   graph: dgl.DGLGraph,
                                   node_idx: int) -> np.ndarray:
        """
        Analyze feature interactions using SHAP and mutual information
        """
        # Get node neighborhood
        neighbors = graph.successors(node_idx).numpy()
        neighbor_embeddings = embeddings[neighbors]
        
        # Compute SHAP values
        shap_values = self.shap_explainer.shap_values(
            neighbor_embeddings.unsqueeze(1)
        )
        
        # Compute mutual information between features
        mi_matrix = np.zeros((embeddings.shape[1], embeddings.shape[1]))
        for i in range(embeddings.shape[1]):
            for j in range(embeddings.shape[1]):
                if i != j:
                    mi_matrix[i, j] = mutual_info_score(
                        embeddings[:, i].numpy(),
                        embeddings[:, j].numpy()
                    )
                    
        # Combine SHAP and MI insights
        feature_importance = np.mean(np.abs(shap_values), axis=0) * np.mean(mi_matrix, axis=1)
        
        return feature_importance

    def _compute_relationship_metrics(self,
                                   embeddings: torch.Tensor,
                                   graph: dgl.DGLGraph,
                                   node_idx: int) -> Dict:
        """
        Compute metrics describing node relationships
        """
        # Get node neighborhood
        neighbors = graph.successors(node_idx).numpy()
        neighbor_embeddings = embeddings[neighbors]
        
        # Compute embedding similarity metrics
        cosine_sim = F.cosine_similarity(
            embeddings[node_idx].unsqueeze(0),
            neighbor_embeddings
        )
        
        # Compute structural metrics
        nx_graph = dgl.to_networkx(graph)
        centrality = nx.eigenvector_centrality_numpy(nx_graph)
        clustering = nx.clustering(nx_graph)
        
        return {
            'cosine_similarities': cosine_sim.numpy(),
            'centrality': centrality[node_idx],
            'clustering': clustering[node_idx],
            'num_neighbors': len(neighbors)
        }

    def visualize_explanations(self,
                             result: ExplainabilityResult) -> go.Figure:
        """
        Create interactive visualization of explanations
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Feature Importance',
                'Attribution Scores',
                'Relationship Network',
                'Embedding Space'
            )
        )
        
        # Feature importance plot
        fig.add_trace(
            go.Bar(
                y=range(len(result.feature_importance)),
                x=result.feature_importance,
                orientation='h',
                name='Feature Importance'
            ),
            row=1, col=1
        )
        
        # Attribution scores plot
        fig.add_trace(
            go.Heatmap(
                z=result.attribution_scores,
                colorscale='RdBu',
                name='Attribution Scores'
            ),
            row=1, col=2
        )
        
        # Relationship network plot
        fig.add_trace(
            go.Scatter(
                x=result.visualization_data['network_layout_x'],
                y=result.visualization_data['network_layout_y'],
                mode='markers+text',
                text=result.visualization_data['node_labels'],
                name='Network'
            ),
            row=2, col=1
        )
        
        # Embedding space plot
        fig.add_trace(
            go.Scatter3d(
                x=result.visualization_data['embedding_coords'][:, 0],
                y=result.visualization_data['embedding_coords'][:, 1],
                z=result.visualization_data['embedding_coords'][:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=result.visualization_data['embedding_colors'],
                    colorscale='Viridis'
                ),
                name='Embeddings'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=False,
            title_text="Graph-Embedding Relationship Explanation"
        )
        
        return fig

    def generate_explanation_report(self,
                                  result: ExplainabilityResult) -> str:
        """
        Generate detailed textual explanation
        """
        report = [
            "# Graph-Embedding Relationship Analysis",
            "\n## Feature Importance Analysis",
            f"Top features: {self._get_top_features(result.feature_importance)}",
            "\n## Attribution Analysis",
            f"Key attributions: {self._summarize_attributions(result.attribution_scores)}",
            "\n## Relationship Metrics",
            self._format_relationship_metrics(result.relationship_metrics),
            "\n## Interpretation",
            result.interpretation
        ]
        
        return "\n".join(report)

    def _explain_subgraph_patterns(self,
                                 subgraph: dgl.DGLGraph,
                                 embeddings: torch.Tensor) -> ExplainabilityResult:
        """
        Explain patterns in subgraph
        """
        # Convert to networkx for pattern analysis
        nx_graph = dgl.to_networkx(subgraph)
        
        # Find motifs and patterns
        motifs = self._find_graph_motifs(nx_graph)
        communities = self._detect_communities(nx_graph)
        
        # Analyze embedding clusters
        clusters = self._analyze_embedding_clusters(embeddings)
        
        # Generate explanation
        feature_importance = self._analyze_feature_interactions(
            embeddings, subgraph, 0  # Use first node as reference
        )
        
        # Prepare visualization data
        viz_data = {
            'motifs': motifs,
            'communities': communities,
            'clusters': clusters
        }
        
        # Generate interpretation
        interpretation = self._interpret_subgraph_patterns(
            motifs, communities, clusters
        )
        
        return ExplainabilityResult(
            feature_importance=feature_importance,
            attribution_scores=np.zeros((1, 1)),  # Placeholder
            relationship_metrics={'patterns': len(motifs)},
            visualization_data=viz_data,
            interpretation=interpretation
        )

def main():
    # Initialize components from previous code
    model = DGLViTGraphNet(num_classes=1000)
    processor = GraphImageProcessor()
    explainer = GraphEmbeddingExplainer(model, processor)
    
    # Process sample data
    images = [Image.open(f"image{i}.jpg") for i in range(5)]
    graph, predictions = processor.process_image_batch(images)
    embeddings = model.extract_vit_features(images)
    
    # Generate explanations
    node_explanation = explainer.explain_node_relationships(
        graph, embeddings, node_idx=0
    )
    
    # Visualize explanations
    fig = explainer.visualize_explanations(node_explanation)
    fig.show()
    
    # Generate report
    report = explainer.generate_explanation_report(node_explanation)
    print(report)

if __name__ == "__main__":
    main()