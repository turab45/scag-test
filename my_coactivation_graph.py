import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import rankdata
from typing import Tuple
import os


class CoactivationGraphBuilder:
    """
    Builds coactivation graphs from neural network pooled activations.
    Handles correlation computation and graph construction.
    """
    
    def __init__(self, pooled_activations: np.ndarray):
        """
        Initialize with pooled activations.
        
        Args:
            pooled_activations: Array of shape (N_images, N_neurons) or similar
        """
        self.raw_activations = pooled_activations
        self.pooled_matrix = None
        self.top_neuron_indices = None
        self.pooled_matrix_sampled = None
        self.ranked_matrix = None
        self.corr_matrix = None
    
    def prepare_activation_matrix(self) -> Tuple[np.ndarray, int, int]:
        """
        Build pooled activation matrix: (N_neurons × N_images)
        
        Returns:
            Tuple of (pooled_matrix, n_neurons, n_images)
        """
        # Stack and transpose to (N_neurons, N_images)
        self.pooled_matrix = np.stack(self.raw_activations, axis=0).T
        n_neurons, n_images = self.pooled_matrix.shape
        return self.pooled_matrix, n_neurons, n_images
    
    def select_variable_neurons(self, top_n: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top variable neurons for correlation analysis (reduces computation).
        
        Args:
            top_n: Number of top neurons to select. If None, uses all neurons.
        
        Returns:
            Tuple of (top_neuron_indices, neuron_variances)
        """
        if self.pooled_matrix is None:
            raise ValueError("Call prepare_activation_matrix() first")
        
        # Compute variance across images for each neuron
        neuron_variances = np.var(self.pooled_matrix, axis=1)
        
        # Sort by variance (descending)
        self.top_neuron_indices = np.argsort(neuron_variances)[::-1]
        
        # Optionally limit to top_n
        if top_n is not None:
            self.top_neuron_indices = self.top_neuron_indices[:top_n]
        
        # Sample pooled matrix using selected neurons
        self.pooled_matrix_sampled = self.pooled_matrix[self.top_neuron_indices]
        
        n_selected = len(self.top_neuron_indices)
        print(f'\nUsing top {n_selected} most variable neurons for coactivation analysis')
        print(f'  → Variance range: [{neuron_variances[self.top_neuron_indices].min():.4f}, '
              f'{neuron_variances[self.top_neuron_indices].max():.4f}]')
        
        return self.top_neuron_indices, neuron_variances
    
    def compute_spearman_correlation(self, top_n: int = None) -> np.ndarray:
        """
        Compute Spearman correlation matrix from ranked activations.
        
        Spearman correlation = Pearson correlation computed on rank-transformed data
        
        Returns:
            Correlation matrix of shape (n_neurons_sampled, n_neurons_sampled)
        """
        self.prepare_activation_matrix()
        self.select_variable_neurons(top_n=top_n)
    
        # Step 1: Compute ranks (Spearman = Pearson on ranks)
        print('\nComputing ranks...')
        self.ranked_matrix = np.apply_along_axis(rankdata, 1, self.pooled_matrix_sampled)
        
        # Step 2: Pearson on ranked data = Spearman correlation
        print('Computing Spearman correlation matrix...')
        self.corr_matrix = np.corrcoef(self.ranked_matrix)
        
        return self.corr_matrix
    
    def print_correlation_stats(self) -> None:
        """Print statistics about the correlation matrix."""
        if self.corr_matrix is None:
            raise ValueError("Call compute_spearman_correlation() first")
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = self.corr_matrix[np.triu_indices_from(self.corr_matrix, k=1)]
        
        print(f'\nCorrelation matrix shape: {self.corr_matrix.shape}')
        print(f'  Min correlation: {self.corr_matrix.min():.3f}')
        print(f'  Max correlation: {self.corr_matrix.max():.3f}')
        print(f'  Mean correlation: {upper_triangle.mean():.3f}')
        print(f'  Std correlation:  {upper_triangle.std():.3f}')
        print(f'  Median correlation: {np.median(upper_triangle):.3f}')
    
    def build_graph(self, threshold: float = 0.3) -> nx.Graph:
        """
        Build NetworkX graph from correlation matrix.
        
        Args:
            threshold: Minimum correlation strength for adding edges
        
        Returns:
            NetworkX Graph with neurons as nodes and correlations as weighted edges
        """
        if self.corr_matrix is None:
            raise ValueError("Call compute_spearman_correlation() first")
        
        G = nx.Graph()
        
        # Add nodes for each neuron
        for i, neuron_id in enumerate(self.top_neuron_indices):
            G.add_node(i, original_id=neuron_id)
        
        # Add edges for correlations above threshold
        n_neurons = self.corr_matrix.shape[0]
        edge_count = 0
        
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                corr_val = self.corr_matrix[i, j]
                if corr_val > threshold:
                    G.add_edge(i, j, weight=corr_val, correlation=corr_val)
                    edge_count += 1
        
        print(f'\nGraph with threshold={threshold}:')
        print(f'  Nodes: {G.number_of_nodes()}')
        print(f'  Edges: {G.number_of_edges()}')
        print(f'  Density: {nx.density(G):.4f}')
        
        return G
    
    def save_graph(self, graph: nx.Graph, output_path: str, filename: str) -> None:
        """
        Save graph to GraphML format.
        
        Args:
            graph: NetworkX graph object
            output_path: Directory path for saving
            filename: Filename (without path)
        """
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        nx.write_graphml(graph, full_path)
        print(f'Graph saved to: {full_path}')
    
    def apply_threshold(self, graph: nx.Graph, threshold: float) -> np.ndarray:
        """
        Apply threshold to the graph and return the new graph with only edges above the threshold.
        """
        edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d['weight'] <= threshold]
        graph.remove_edges_from(edges_to_remove)
        print(f'Applied threshold: {threshold}, removed {len(edges_to_remove)} edges')
        return graph

    def load_graph(self, input_path: str, filename: str) -> nx.Graph:
        """
        Load graph from GraphML format.
        
        Args:
            input_path: Directory path for loading
            filename: Filename (without path)
        Returns:
            Loaded NetworkX graph object
        """
        full_path = os.path.join(input_path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Graph file not found: {full_path}")
        
        graph = nx.read_graphml(full_path)
        print(f'Graph loaded from: {full_path}')
        return graph
    
    def save_correlation_matrix(self, output_path: str, filename: str) -> None:
        """
        Save correlation matrix to CSV.
        
        Args:
            output_path: Directory path for saving
            filename: Filename (without path)
        """
        if self.corr_matrix is None:
            raise ValueError("Call compute_spearman_correlation() first")
        
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        pd.DataFrame(self.corr_matrix).to_csv(full_path, index=False, header=False)
        print(f'Correlation matrix saved to: {full_path}')
    
    def run_pipeline(self, threshold: float = 0.3, top_n: int = None, 
                    output_path: str = './results/', graph_filename: str = 'coactivation_graph.graphml',
                    corr_filename: str = 'correlation_matrix.csv') -> Tuple[nx.Graph, np.ndarray]:
        """
        Execute the complete pipeline: prepare → select → correlate → build → save
        
        Args:
            threshold: Correlation threshold for graph edges
            top_n: Number of top variable neurons to use (None = all)
            output_path: Directory for saving outputs
            graph_filename: Name of output GraphML file
            corr_filename: Name of output correlation matrix CSV
        
        Returns:
            Tuple of (graph, correlation_matrix)
        """
        print("=" * 80)
        print("COACTIVATION GRAPH PIPELINE")
        print("=" * 80)
        
        self.compute_spearman_correlation(top_n=top_n)
        self.print_correlation_stats()
        
        graph = self.build_graph(threshold=threshold)
        
        self.save_graph(graph, output_path, graph_filename)
        self.save_correlation_matrix(output_path, corr_filename)
        
        print("=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        
        return graph, self.corr_matrix


# Example usage:
# builder = CoactivationGraphBuilder(pooled_activations)
# graph, corr_matrix = builder.run_pipeline(threshold=0.3, 
#                                            output_path='./results_mnm2/',
#                                            graph_filename='coactivation_graph_densenet161_mnm2.graphml',
#                                            corr_filename='correlation_matrix_densenet161_mnm2.csv')
