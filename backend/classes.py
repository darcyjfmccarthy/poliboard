import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.manifold import TSNE
import umap.umap_ as UMAP
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from collections import defaultdict

class PokemonTeamClustering:
    def __init__(self, df):
        """
        Initialize with a dataframe where each row is a team and columns are binary indicators
        for Pokemon and moves.
        """
        self.df = df[[col for col in df.columns[5:]]]
        self.pokemon_cols = [col for col in df.columns[5:-2]]
        self.move_cols = [col for col in df.columns[-2:]]
    
    def select_features(self, method='variance', threshold=0.01):
        """
        Select features based on different criteria
        
        Parameters:
        method: str
            'variance' - Remove low variance features
            'correlation' - Remove highly correlated features
            'frequency' - Keep only frequently used Pokemon/moves
        threshold: float
            Threshold for feature selection
        """
        if method == 'variance':
            # Remove features that barely vary (mostly 0s or mostly 1s)
            variances = self.df.var()
            selected_features = variances[variances > threshold].index
            self.df = self.df[selected_features]
            
        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = self.df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 1 - threshold)]
            self.df = self.df.drop(to_drop, axis=1)
            
        elif method == 'frequency':
            # Keep only Pokemon/moves that appear in more than threshold% of teams
            frequencies = self.df.mean()
            selected_features = frequencies[frequencies > threshold].index
            self.df = self.df[selected_features]
            
        # Update pokemon_cols and move_cols
        self.pokemon_cols = [col for col in self.df.columns if not col.startswith('move_')]
        self.move_cols = [col for col in self.df.columns if col.startswith('move_')]
        
        return self.df.columns.tolist()

    def normalize_features(self, pokemon_weight=1.0, move_weight=0.5):
        """
        Apply different weights to Pokemon vs move features
        """
        for col in self.df.columns:
            if col in self.pokemon_cols:
                self.df[col] = self.df[col] * pokemon_weight
            elif col in self.move_cols:
                self.df[col] = self.df[col] * move_weight
        
        return self.df
    
    def compute_feature_importance(self):
        """
        Compute importance of each feature based on cluster discrimination
        """
        if self.labels_ is None:
            self.labels_ = self.cluster_teams()
            
        importance_scores = {}
        
        for feature in self.df.columns:
            # Calculate how well this feature separates clusters
            cluster_means = [
                self.df[self.labels_ == i][feature].mean()
                for i in sorted(set(self.labels_)) if i != -1
            ]
            # Use standard deviation of means as importance score
            importance_scores[feature] = np.std(cluster_means)
            
        return pd.Series(importance_scores).sort_values(ascending=False)
        
    def find_optimal_clusters(self, max_clusters=20):
        """
        Use silhouette score to suggest optimal number of clusters
        """
        scores = []
        for n in range(2, max_clusters + 1):
            clustering = AgglomerativeClustering(
                n_clusters=n,
                affinity='cosine',
                linkage='complete'
            )
            labels = clustering.fit_predict(self.df)
            score = silhouette_score(self.df, labels, metric='cosine')
            scores.append((n, score))
        
        return pd.DataFrame(scores, columns=['n_clusters', 'silhouette_score'])
    
    def cluster_teams(self, method='hierarchical', n_clusters=10, eps=0.3, min_samples=5):
        """
        Cluster teams using specified method
        """
        if method == 'hierarchical':
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='cosine',
                linkage='complete'
            )
        elif method == 'dbscan':
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine'
            )
        else:
            raise ValueError("Method must be 'hierarchical' or 'dbscan'")
            
        self.labels_ = clustering.fit_predict(self.df)
        return self.labels_
    
    def analyze_clusters(self):
        """
        Analyze characteristics of each cluster with exact matching
        """
        analysis = defaultdict(dict)
        
        for cluster_id in set(self.labels_):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            # Get teams in this cluster
            cluster_mask = self.labels_ == cluster_id
            cluster_teams = self.df[cluster_mask]
            
            # Analyze Pokemon composition with stricter criteria
            pokemon_freq = cluster_teams[self.pokemon_cols].mean()
            
            # Get Pokemon that appear in >50% of teams
            frequent_pokemon = pokemon_freq[pokemon_freq > 0.5].index.tolist()
            
            if frequent_pokemon:
                # Count teams that have ALL of these Pokemon
                exact_match_mask = cluster_teams[frequent_pokemon].all(axis=1)
                exact_match_count = exact_match_mask.sum()
                
                # Only include Pokemon if they contribute to actual team compositions
                if exact_match_count > len(self.df) * 0.01:  # 1% threshold
                    analysis[cluster_id] = {
                        'size': int(exact_match_count),  # Exact number of teams with this combination
                        'core_pokemon': frequent_pokemon,
                        'common_moves': [],  # We could do similar exact matching for moves
                        'pokemon_frequencies': pokemon_freq[pokemon_freq > 0.2].to_dict(),
                        'move_frequencies': cluster_teams[self.move_cols].mean()[cluster_teams[self.move_cols].mean() > 0.3].to_dict()
                    }
        
        return analysis
    
    def dimension_reduction(self, method='tsne'):
        """
        Reduce dimensionality for visualization
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            reducer = UMAP.UMAP(random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'umap'")
            
        self.coords_2d_ = reducer.fit_transform(self.df)
        return self.coords_2d_
    
    def find_similar_teams(self, team_idx, n=5):
        """
        Find most similar teams to a given team
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        team_vector = self.df.iloc[team_idx].values.reshape(1, -1)
        similarities = cosine_similarity(team_vector, self.df)
        most_similar = np.argsort(similarities[0])[-n-1:-1][::-1]
        
        return {
            'similar_teams': most_similar.tolist(),
            'similarity_scores': similarities[0][most_similar].tolist()
        }
    
    def identify_archetypes(self):
        """
        Identify distinct archetypes based on cluster analysis
        """
        cluster_analysis = self.analyze_clusters()
        archetypes = {}
        cluster_num = 0
        for cluster_id, data in cluster_analysis.items():
            # Skip small clusters
            if (data['size'] < len(self.df) * 0.01) or (len(data['core_pokemon']) < 3):  # Less than 1% of total teams, or cluster is only 0-2 pokemon
                continue
            cluster_num += 1   
            
            # Look for known archetypes based on core Pokemon combinations
            archetype_name = f"Cluster_{cluster_num}"  # Default name
            key_moves = [move.replace('move_', '') for move in data['common_moves'][:5]]
            
            archetypes[archetype_name] = {
                'core_pokemon': data['core_pokemon'],
                'key_moves': key_moves,
                'team_count': data['size'],
                'frequency': data['size'] / len(self.df)
            }
            
        return archetypes

    def plot_cluster_heatmap(self, top_n_features=20):
        """
        Create a heatmap showing feature importance per cluster
        
        Parameters:
        top_n_features (int): Number of most variable features to include in heatmap
        
        Returns:
        DataFrame: Cluster means for the top N most variable features
        
        Note: Must run cluster_teams() first to generate cluster labels
        """
        self.labels_ = self.cluster_teams()
        if self.labels_ is None:
            raise ValueError("Must run cluster_teams() before creating heatmap")
            
        cluster_means = pd.DataFrame([
            self.df[self.labels_ == i].mean() 
            for i in sorted(set(self.labels_)) if i != -1
        ])
        
        # Select top N most variable features
        feature_variance = cluster_means.var()
        top_features = feature_variance.nlargest(top_n_features).index
        
        return sns.heatmap(cluster_means[top_features], cmap='viridis')
    
    def plot_cluster_scatter(self, reduction_method='tsne'):
        """
        Create a scatter plot of teams colored by cluster.
        Uses dimension reduction to plot in 2D space.
        
        Parameters:
        reduction_method: str
            'tsne' or 'umap' for dimension reduction
        
        Returns:
        matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        # Get 2D coordinates using dimension reduction
        coords = self.dimension_reduction(method=reduction_method)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot points, coloring by cluster
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=self.labels_,
                            cmap='tab20',  # Color map with distinct colors
                            alpha=0.6)     # Some transparency
        
        # Add legend
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        plt.legend(handles=scatter.legend_elements()[0], 
                labels=[f'Cluster {i}' for i in range(n_clusters)],
                title="Clusters",
                bbox_to_anchor=(1.05, 1), 
                loc='upper left')
        
        plt.title(f'Team Clusters ({reduction_method.upper()} projection)')
        plt.xlabel(f'{reduction_method.upper()} dimension 1')
        plt.ylabel(f'{reduction_method.upper()} dimension 2')
        
        # Make layout work with legend
        plt.tight_layout()
        
        return plt.gcf()

    def create_long_cluster_features(self):
        """
        Create a long-format table with cluster_id and core Pokemon
        """
        # Get cluster analysis first
        cluster_analysis = self.analyze_clusters()
        
        # Prepare data for long-format DataFrame
        long_data = []
        
        # Iterate through clusters
        for i, (cluster_id, data) in enumerate(cluster_analysis.items()):
            cluster_name = f"cluster_{i+1}"
            
            # Create an entry for each core Pokemon in the cluster
            for pokemon in data['core_pokemon']:
                long_data.append({
                    'cluster_id': i + 1,
                    'cluster_name': cluster_name,
                    'core_pokemon': pokemon
                })
        
        # Create DataFrame from the long-format data
        cluster_features_long = pd.DataFrame(long_data)
        
        return cluster_features_long
    
