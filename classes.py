import numpy as np
import pandas as pd
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
        Analyze characteristics of each cluster
        """
        analysis = defaultdict(dict)

        
        for cluster_id in set(self.labels_):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
                
            # Get teams in this cluster
            cluster_mask = self.labels_ == cluster_id
            cluster_teams = self.df[cluster_mask]
            
            # Analyze Pokemon composition
            pokemon_freq = cluster_teams[self.pokemon_cols].mean()
            core_pokemon = pokemon_freq[pokemon_freq > 0.5].index.tolist()
            
            # Analyze common moves
            move_freq = cluster_teams[self.move_cols].mean()
            common_moves = move_freq[move_freq > 0.3].index.tolist()
            
            analysis[cluster_id] = {
                'size': sum(cluster_mask),
                'core_pokemon': core_pokemon,
                'common_moves': common_moves,
                'pokemon_frequencies': pokemon_freq[pokemon_freq > 0.2].to_dict(),
                'move_frequencies': move_freq[move_freq > 0.2].to_dict()
            }
            
        return analysis
    
    def dimension_reduction(self, method='tsne'):
        """
        Reduce dimensionality for visualization
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            reducer = UMAP(random_state=42)
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
        
        for cluster_id, data in cluster_analysis.items():
            # Skip small clusters
            if data['size'] < len(self.df) * 0.01:  # Less than 5% of total teams
                continue
                
            # Identify key characteristics
            core_combo = set(data['core_pokemon'])
            
            # Look for known archetypes based on core Pokemon combinations
            archetype_name = f"Cluster_{cluster_id}"  # Default name
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