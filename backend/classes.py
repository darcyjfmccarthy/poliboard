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
        self.raw_df = df.copy()  # full version (metadata + features)
        
        # Identify columns
        self.raw_pokemon_cols = [col for col in df.columns[6:-2]]  # all original Pokémon columns
        self.raw_move_cols = [col for col in df.columns[-2:]]      # move columns
        
        # Feature subset (to be filtered later)
        self.df = df[[col for col in df.columns[6:]]].copy()
        self.pokemon_cols = self.raw_pokemon_cols.copy()
        self.move_cols = self.raw_move_cols.copy()
    
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
        Identify distinct archetypes based on cluster analysis,
        create descriptive names with key Pokémon, and save mapping for plotting.
        """
        cluster_analysis = self.analyze_clusters()
        archetypes = {}
        cluster_num = 0
        team_to_archetype = {}

        for cluster_id, data in cluster_analysis.items():
            # Skip small/weak clusters
            if (data['size'] < len(self.df) * 0.01) or (len(data['core_pokemon']) < 3):
                continue
            cluster_num += 1

            # Create descriptive name: Archetype N: Pokémon, Pokémon, Pokémon
            core_pokemon_list = data['core_pokemon']
            pokemon_str = ", ".join(core_pokemon_list)
            archetype_name = f"Archetype {cluster_num}: {pokemon_str}"

            key_moves = [move.replace('move_', '') for move in data['common_moves'][:5]]

            archetypes[archetype_name] = {
                'core_pokemon': data['core_pokemon'],
                'key_moves': key_moves,
                'team_count': data['size'],
                'frequency': data['size'] / len(self.df),
            }

            # Assign each team in this cluster to this archetype
            mask = self.labels_ == cluster_id
            for idx in np.where(mask)[0]:
                team_to_archetype[idx] = archetype_name

        # Save mapping as Series
        self.final_archetypes_ = pd.Series(team_to_archetype)
        self.final_archetypes_.index.name = "team_index"

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
        Create an interactive scatter plot showing final archetypes with hover info including
        player name, tournament, record, and Pokémon team. Point size scales softly with wins.
        """
        import plotly.express as px
        import numpy as np

        # Reduce dimensions for plotting
        coords = self.dimension_reduction(method=reduction_method)
        df_plot = pd.DataFrame(coords, columns=['x', 'y'])

        if not hasattr(self, 'final_archetypes_'):
            raise ValueError("Must run identify_archetypes() before plotting final archetypes.")

        # Filter only teams with identified archetypes
        df_plot['archetype'] = self.final_archetypes_
        df_plot = df_plot.dropna(subset=['archetype'])

        # Pull metadata from raw_df (not filtered)
        df_plot['Name'] = self.raw_df.loc[df_plot.index, 'Name']
        df_plot['Tournament'] = self.raw_df.loc[df_plot.index, 'Competition']  # or 'Tournament'
        df_plot['Wins'] = self.raw_df.loc[df_plot.index, 'Wins']
        df_plot['Losses'] = self.raw_df.loc[df_plot.index, 'Losses']

        # Build hover text for Pokémon list using original (unfiltered) Pokémon columns
        hover_texts = []
        for i in df_plot.index:
            team_row = self.raw_df.loc[i, self.raw_pokemon_cols]
            team_pokemon = team_row[team_row == 1].index.tolist()
            formatted = "<br>".join(
                [", ".join(team_pokemon[j:j+3]) for j in range(0, len(team_pokemon), 3)]
            )
            hover_texts.append(formatted if formatted else "(No Pokémon detected)")

        df_plot['team_pokemon'] = hover_texts

        # Interactive scatter plot
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color='archetype',
            hover_name='Name',
            size=np.log1p(df_plot['Wins'])/5,  # Soft scaling for wins
            hover_data={
                'Tournament': True,
                'Wins': True,
                'Losses': True,
                'team_pokemon': True,
                'x': False,
                'y': False,
            },
            title=f"Pokémon Team Archetypes ({reduction_method.upper()} projection)",
            category_orders={'archetype': sorted(df_plot['archetype'].unique())},
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        # Hover box styling
        fig.update_traces(
            marker=dict(line=dict(width=0.5, color='DarkSlateGrey')),
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>"
                "🏆 <b>Tournament:</b> %{customdata[0]}<br>"
                "⚔️ <b>Record:</b> %{customdata[1]}W - %{customdata[2]}L<br><br>"
                "🐉 <b>Pokémon:</b><br>%{customdata[3]}<extra></extra>"
            )
        )

        # Layout: wide figure + legend outside
        fig.update_layout(
            legend_title="Archetypes",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1
            ),
            width=1300,
            height=750,
            margin=dict(l=60, r=260, t=80, b=60),
            template="plotly_dark",
            hovermode="closest"
        )

        fig.show()

    def create_long_cluster_features(self):
        """
        Create a long-format table with cluster_id and core Pokemon
        """
        # Get cluster analysis first
        cluster_analysis = self.analyze_clusters()

        # Prune clusters that are too small
        new_clusters = {}
        cluster_id = 0
        for cluster_id, data in cluster_analysis.items():
            # Skip small clusters
            if (data['size'] < len(self.df) * 0.01) or (len(data['core_pokemon']) < 3):  # Less than 1% of total teams, or cluster is only 0-2 pokemon
                continue
            cluster_id += 1
            new_clusters[cluster_id] = data
        
        # Prepare data for long-format DataFrame
        long_data = []
        
        # Iterate through clusters
        for i, (cluster_id, data) in enumerate(new_clusters.items()):
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
    
