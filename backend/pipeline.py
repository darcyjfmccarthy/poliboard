import argparse
import pandas as pd
from classes import PokemonTeamClustering
from functions import *
import json



def main():
    parser = argparse.ArgumentParser(description='Pokemon Team Clustering Pipeline')
    parser.add_argument('input_file', type=str, help='Path to input JSON file')
    parser.add_argument('tournament_name', type=str, help='Tournament name')
    
    args = parser.parse_args()
    
    try:
        # Load data
        with open(args.input_file, encoding='utf-8') as f:
            worlds_data = json.load(f)
        
        pokemon_list = []
        
        for player in worlds_data:
            # Loop through each Pokemon in the player's decklist
            for pokemon in player['decklist']:
                pokemon_list.append(pokemon['name'])

        pokemon_set = list(set(pokemon_list))

        df = process_multiple_entries(worlds_data, pokemon_set, args.tournament_name)
        print(f"Loaded {len(df)} teams from {args.input_file}. Pokemon count: {len(pokemon_set)}")
        
        # Initialize clusterer
        clusterer = PokemonTeamClustering(df)


        clusterer.select_features(method='frequency', threshold=0.03)

        clusterer.select_features(method='variance', threshold=0.01)

        # Weight Pokemon more heavily than moves
        clusterer.normalize_features(pokemon_weight=1.0, move_weight=0.6)

        optimal_clusters = clusterer.find_optimal_clusters(max_clusters=20)

        best_n = optimal_clusters.loc[optimal_clusters['silhouette_score'].idxmax(), 'n_clusters']

        clusterer.cluster_teams(n_clusters=best_n)


        print(clusterer.identify_archetypes())
        
    except FileNotFoundError:
        print(f"Error: Could not find file {args.input_file}")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

if __name__ == "__main__":
    main()