# Proposed unified pipeline script (pipeline.py)
import json
from sqlalchemy import create_engine
from backend.functions import process_tournament_data
from backend.classes import PokemonTeamClustering
from backend.database import engine  # Use your existing engine

def process_tournament_pipeline(json_path, tournament_name):
    """
    Unified pipeline to:
    1. Load JSON
    2. Process tournament data
    3. Insert into database
    4. Run clustering
    5. Insert cluster features into database
    """
    try:
        # 1. Load tournament data
        with open(json_path, encoding='utf-8') as f:
            tournament_data = json.load(f)
        
        # 2. Process data into DataFrame
        df = process_tournament_data(tournament_data, tournament_name)
        
        # 3. Insert raw tournament data to database
        df.to_sql(
            'tournament_teams', 
            engine, 
            if_exists='replace', 
            index=False
        )
        
        # 4. Run clustering
        clusterer = PokemonTeamClustering(df)
        clusterer.select_features(method='frequency', threshold=0.02)
        clusterer.select_features(method='variance', threshold=0.01)
        clusterer.normalize_features(pokemon_weight=1.0, move_weight=0.5)
        clusterer.cluster_teams()
        
        # 5. Insert cluster features to database
        cluster_features = clusterer.create_long_cluster_features()
        cluster_features.to_sql(
            'cluster_features', 
            engine, 
            if_exists='replace', 
            index=False
        )
        
        return clusterer, df
    
    except Exception as e:
        print(f"Error in tournament processing pipeline: {e}")
        raise

def main():
    process_tournament_pipeline(
        'data/worlds.json', 
        'Worlds 2024'
    )

if __name__ == "__main__":
    main()