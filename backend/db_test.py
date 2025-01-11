from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import json
from functions import process_tournament_data  # using the new simplified function
from classes import PokemonTeamClustering

# Local connection string
DATABASE_URL = "postgresql://localhost/pokemon_vgc"
engine = create_engine(DATABASE_URL)

def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Connected to PostgreSQL!")
    except Exception as e:
        print(f"Connection failed: {e}")

def load_tournament_data(filename='data/worlds.json'):
    try:
        # Load JSON data
        with open(filename, encoding='utf-8') as f:
            tournament_data = json.load(f)
        
        # Process data with new simplified function
        df = process_tournament_data(tournament_data, "Worlds2024")
        
        # Save to PostgreSQL
        df.to_sql(
            'tournament_teams',
            engine,
            if_exists='replace',
            index=False
        )
        
        print(f"Successfully loaded {len(df)} teams into database!")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")



if __name__ == "__main__":
    test_connection()
    df = load_tournament_data()

    clusterer = PokemonTeamClustering(df)
    clusterer.cluster_teams()
    cluster_features = clusterer.create_cluster_features()
    cluster_features.to_sql(
        'cluster_features',
        engine,
        if_exists='replace',
        index=False  # Don't save the index as a column
    )
    
    # Print first few rows to verify
    try:
        with engine.connect() as conn:
            
            # Print sample tournament data
            print("\nSample data:")
            result = conn.execute(text("SELECT * FROM tournament_teams LIMIT 5"))
            for row in result:
                print(row)

            result = conn.execute(text("SELECT cluster_id, team_count FROM cluster_features WHERE koraidon = 1;"))
            for row in result:
                print(row)
    except Exception as e:
        print(f"Error querying data: {e}")