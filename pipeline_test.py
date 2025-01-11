from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import json
from backend.functions import process_tournament_data, test_connection, load_tournament_data
from backend.classes import PokemonTeamClustering

# Local connection string
DATABASE_URL = "postgresql://localhost/vgc_clustering"
engine = create_engine(DATABASE_URL)

if __name__ == "__main__":
    test_connection(engine)
    df = load_tournament_data(engine)

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

            result = conn.execute(text('SELECT cluster_id, team_count FROM cluster_features WHERE "Incineroar" = 1;'))
            for row in result:
                print(row)
    except Exception as e:
        print(f"Error querying data: {e}")