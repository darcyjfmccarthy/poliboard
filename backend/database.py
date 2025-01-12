from sqlalchemy import create_engine, Table, MetaData, text, inspect
import pandas as pd
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Get environment setting
ENV = os.getenv('APP_ENV', 'development')  # Default to development if not set

print("Database configuration:")
print(f"ENV: {ENV}")
print(f"DATABASE_URL from env: {os.getenv('DATABASE_URL')}")

if ENV == 'production':
    # Supabase configuration
    DB_URL = os.getenv('DATABASE_URL')
    if not DB_URL:
        raise ValueError("No DATABASE_URL found in environment variables")
    
    print(f"Using production database URL: {DB_URL}")
    
    engine = create_engine(
        DB_URL,
        connect_args={"sslmode": "require"}
    )
else:
    print("Using local database")
    engine = create_engine("postgresql://localhost/vgc_clustering")

def get_db_info():
    """Helper function to check which database we're connected to"""
    return {
        "environment": ENV,
        "database": "Supabase" if ENV == "production" else "Local PostgreSQL"
    }


metadata = MetaData()

# Define the tables
def get_tables():
    """Initialize tables only if they exist"""
    inspector = inspect(engine)
    if 'tournament_teams' in inspector.get_table_names(schema='public'):
        global teams_table
        teams_table = Table('tournament_teams', metadata, autoload_with=engine, schema='public')
    if 'cluster_features' in inspector.get_table_names(schema='public'):
        global cluster_features
        cluster_features = Table('cluster_features', metadata, autoload_with=engine, schema='public')

def find_teams_from_cluster(engine, cluster_id: int, limit: int = 20) -> tuple[pd.DataFrame, dict]:
    try:
        # First get the cluster features
        with engine.connect() as conn:
            cluster_data = conn.execute(text(
                "SELECT * from cluster_features WHERE cluster_id = :cluster_id;"
            ), {"cluster_id": cluster_id})
            df = pd.DataFrame(cluster_data.mappings().all())

        if df.empty:
            raise ValueError(f"No cluster found with ID {cluster_id}")

        # Build the query with proper parameterization
        query = """
            WITH cluster_teams AS (
                SELECT *
                FROM tournament_teams
                WHERE 1=1 {features}
            ),
            stats AS (
                SELECT COUNT(*) as total_appearances,
                       CAST(SUM("Wins") AS FLOAT) / NULLIF(SUM("Wins" + "Losses"), 0) * 100 as winrate
                FROM cluster_teams
            )
            SELECT t.*, s.total_appearances, s.winrate
            FROM cluster_teams t
            CROSS JOIN stats s
            ORDER BY t."Wins" DESC
            LIMIT :limit
        """

        # Build features string safely
        features = ""
        for row in df['core_pokemon'].to_list():
            features += f'\n\tAND "{row}" = 1'

        # Execute the final query
        with engine.connect() as conn:
            result = conn.execute(
                text(query.format(features=features)), 
                {"limit": limit}
            )
            # Get column names from result
            columns = result.keys()
            # Fetch all rows
            rows = result.fetchall()
            
            if not rows:
                return pd.DataFrame(), {'appearances': 0, 'winrate': 0.0}
            
            # Create DataFrame with explicit column names
            result_df = pd.DataFrame(rows, columns=columns)

            # Extract stats
            stats = {
                'appearances': int(result_df['total_appearances'].iloc[0]),
                'winrate': round(float(result_df['winrate'].iloc[0]), 1)
            }

            # Remove stats columns from main DataFrame
            result_df = result_df.drop(['total_appearances', 'winrate'], axis=1)

            return result_df, stats

    except Exception as e:
        print(f"Error in find_teams_from_cluster: {str(e)}")
        raise

def get_top_teams_in_cluster(cluster_id: int, limit: int = 20) -> list:
    """
    Get top teams from a cluster in a format suitable for the frontend.
    
    Args:
        cluster_id: ID of the cluster to search for
        limit: Maximum number of teams to return
    
    Returns:
        List of teams, where each team is a list of Pokemon names
    """
    try:
        # Correctly unpack both values
        df, stats = find_teams_from_cluster(engine, cluster_id, limit)
        
        if df.empty:
            return []

        # Convert DataFrame to list of Pokemon teams
        teams = []
        for _, row in df.iterrows():
            # Get all Pokemon columns where value is 1
            team_pokemon = [col for col in df.columns 
                          if not col.startswith(('Competition', 'Name', 'Nationality', 'Wins', 'Losses', 'move_'))
                          and row[col] == 1]
            teams.append(team_pokemon)

        return teams

    except Exception as e:
        print(f"Error in get_top_teams_in_cluster: {str(e)}")
        raise

def get_all_clusters() -> dict:
    """
    Get all clusters and their core Pokemon.
    
    Returns:
        Dictionary mapping cluster IDs to lists of core Pokemon
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT cluster_id, core_pokemon FROM cluster_features"))
            clusters = {}
            for row in result:
                cluster_id = row[0]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(row[1])
            
            return {f"Cluster_{k}": {"core_pokemon": v} for k, v in clusters.items()}

    except Exception as e:
        print(f"Error in get_all_clusters: {str(e)}")
        raise