from sqlalchemy import create_engine, Table, MetaData, text
import pandas as pd
from typing import Optional

# Database connection
engine = create_engine("postgresql://localhost/vgc_clustering")
metadata = MetaData()

# Define the tables
teams_table = Table('tournament_teams', metadata, autoload_with=engine, schema='public')
cluster_features = Table('cluster_features', metadata, autoload_with=engine, schema='public')

def find_teams_from_cluster(engine, cluster_id: int, limit: int = 20) -> pd.DataFrame:
    """
    Find teams from a specific cluster using direct SQL queries.
    
    Args:
        engine: SQLAlchemy engine instance
        cluster_id: ID of the cluster to search for
        limit: Maximum number of teams to return (default: 20)
    
    Returns:
        DataFrame containing the matching teams
    """
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
            SELECT *
            FROM tournament_teams
            WHERE 1=1 {features}
            ORDER BY "Wins" DESC
            LIMIT :limit
        """

        # Build features string safely
        features = ""
        for row in df['core_pokemon'].to_list():
            features += f'\n\tAND "{row}" = 1'

        # Execute the final query
        with engine.connect() as conn:
            data = conn.execute(
                text(query.format(features=features)), 
                {"limit": limit}
            )
            result_df = pd.DataFrame(data.mappings().all())

        return result_df

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
        df = find_teams_from_cluster(engine, cluster_id, limit)
        
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