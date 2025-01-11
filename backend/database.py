# database.py
from sqlalchemy import create_engine, text

# Create engine once and reuse
DATABASE_URL = "postgresql://localhost/pokemon_vgc"
engine = create_engine(DATABASE_URL)

from sqlalchemy import inspect, text
import re

def clean_pokemon_name(pokemon_name):
    """
    Clean Pokemon names to match the column name generation in create_cluster_features
    """
    return re.sub(r'[\'\[\]]', '', pokemon_name.lower().replace(' ', '_'))

def get_top_teams_in_cluster(cluster_id, limit=20):
    
    """
    Dynamically get top teams for a cluster based on core Pokemon
    """
    # First, find the core Pokemon for this cluster
    with engine.connect() as conn:
        # Query to find core Pokemon for the specific cluster
        core_pokemon_query = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'cluster_features' 
        AND column_name NOT IN ('cluster_id', (
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'cluster_features' 
            LIMIT 1
        ))
        AND EXISTS (
            SELECT 1 
            FROM cluster_features 
            WHERE cluster_id = :cluster_id 
            AND cluster_features."{column_name}" = 1
        )
        """)
        
        core_pokemon_result = conn.execute(
            core_pokemon_query,
            {"cluster_id": cluster_id}
        )
        core_pokemon_list = [row.column_name for row in core_pokemon_result]
        
        print(f"Core Pokemon for cluster {cluster_id}: {core_pokemon_list}")
        
        # If no core Pokemon found, return empty list
        if not core_pokemon_list:
            return []
        
        # Construct the main query dynamically
        teams_query = text(f"""
        WITH matching_teams AS (
            SELECT 
                ARRAY(
                    SELECT column_name 
                    FROM (VALUES {', '.join([f'(t."{col}")' for col in core_pokemon_list])}) 
                    AS team_columns(column_name)
                    WHERE team_columns.column_name = 1
                ) AS team_pokemon,
                ("Wins"::float / NULLIF("Wins" + "Losses", 0)) as win_rate
            FROM tournament_teams t
            WHERE
                {' OR '.join([f't."{col}" = 1' for col in core_pokemon_list])}
            ORDER BY
                win_rate DESC
            LIMIT :limit
        )
        SELECT 
            team_pokemon
        FROM matching_teams
        WHERE 
            array_length(team_pokemon, 1) > 0
        """)
        
        # Execute the query
        result = conn.execute(
            teams_query,
            {"cluster_id": cluster_id, "limit": limit}
        )
        
        teams = []
        for row in result:
            teams.append(row.team_pokemon)
        
        return teams

if __name__ == "__main__":
    get_top_teams_in_cluster(1)