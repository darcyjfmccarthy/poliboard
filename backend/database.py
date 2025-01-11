# database.py
from sqlalchemy import create_engine, text

# Create engine once and reuse
DATABASE_URL = "postgresql://localhost/pokemon_vgc"
engine = create_engine(DATABASE_URL)

def get_top_teams_in_cluster(cluster_id, limit=20):
    """
    Get the top performing teams that match a cluster's core Pokemon
    """
    query = """
    WITH cluster_core_pokemon AS (
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'cluster_features'
        AND column_name NOT IN ('cluster_id', 'team_count')
        AND (SELECT {column_name} FROM cluster_features WHERE cluster_id = :cluster_id) = 1
    )
    SELECT 
        t.name as player_name,
        t.nationality,
        t.wins,
        t.losses,
        CAST(t.wins AS FLOAT) / NULLIF(t.wins + t.losses, 0) as win_rate
    FROM tournament_teams t
    WHERE NOT EXISTS (
        SELECT column_name 
        FROM cluster_core_pokemon
        WHERE t.{column_name} != 1
    )
    ORDER BY 
        t.wins DESC,
        win_rate DESC
    LIMIT :limit
    """
    
    with engine.connect() as conn:
        result = conn.execute(
            text(query),
            {"cluster_id": cluster_id, "limit": limit}
        )
        return [dict(row) for row in result]