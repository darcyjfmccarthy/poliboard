WITH core_pokemon AS (
        SELECT column_name
        FROM information_schema.columns 
        WHERE table_name = 'cluster_features' 
        AND column_name NOT IN ('cluster_id', (SELECT column_name FROM information_schema.columns WHERE table_name = 'cluster_features' LIMIT 1))
        AND EXISTS (
            SELECT 1 
            FROM cluster_features 
            WHERE cluster_id = 1
            AND (cluster_features.* -> column_name)::text = '1'
        )
    ),
    matching_teams AS (
        SELECT 
            ARRAY[
                COALESCE(pokemon1, ''),
                COALESCE(pokemon2, ''),
                COALESCE(pokemon3, ''),
                COALESCE(pokemon4, ''),
                COALESCE(pokemon5, ''),
                COALESCE(pokemon6, '')
            ] AS team_pokemon,
            (wins::float / NULLIF(wins + losses, 0)) as win_rate
        FROM tournament_teams t
        WHERE 
            pokemon1 IN (SELECT column_name FROM core_pokemon) OR
            pokemon2 IN (SELECT column_name FROM core_pokemon) OR
            pokemon3 IN (SELECT column_name FROM core_pokemon) OR
            pokemon4 IN (SELECT column_name FROM core_pokemon) OR
            pokemon5 IN (SELECT column_name FROM core_pokemon) OR
            pokemon6 IN (SELECT column_name FROM core_pokemon)
        ORDER BY 
            win_rate DESC
        LIMIT 5
    )
    SELECT 
        team_pokemon
    FROM matching_teams
    WHERE 
        team_pokemon[1] != '' OR 
        team_pokemon[2] != '' OR 
        team_pokemon[3] != ''