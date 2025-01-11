from sqlalchemy import create_engine, Table, MetaData, select, and_

# Example database URL (replace with your actual connection string)
engine = create_engine("postgresql://localhost/vgc_clustering")

metadata = MetaData()

# Define the tables
teams_table = Table('tournament_teams', metadata, autoload_with=engine, schema='public')
cluster_features = Table('cluster_features', metadata, autoload_with=engine, schema='public')

# Function to query teams based on cluster ID
def get_teams_for_cluster(cluster_id: int):
    # Step 1: Fetch the Pokémon requirements for the given cluster
    cluster_query = select([cluster_features]).where(cluster_features.c.cluster_id == cluster_id)
    with engine.connect() as connection:
        cluster_result = connection.execute(cluster_query).fetchone()
    
    if not cluster_result:
        print(f"No cluster found with ID {cluster_id}")
        return []
    
    # Step 2: Build the condition for querying teams that meet the cluster's Pokémon requirements
    conditions = []
    
    # Dynamically generate conditions based on the Pokémon columns from the cluster
    for column_name in cluster_result.keys():
        if column_name != 'cluster_id' and column_name != 'team_count':  # Skip non-Pokemon columns
            if cluster_result[column_name] == 1:
                # Add the condition that the team has this Pokémon with value 1
                conditions.append(teams_table.c[column_name] == 1)
    
    # Step 3: Query teams that match all conditions
    team_query = select([teams_table]).where(and_(*conditions))
    
    # Step 4: Execute the query
    with engine.connect() as connection:
        team_results = connection.execute(team_query).fetchall()
    
    return team_results

# Example usage
cluster_id = 1
teams = get_teams_for_cluster(cluster_id)

for team in teams:
    print(team)
