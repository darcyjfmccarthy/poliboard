import re
import pandas as pd
import json
from collections import defaultdict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def process_tournament_data(tournament_data, competition_name):
    """
    Process tournament data into a DataFrame with one-hot encoded Pokemon.
    """
    # Get unique Pokemon first
    pokemon_set = {
        pokemon['name']
        for entry in tournament_data 
        for pokemon in entry['decklist']
    }
    
    processed_entries = []
    for entry in tournament_data:
        # Extract player info
        name_parts = re.match(r'(.*?)\s*\[(\w+)\]', entry['name'])
        if name_parts:
            name, nationality = name_parts.group(1).strip(), name_parts.group(2)
        else:
            name, nationality = entry['name'], 'Unknown'
            
        # Create base entry with metadata first
        processed_entry = {
            'Competition': competition_name,
            'Name': name,
            'Nationality': nationality,
            'Wins': entry['record']['wins'],
            'Losses': entry['record']['losses']
        }
        
        # Add Pokemon presence (1/0) as additional columns
        team_pokemon = {pokemon['name'] for pokemon in entry['decklist']}
        pokemon_columns = {pokemon: (1 if pokemon in team_pokemon else 0) for pokemon in pokemon_set}
        processed_entry.update(pokemon_columns)
            
        processed_entries.append(processed_entry)
    
    df = pd.DataFrame(processed_entries)
    
    # Ensure correct column order
    metadata_cols = ['Competition', 'Name', 'Nationality', 'Wins', 'Losses']
    pokemon_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + pokemon_cols]  # Reorder columns
    
    return df

def calculate_cluster_winrates(tournament_data, labels, clusterer):
    """Calculate win rates based on exact Pokemon matches"""
    cluster_analysis = clusterer.analyze_clusters()
    cluster_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'teams': 0})
    
    for team_data, cluster_label in zip(tournament_data, labels):
        if cluster_label == -1 or cluster_label not in cluster_analysis:
            continue
            
        cluster_name = f"Cluster_{cluster_label+1}"
        core_pokemon = cluster_analysis[cluster_label]['core_pokemon']
        
        # Check if team has all core Pokemon
        has_all_core = all(
            any(pokemon['name'] == core_mon for pokemon in team_data['decklist'])
            for core_mon in core_pokemon
        )
        
        if has_all_core:
            cluster_stats[cluster_name]["teams"] += 1
            cluster_stats[cluster_name]["wins"] += team_data['record']['wins']
            cluster_stats[cluster_name]["losses"] += team_data['record']['losses']
    
    return dict(cluster_stats)

def test_connection(engine):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Connected to PostgreSQL!")
    except Exception as e:
        print(f"Connection failed: {e}")

def load_tournament_data(engine, filename='data/worlds.json'):
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
