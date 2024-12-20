import re
import pandas as pd
import json
from collections import defaultdict

def process_tournament_data(data_dict, mon_set, competition_name):
    """
    Process Pokemon tournament player data into a DataFrame row.
    
    Parameters:
    data_dict (dict): Player data dictionary
    competition_name (str): Name of the competition
    
    Returns:
    pandas.DataFrame: Single row DataFrame with processed player data
    """
    # Extract nationality from name
    name_parts = re.match(r'(.*?)\s*\[(\w+)\]', data_dict['name'])
    if name_parts:
        name = name_parts.group(1).strip()
        nationality = name_parts.group(2)
    else:
        name = data_dict['name']
        nationality = 'Unknown'
    
    # Create base dictionary with main info
    processed_data = {
        'Competition': competition_name,
        'Name': name,
        'Nationality': nationality,
        'Wins': data_dict['record']['wins'],
        'Losses': data_dict['record']['losses']
    }
    
    # Add Pokemon information
    for i in mon_set:
        if f"'{i}'" in str(data_dict['decklist']):
            processed_data[i] = 1
        else:
            processed_data[i] = 0
    
    # Convert to DataFrame
    df = pd.DataFrame([processed_data])
    
    return df

# To process multiple entries:
def process_multiple_entries(entries_list, mon_set, competition_name):
    """
    Process multiple tournament entries into a single DataFrame.
    
    Parameters:
    entries_list (list): List of player data dictionaries
    competition_name (str): Name of the competition
    
    Returns:
    pandas.DataFrame: DataFrame containing all processed entries
    """
    return pd.concat([
        process_tournament_data(entry, mon_set, competition_name) 
        for entry in entries_list
    ], ignore_index=True)

def preprocess_all_data(input_file_name, tournament_name):
    with open(input_file_name, encoding='utf-8') as f:
        worlds_data = json.load(f)
        
    pokemon_list = []
    
    for player in worlds_data:
        # Loop through each Pokemon in the player's decklist
        for pokemon in player['decklist']:
            pokemon_list.append(pokemon['name'])

    pokemon_set = list(set(pokemon_list))

    df = process_multiple_entries(worlds_data, pokemon_set, tournament_name)


def calculate_cluster_winrates(tournament_data, cluster_labels):
    """
    Calculate win rates for each cluster using tournament results
    
    Parameters:
    tournament_data: List of dictionaries containing team data with win/loss records
    cluster_labels: Array of cluster assignments matching the order of tournament_data
    
    Returns:
    Dictionary of cluster stats including win rates
    """
    cluster_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'teams': 0})
    
    # Combine tournament results with cluster assignments
    for team_data, cluster in zip(tournament_data, cluster_labels):
        if cluster == -1:  # Skip noise cluster if using DBSCAN
            continue
            
        cluster_stats[f"Cluster_{cluster+1}"]["teams"] += 1
        cluster_stats[f"Cluster_{cluster+1}"]["wins"] += team_data['record']['wins']
        cluster_stats[f"Cluster_{cluster+1}"]["losses"] += team_data['record']['losses']
    
    # Calculate win rates and add to stats
    for cluster_id, stats in cluster_stats.items():
        total_games = stats['wins'] + stats['losses']
        if total_games > 0:
            stats['win_rate'] = stats['wins'] / total_games
        else:
            stats['win_rate'] = 0
    
    return dict(cluster_stats)

# Usage example:
def analyze_cluster_performance(tournament_data, clusterer):
    """
    Combine clustering results with performance analysis
    """
    # Get cluster assignments if not already done
    if not hasattr(clusterer, 'labels_'):
        clusterer.cluster_teams()
    
    # Calculate win rates
    cluster_stats = calculate_cluster_winrates(tournament_data, clusterer.labels_)
    
    # Combine with archetype information
    archetypes = clusterer.identify_archetypes()
    
    # Merge stats into archetype data
    for cluster_id, archetype in archetypes.items():
        if cluster_id in cluster_stats:
            archetype['win_rate'] = cluster_stats[cluster_id]['win_rate']
            archetype['total_games'] = (
                cluster_stats[cluster_id]['wins'] + 
                cluster_stats[cluster_id]['losses']
            )
            archetype['total_wins'] = cluster_stats[cluster_id]['wins']
            archetype['total_losses'] = cluster_stats[cluster_id]['losses']
    
    return archetypes