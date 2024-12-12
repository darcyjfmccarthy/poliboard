import re
import pandas as pd

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