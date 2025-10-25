import json
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from backend.functions import process_tournament_data
from backend.classes import PokemonTeamClustering
from backend.database import engine

def load_tournament_files(data_dir):
    """
    Load all JSON tournament files from the specified directory.
    """
    tournaments = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    print(f"Found {len(json_files)} tournament files")
    
    for filename in json_files:
        tournament_name = os.path.splitext(filename)[0]
        file_path = os.path.join(data_dir, filename)
        
        try:
            with open(file_path, encoding='utf-8') as f:
                tournament_data = json.load(f)
                tournaments.append((tournament_data, tournament_name))
                print(f"Successfully loaded {tournament_name} with {len(tournament_data)} entries")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    return tournaments

def clean_combined_data(df):
    """
    Clean the combined dataset by handling missing values and ensuring data quality.
    """
    print("\nCleaning combined dataset...")
    initial_rows = len(df)
    print(f"Initial dataset shape: {df.shape}")
    
    # Fill NaN values with 0 for Pokemon and move columns
    feature_cols = [col for col in df.columns 
                   if not col.startswith(('Competition', 'Name', 'Nationality', 'Wins', 'Losses'))]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Ensure numeric columns are properly typed
    df[feature_cols] = df[feature_cols].astype(float)
    
    # Drop any remaining rows with NaN values
    df = df.dropna()
    if initial_rows > len(df):
        print(f"Dropped {initial_rows - len(df)} rows with NaN values")
    
    print(f"Final dataset shape: {df.shape}")
    return df

def remove_zero_vector_teams(df, clusterer):
    """
    Remove teams that become zero vectors after feature selection.
    """
    print("\nChecking for zero-vector teams...")
    initial_rows = len(clusterer.df)
    
    # Calculate row sums
    row_sums = clusterer.df.sum(axis=1)
    
    # Find non-zero rows
    non_zero_mask = row_sums > 0
    
    # Filter both dataframes
    clusterer.df = clusterer.df[non_zero_mask]
    df = df[non_zero_mask]
    
    removed_count = initial_rows - len(clusterer.df)
    if removed_count > 0:
        print(f"Removed {removed_count} teams that had no features after selection")
    print(f"Remaining teams: {len(clusterer.df)}")
    
    return df, clusterer

def process_multiple_tournaments_pipeline(data_dir):
    """
    Unified pipeline to process multiple tournaments.
    """
    try:
        print("Starting multi-tournament processing pipeline")
        
        # 1. Load all tournament files
        tournaments = load_tournament_files(data_dir)
        if not tournaments:
            raise ValueError("No tournaments loaded successfully")
        
        # 2. Process and combine tournament data
        combined_df = pd.DataFrame()
        for tournament_data, tournament_name in tournaments:
            print(f"\nProcessing tournament: {tournament_name}")
            df = process_tournament_data(tournament_data, tournament_name)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        print(f"\nCombined dataset contains {len(combined_df)} teams")
        
        # 3. Clean the combined data
        combined_df = clean_combined_data(combined_df)
        
        # 4. Initialize clustering and apply feature selection
        print("Initializing clustering...")
        clusterer = PokemonTeamClustering(combined_df)
        
        print("Applying feature selection...")
        clusterer.select_features(method='frequency', threshold=0.02)
        clusterer.select_features(method='variance', threshold=0.01)
        
        # 5. Remove zero-vector teams after feature selection
        combined_df, clusterer = remove_zero_vector_teams(combined_df, clusterer)
        
        # 6. Apply feature normalization
        print("Applying feature normalization...")
        clusterer.normalize_features(pokemon_weight=1.0, move_weight=0.5)
        
        # 7. Save processed data to database
        print("Saving processed tournament data to database...")
        combined_df.to_sql(
            'tournament_teams', 
            engine, 
            if_exists='replace', 
            index=False
        )
        
        # 8. Perform clustering
        print("Performing clustering...")
        clusterer.cluster_teams()
        
        # 9. Save cluster features to database
        print("Saving cluster features to database...")
        cluster_features = clusterer.create_long_cluster_features()
        cluster_features.to_sql(
            'cluster_features', 
            engine, 
            if_exists='replace', 
            index=False
        )
        
        print("\nPipeline completed successfully!")
        print(f"Processed {len(tournaments)} tournaments")
        print(f"Total teams analyzed: {len(combined_df)}")
        
        return clusterer, combined_df
        
    except Exception as e:
        print(f"Error in tournament processing pipeline: {e}")
        raise

def main():
    process_multiple_tournaments_pipeline('data')

if __name__ == "__main__":
    main()