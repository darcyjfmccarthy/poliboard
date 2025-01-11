from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import numpy as np
import json
from backend.classes import PokemonTeamClustering
from backend.functions import process_tournament_data, calculate_cluster_winrates
from fastapi.responses import StreamingResponse
import httpx
import asyncio
from collections import defaultdict
from backend.database import get_top_teams_in_cluster

def load_tournament_data(filename='data/worlds.json'):
    try:
        # Load JSON data
        with open(filename, encoding='utf-8') as f:
            tournament_data = json.load(f)
        
        # Process data with new simplified function
        df = process_tournament_data(tournament_data, "Worlds2024")
    
        
        print(f"Successfully loaded {len(df)} teams into database!")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")



    

def main():
    try:
        with open('data/worlds.json', 'r', encoding='utf-8') as f:
            tournament_data = json.load(f)
            print(f"Successfully loaded tournament data with {len(tournament_data)} entries")
        df = load_tournament_data()

        clusterer = PokemonTeamClustering(df)
        clusterer.cluster_teams()
        cluster_features = clusterer.create_cluster_features()
# Get archetypes
        archetypes = clusterer.identify_archetypes()
        print("Raw archetypes:", archetypes)  # Debug print
        
        # Get cluster stats
        cluster_stats = calculate_cluster_winrates(tournament_data, clusterer.labels_, clusterer)
        print("Cluster stats:", cluster_stats)  # Debug print
        
        # Make sure archetypes is not None
        if archetypes is None:
            archetypes = {}
            
        # Add win rates to archetypes
        for cluster_id, archetype in archetypes.items():
            if cluster_id in cluster_stats:
                stats = cluster_stats[cluster_id]
                total_games = stats['wins'] + stats['losses']
                archetype['win_rate'] = stats['wins'] / total_games if total_games > 0 else 0
                archetype['total_wins'] = stats['wins']
                archetype['total_losses'] = stats['losses']
        
        response_data = {"archetypes": (archetypes)}
        print("Final response:", response_data)  # Debug print
        return response_data
        
    except Exception as e:
        print(f"Error in get_archetypes: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    main()