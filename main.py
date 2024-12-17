from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import numpy as np
import json
from classes import PokemonTeamClustering
from functions import process_multiple_entries

app = FastAPI(title="Pokemon Team Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clusterer_instance = None

def numpy_to_python(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    return obj

@app.post("/analyze")
async def analyze_teams(tournament_data: List[Dict[str, Any]], tournament_name: str = "Unknown"):
    try:
        # Extract Pokemon list
        pokemon_list = []
        for player in tournament_data:
            for pokemon in player['decklist']:
                pokemon_list.append(pokemon['name'])
        
        pokemon_set = list(set(pokemon_list))
        
        # Process data
        df = process_multiple_entries(tournament_data, pokemon_set, tournament_name)
        
        # Initialize clusterer
        global clusterer_instance
        clusterer = PokemonTeamClustering(df)
        
        # Apply preprocessing
        clusterer.select_features(method='frequency', threshold=0.01)
        clusterer.normalize_features(pokemon_weight=1.0, move_weight=0.5)
        
        # Perform clustering
        labels = clusterer.cluster_teams()
        
        # Store instance
        clusterer_instance = clusterer
        
        # Convert numpy types to Python native types
        response_data = {
            "status": "success",
            "archetypes": numpy_to_python(clusterer.identify_archetypes())
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize")
async def get_visualization():
    if clusterer_instance is None:
        raise HTTPException(status_code=400, detail="No clustering results available. Run /analyze first.")
    
    coords = clusterer_instance.dimension_reduction()
    labels = clusterer_instance.labels_
    
    return {
        "coordinates": numpy_to_python(coords),
        "labels": numpy_to_python(labels)
    }