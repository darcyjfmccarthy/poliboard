from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import numpy as np
import json
from classes import PokemonTeamClustering
from functions import process_multiple_entries
from fastapi.responses import StreamingResponse
import httpx
import asyncio

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
        clusterer.select_features(method='frequency', threshold=0.03)

        clusterer.select_features(method='variance', threshold=0.01)

        # Weight Pokemon more heavily than moves
        clusterer.normalize_features(pokemon_weight=1.0, move_weight=0.6)

        optimal_clusters = clusterer.find_optimal_clusters(max_clusters=20)
        
        # Perform clustering
        labels = clusterer.cluster_teams(n_clusters=optimal_clusters)
        
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

@app.get("/archetypes")
async def get_archetypes():
    if clusterer_instance is None:
        raise HTTPException(status_code=400, detail="No clustering results available. Run /analyze first.")
    
    return {
        "archetypes": numpy_to_python(clusterer_instance.identify_archetypes())
    }

@app.get("/pokemon/sprite/{pokemon_name}")
async def get_pokemon_sprite(pokemon_name: str):
    async with httpx.AsyncClient() as client:
        try:
            # Try to fetch from Pokemon Showdown
            response = await client.get(
                f"https://play.pokemonshowdown.com/sprites/gen5/{pokemon_name.lower()}.png",
                timeout=10.0
            )
            
            if response.status_code == 200:
                return StreamingResponse(
                    response.iter_bytes(),
                    media_type="image/png",
                    headers={
                        "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
                    }
                )
            else:
                raise HTTPException(status_code=404, detail=f"Sprite not found for {pokemon_name}")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching sprite: {str(e)}")