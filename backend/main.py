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

app = FastAPI(title="Pokemon Team Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clusterer_instance = None
tournament_data_cache = None

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
        # Process data
        df = process_tournament_data(tournament_data, "Worlds2024")
        clusterer = PokemonTeamClustering(df)
        
        clusterer.select_features(method='frequency', threshold=0.03)
        clusterer.select_features(method='variance', threshold=0.01)
        clusterer.normalize_features(pokemon_weight=1.0, move_weight=0.6)
        clusterer.cluster_teams()
        
        global clusterer_instance, tournament_data_cache
        clusterer_instance = clusterer
        tournament_data_cache = tournament_data

        archetypes = clusterer.identify_archetypes()
        return {
            "status": "success",
            "archetypes": numpy_to_python(archetypes),
        }
    except Exception as e:
        print(f"Error: {e}")
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
    
    if tournament_data_cache is None:
        raise HTTPException(status_code=400, detail="No tournament data available")
    
    try:
        # Get archetypes
        archetypes = clusterer_instance.identify_archetypes()
        print("Raw archetypes:", archetypes)  # Debug print
        
        # Get cluster stats
        cluster_stats = calculate_cluster_winrates(tournament_data_cache, clusterer_instance.labels_, clusterer_instance)
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
        
        response_data = {"archetypes": numpy_to_python(archetypes)}
        print("Final response:", response_data)  # Debug print
        return response_data
        
    except Exception as e:
        print(f"Error in get_archetypes: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))


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
        
@app.get("/cluster/{cluster_id}/top_teams")
async def get_cluster_top_teams(cluster_id: int, limit: int = 20):
    try:
        teams = get_top_teams_in_cluster(cluster_id, limit)
        return {
            "teams": teams
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
