# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import os
from backend.database import get_top_teams_in_cluster, find_teams_from_cluster, engine
from sqlalchemy import text

app = FastAPI(title="Pokemon Team Analysis API")

# CORS setup
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "https://darcyjfmccarthy.github.io",
    "https://poliboard-swzt.onrender.com",
    "https://poliboard-swzt.onrender.com/archetypes",
    "https://poliboard.net"
    # Add your frontend URL when you deploy it
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing endpoints...
@app.get("/cluster/{cluster_id}/top_teams")
async def get_cluster_top_teams(cluster_id: int, limit: int = 20):
    try:
        teams_df, stats = find_teams_from_cluster(engine, cluster_id, limit)
        
        teams_list = []
        for _, row in teams_df.iterrows():
            team_obj = {
                "Competition": row["Competition"],
                "Name": row["Name"],
                "Nationality": row["Nationality"],
                "Wins": int(row["Wins"]),
                "Losses": int(row["Losses"]),
                "pokemon": [col for col in teams_df.columns 
                           if not col.startswith(('Competition', 'Name', 'Nationality', 'Wins', 'Losses', 'move_'))
                           and row[col] == 1]
            }
            teams_list.append(team_obj)
            
        return {
            "teams": teams_list,
            "stats": stats
        }
        
    except Exception as e:
        print(f"Error in get_cluster_top_teams: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/archetypes")
async def get_archetypes_from_db():
    try:
        with engine.connect() as conn:
            cluster_data = conn.execute(text("SELECT cluster_id, core_pokemon FROM cluster_features"))
            archetypes = {}
            
            for row in cluster_data:
                cluster_id = row[0]
                pokemon = row[1]
                if cluster_id not in archetypes:
                    archetypes[cluster_id] = []
                archetypes[cluster_id].append(pokemon)
            
            formatted_archetypes = {
                f"Cluster_{cluster_id}": {
                    "core_pokemon": pokemons
                } for cluster_id, pokemons in archetypes.items()
            }
            
            return formatted_archetypes
            
    except Exception as e:
        print(f"Error fetching archetypes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/total_teams")
async def get_total_teams():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM tournament_teams"))
            count = result.scalar()
            return {"total": count}
    except Exception as e:
        print(f"Error getting total teams: {str(e)}")
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