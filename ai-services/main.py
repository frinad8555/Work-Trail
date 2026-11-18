import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from services import SearchService

app = FastAPI(
    title="CareerGraph AI Service",
    description="Agentic AI Job Discovery, Hybrid Search, and Explainable Recommendation Engine",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Schemas =================

class SearchRequest(BaseModel):
    query: str = Field(..., example="Backend engineer Python distributed systems", description="Natural language search query")
    alpha: float = Field(0.65, ge=0.0, le=1.0, description="Weight of semantic search vs keyword search (0.0 to 1.0)")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to retrieve")

class RecommendRequest(BaseModel):
    skills: List[str] = Field(..., example=["Python", "FastAPI", "PostgreSQL", "Docker"])
    experience: Optional[str] = Field("0-1 years", example="0-1 years")
    preferences: Optional[str] = Field("Remote or Hybrid", example="Remote or Hybrid")
    top_k: int = Field(5, ge=1, le=20)

class EmbedRequest(BaseModel):
    text: str = Field(..., example="Senior React Developer with Redux and TypeScript")

# ================= Routes =================

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "online",
        "service": "CareerGraph AI Service",
        "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    }

@app.post("/api/search", tags=["Retrieval"])
def search_jobs(req: SearchRequest):
    """
    Executes Hybrid Retrieval combining:
      1. Dense vector similarity via pgvector (HNSW index)
      2. Sparse lexical search via PostgreSQL Full-Text Search
      3. Linear score fusion: S_retrieval = alpha * S_semantic + (1 - alpha) * S_keyword
    """
    try:
        results = SearchService.hybrid_search(
            query=req.query,
            alpha=req.alpha,
            top_k=req.top_k
        )
        return {
            "query": req.query,
            "alpha": req.alpha,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/recommend", tags=["Recommendation"])
def get_recommendations(req: RecommendRequest):
    """
    Calculates multi-factor personalized job compatibility:
      S(u, j) = w1 * S_semantic + w2 * S_skills + w3 * S_exp + w4 * S_pref
    Provides explainable match breakdown:
      - Matched skills
      - Skill gaps (missing skills)
      - Category percentage scores
    """
    try:
        recommendations = SearchService.recommend_jobs(
            candidate_skills=req.skills,
            candidate_experience=req.experience,
            candidate_preference=req.preferences,
            top_k=req.top_k
        )
        return {
            "candidate_profile": {
                "skills": req.skills,
                "experience": req.experience,
                "preferences": req.preferences
            },
            "count": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/api/embed", tags=["Embeddings"])
def generate_embedding(req: EmbedRequest):
    """
    Generates a 384-dimensional vector embedding for any text input.
    Used by background job ingestion pipelines.
    """
    try:
        vector = SearchService.get_embedding(req.text)
        return {
            "dimension": len(vector),
            "embedding": vector
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
