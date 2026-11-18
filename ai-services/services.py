import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment or .env file")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Initialize model (singleton in memory)
print(f"[INFO] Initializing SentenceTransformer model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Connection Pool for high concurrency
db_pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=10,
    open=True
)

class SearchService:
    @staticmethod
    def get_embedding(text: str) -> List[float]:
        return embedding_model.encode(text).tolist()

    @staticmethod
    def hybrid_search(query: str, alpha: float = 0.65, top_k: int = 10) -> List[Dict[str, Any]]:
        query_vector = SearchService.get_embedding(query)
        query_vec_str = f"[{','.join(map(str, query_vector))}]"

        sql = """
        WITH scored AS (
            SELECT 
                j.id,
                j.title,
                j.company,
                j.location,
                j.employment_type,
                j.experience_required,
                j.description,
                j.application_url,
                (1 - (j.embedding <=> %s::vector)) AS semantic_score,
                LEAST(ts_rank_cd(to_tsvector('english', j.title || ' ' || j.company || ' ' || j.description), plainto_tsquery('english', %s)), 1.0) AS keyword_score,
                COALESCE(
                    ARRAY_AGG(s.skill_name) FILTER (WHERE s.skill_name IS NOT NULL), 
                    ARRAY[]::VARCHAR[]
                ) as required_skills
            FROM jobs j
            LEFT JOIN job_skills js ON j.id = js.job_id
            LEFT JOIN skills s ON js.skill_id = s.id
            GROUP BY j.id
        )
        SELECT 
            id,
            title,
            company,
            location,
            employment_type,
            experience_required,
            description,
            application_url,
            required_skills,
            ROUND(CAST(semantic_score AS numeric), 4) as semantic_score,
            ROUND(CAST(keyword_score AS numeric), 4) as keyword_score,
            ROUND(CAST((%s * semantic_score + (1.0 - %s) * keyword_score) AS numeric), 4) as hybrid_score
        FROM scored
        ORDER BY hybrid_score DESC
        LIMIT %s;
        """

        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query_vec_str, query, alpha, alpha, top_k))
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "title": row[1],
                        "company": row[2],
                        "location": row[3],
                        "employment_type": row[4],
                        "experience_required": row[5],
                        "description": row[6],
                        "application_url": row[7],
                        "required_skills": row[8],
                        "semantic_score": float(row[9]),
                        "keyword_score": float(row[10]),
                        "hybrid_score": float(row[11]),
                        "match_percentage": round(float(row[11]) * 100, 1)
                    })
                return results

    @staticmethod
    def recommend_jobs(
        candidate_skills: List[str],
        candidate_experience: Optional[str] = None,
        candidate_preference: Optional[str] = None,
        w_semantic: float = 0.40,
        w_skills: float = 0.35,
        w_exp: float = 0.15,
        w_pref: float = 0.10,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Multi-factor personalized ranking:
        S(u, j) = w1 * S_semantic + w2 * S_skills + w3 * S_exp + w4 * S_pref
        """
        candidate_text = f"Skills: {', '.join(candidate_skills)}. Experience: {candidate_experience or ''}. Preferences: {candidate_preference or ''}"
        candidate_vec = SearchService.get_embedding(candidate_text)
        candidate_vec_str = f"[{','.join(map(str, candidate_vec))}]"

        # Fetch jobs and skills
        sql = """
        SELECT 
            j.id,
            j.title,
            j.company,
            j.location,
            j.employment_type,
            j.experience_required,
            j.description,
            j.application_url,
            (1 - (j.embedding <=> %s::vector)) AS semantic_score,
            COALESCE(
                ARRAY_AGG(s.skill_name) FILTER (WHERE s.skill_name IS NOT NULL), 
                ARRAY[]::VARCHAR[]
            ) as required_skills
        FROM jobs j
        LEFT JOIN job_skills js ON j.id = js.job_id
        LEFT JOIN skills s ON js.skill_id = s.id
        GROUP BY j.id;
        """

        candidate_skills_lower = {s.strip().lower() for s in candidate_skills}

        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (candidate_vec_str,))
                rows = cur.fetchall()

                scored_results = []
                for row in rows:
                    job_id, title, company, loc, emp_type, exp_req, desc, app_url, sem_score, req_skills = row
                    
                    req_skills_set = {s.strip().lower() for s in req_skills}
                    
                    # 1. Skill Overlap Score (Jaccard / Recall of required skills)
                    matched_skills = [s for s in req_skills if s.strip().lower() in candidate_skills_lower]
                    missing_skills = [s for s in req_skills if s.strip().lower() not in candidate_skills_lower]
                    
                    if req_skills_set:
                        s_skills = len(matched_skills) / len(req_skills_set)
                    else:
                        s_skills = 0.5  # Neutral if no explicit skills listed
                    
                    # 2. Semantic Similarity Score
                    s_semantic = max(0.0, float(sem_score))
                    
                    # 3. Experience Compatibility Score
                    s_exp = 0.8  # Default compatible baseline
                    if candidate_experience and exp_req:
                        if "intern" in candidate_experience.lower() and "intern" in exp_req.lower():
                            s_exp = 1.0
                        elif "0-" in exp_req:
                            s_exp = 0.95

                    # 4. Location / Work preference score
                    s_pref = 0.7
                    if candidate_preference:
                        pref_clean = candidate_preference.lower()
                        if "remote" in pref_clean and "remote" in (loc.lower() + " " + emp_type.lower()):
                            s_pref = 1.0
                        elif any(city in loc.lower() for city in ["bengaluru", "hyderabad", "pune", "mumbai"] if city in pref_clean):
                            s_pref = 1.0

                    # Multi-factor formula from Section 4.3
                    final_score = (
                        w_semantic * s_semantic +
                        w_skills * s_skills +
                        w_exp * s_exp +
                        w_pref * s_pref
                    )

                    scored_results.append({
                        "id": job_id,
                        "title": title,
                        "company": company,
                        "location": loc,
                        "employment_type": emp_type,
                        "experience_required": exp_req,
                        "description": desc,
                        "application_url": app_url,
                        "required_skills": req_skills,
                        "matched_skills": matched_skills,
                        "missing_skills": missing_skills,
                        "score_breakdown": {
                            "semantic": round(s_semantic * 100, 1),
                            "skills_match": round(s_skills * 100, 1),
                            "experience": round(s_exp * 100, 1),
                            "preference": round(s_pref * 100, 1)
                        },
                        "final_compatibility_score": round(final_score * 100, 1),
                        "explanation": f"Matches {len(matched_skills)}/{len(req_skills)} required skills. Role aligns with your target profile in {loc}."
                    })

                # Sort by final compatibility score descending
                scored_results.sort(key=lambda x: x["final_compatibility_score"], reverse=True)
                return scored_results[:top_k]
