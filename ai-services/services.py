import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg
from sentence_transformers import SentenceTransformer

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment or .env file")

if "sslmode" not in DATABASE_URL:
    delimiter = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = f"{DATABASE_URL}{delimiter}sslmode=require"

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

print(f"[INFO] Initializing SentenceTransformer model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_db_connection():
    return psycopg.connect(DATABASE_URL)

class SearchService:
    @staticmethod
    def get_embedding(text: str) -> List[float]:
        return embedding_model.encode(text).tolist()

    @staticmethod
    def hybrid_search(query: str, k_rrf: float = 60.0, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes Reciprocal Rank Fusion (RRF) Hybrid Search:
          - Semantic ranking via pgvector cosine similarity
          - Sparse lexical ranking via PostgreSQL FTS with Title 'A' weight boost
          - RRF Formula: RRF_score(d) = 1/(k + rank_sem) + (1/(k + rank_fts) if fts_match else 0)
        """
        query_vector = SearchService.get_embedding(query)
        query_vec_str = f"[{','.join(map(str, query_vector))}]"

        sql = """
        WITH semantic_ranks AS (
            SELECT 
                id, 
                title,
                ROW_NUMBER() OVER (ORDER BY (1 - (embedding <=> %s::vector)) DESC) as sem_rank,
                (1 - (embedding <=> %s::vector)) as sem_score
            FROM jobs
        ),
        fts_ranks AS (
            SELECT 
                id,
                title,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(
                        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(company, '')), 'B') ||
                        setweight(to_tsvector('english', COALESCE(description, '')), 'D'),
                        plainto_tsquery('english', %s),
                        1|4
                    ) DESC
                ) as fts_rank,
                ts_rank_cd(
                    setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(company, '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(description, '')), 'D'),
                    plainto_tsquery('english', %s),
                    1|4
                ) as fts_score
            FROM jobs
        ),
        scored_jobs AS (
            SELECT 
                j.id,
                j.title,
                j.company,
                j.location,
                j.employment_type,
                j.experience_required,
                j.description,
                j.application_url,
                sr.sem_score,
                fr.fts_score,
                sr.sem_rank,
                fr.fts_rank,
                ( (1.0 / (%s + sr.sem_rank)) + (CASE WHEN fr.fts_score > 0 THEN 1.0 / (%s + fr.fts_rank) ELSE 0 END) ) AS rrf_score,
                COALESCE(
                    ARRAY_AGG(s.skill_name) FILTER (WHERE s.skill_name IS NOT NULL), 
                    ARRAY[]::VARCHAR[]
                ) as required_skills
            FROM jobs j
            JOIN semantic_ranks sr ON j.id = sr.id
            JOIN fts_ranks fr ON j.id = fr.id
            LEFT JOIN job_skills js ON j.id = js.job_id
            LEFT JOIN skills s ON js.skill_id = s.id
            GROUP BY j.id, sr.sem_score, fr.fts_score, sr.sem_rank, fr.fts_rank
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
            ROUND(CAST(sem_score AS numeric), 4) as semantic_score,
            ROUND(CAST(fts_score AS numeric), 4) as keyword_score,
            ROUND(CAST(rrf_score AS numeric), 6) as rrf_score
        FROM scored_jobs
        ORDER BY rrf_score DESC
        LIMIT %s;
        """

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    query_vec_str, 
                    query_vec_str, 
                    query, 
                    query, 
                    k_rrf, 
                    k_rrf, 
                    top_k
                ))
                rows = cur.fetchall()
                
                results = []
                # Max possible RRF score for top-1 in both is 1/61 + 1/61 ~= 0.0328
                max_rrf = (1.0 / (k_rrf + 1)) * 2
                for row in rows:
                    rrf_val = float(row[11])
                    normalized_pct = min(100.0, round((rrf_val / max_rrf) * 100, 1))
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
                        "rrf_score": rrf_val,
                        "match_percentage": normalized_pct
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
        candidate_text = f"Skills: {', '.join(candidate_skills)}. Experience: {candidate_experience or ''}. Preferences: {candidate_preference or ''}"
        candidate_vec = SearchService.get_embedding(candidate_text)
        candidate_vec_str = f"[{','.join(map(str, candidate_vec))}]"

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

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (candidate_vec_str,))
                rows = cur.fetchall()

                scored_results = []
                for row in rows:
                    job_id, title, company, loc, emp_type, exp_req, desc, app_url, sem_score, req_skills = row
                    req_skills_set = {s.strip().lower() for s in req_skills}
                    
                    matched_skills = [s for s in req_skills if s.strip().lower() in candidate_skills_lower]
                    missing_skills = [s for s in req_skills if s.strip().lower() not in candidate_skills_lower]
                    
                    s_skills = len(matched_skills) / len(req_skills_set) if req_skills_set else 0.5
                    s_semantic = max(0.0, float(sem_score))
                    s_exp = 0.8
                    if candidate_experience and exp_req:
                        if "intern" in candidate_experience.lower() and "intern" in exp_req.lower():
                            s_exp = 1.0
                        elif "0-" in exp_req:
                            s_exp = 0.95

                    s_pref = 0.7
                    if candidate_preference:
                        pref_clean = candidate_preference.lower()
                        if "remote" in pref_clean and "remote" in (loc.lower() + " " + emp_type.lower()):
                            s_pref = 1.0
                        elif any(city in loc.lower() for city in ["bengaluru", "hyderabad", "pune", "mumbai"] if city in pref_clean):
                            s_pref = 1.0

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

                scored_results.sort(key=lambda x: x["final_compatibility_score"], reverse=True)
                return scored_results[:top_k]
