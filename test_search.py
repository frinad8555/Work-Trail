import os
import sys
from dotenv import load_dotenv
import psycopg
from sentence_transformers import SentenceTransformer

load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("[ERROR] DATABASE_URL is not set!")
    sys.exit(1)

model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
print(f"[INFO] Loading embedding model: {model_name}...")
model = SentenceTransformer(model_name)

def hybrid_search(query: str, alpha: float = 0.65, top_k: int = 5):
    """
    Executes Hybrid Search:
      - S_semantic: Cosine similarity derived from 1 - (embedding <=> query_vec)
      - S_keyword: Normalized Full-Text Search ts_rank_cd
      - S_retrieval = alpha * S_semantic + (1 - alpha) * S_keyword
    """
    # 1. Generate query embedding
    query_vector = model.encode(query).tolist()
    query_vec_str = f"[{','.join(map(str, query_vector))}]"

    # 2. Hybrid SQL Query with clean parameter binding
    sql = """
    WITH scored AS (
        SELECT 
            id,
            title,
            company,
            location,
            employment_type,
            experience_required,
            -- Cosine similarity: (1 - cosine distance)
            (1 - (embedding <=> %s::vector)) AS semantic_score,
            -- Full-text keyword rank
            LEAST(ts_rank_cd(to_tsvector('english', title || ' ' || company || ' ' || description), plainto_tsquery('english', %s)), 1.0) AS keyword_score
        FROM jobs
    )
    SELECT 
        id,
        title,
        company,
        location,
        employment_type,
        experience_required,
        semantic_score,
        keyword_score,
        (%s * semantic_score + (1.0 - %s) * keyword_score) AS hybrid_score
    FROM scored
    ORDER BY hybrid_score DESC
    LIMIT %s;
    """

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                query_vec_str,  # %s for embedding <=> vector
                query,          # %s for plainto_tsquery
                alpha,          # %s for alpha * semantic_score
                alpha,          # %s for (1 - alpha) * keyword_score
                top_k           # %s for LIMIT
            ))
            results = cur.fetchall()
            return results

if __name__ == "__main__":
    test_queries = [
        "Distributed systems and microservices with Python",
        "React frontend developer with Tailwind CSS",
        "LLMs, LangGraph and AI agents",
        "Remote junior backend role"
    ]

    print("\n================ TESTING HYBRID SEARCH ================\n")
    for q in test_queries:
        print(f"\n🔍 Query: '{q}'")
        print("-" * 65)
        results = hybrid_search(q, alpha=0.65, top_k=3)
        for rank, r in enumerate(results, 1):
            job_id, title, company, loc, emp_type, exp, sem, kw, hybrid = r
            print(f"#{rank} {title} @ {company} [{loc}]")
            print(f"   Hybrid: {hybrid*100:.1f}% | Semantic: {sem*100:.1f}% | Keyword: {kw*100:.1f}%")
    print("\n=======================================================\n")
