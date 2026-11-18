import os
import sys
import json
from dotenv import load_dotenv
import psycopg
from sentence_transformers import SentenceTransformer

load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("[ERROR] DATABASE_URL is not set in .env!")
    sys.exit(1)

# Initialize local embedding model
model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
print(f"[INFO] Loading embedding model: {model_name}...")
model = SentenceTransformer(model_name)

sample_jobs = [
    {
        "title": "Backend Engineering Intern",
        "company": "CloudScale Systems",
        "location": "Bengaluru, India (Hybrid)",
        "employment_type": "Internship",
        "experience_required": "0-1 years",
        "description": "Looking for a backend intern to build high-throughput microservices using Python, FastAPI, and PostgreSQL. You will work on Redis caching, background worker queues, and REST APIs. Knowledge of distributed systems and Docker is a plus.",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "REST APIs"],
        "application_url": "https://careers.cloudscale.example.com/backend-intern"
    },
    {
        "title": "Junior Full Stack Developer",
        "company": "Nexus HealthTech",
        "location": "Remote",
        "employment_type": "Full-time",
        "experience_required": "0-2 years",
        "description": "Join our product engineering team to build scalable healthcare portals. You will develop modern web interfaces using React, TypeScript, and Tailwind CSS, while building backend services with Node.js, Express, and PostgreSQL.",
        "skills": ["React", "TypeScript", "Node.js", "Express", "Tailwind CSS", "PostgreSQL"],
        "application_url": "https://nexushealth.example.com/junior-fullstack"
    },
    {
        "title": "Associate AI Engineer",
        "company": "CognitiveLabs",
        "location": "Hyderabad, India (In-office)",
        "employment_type": "Full-time",
        "experience_required": "0-2 years",
        "description": "We are seeking an enthusiastic AI engineer to develop agentic LLM workflows and semantic search pipelines. Experience with LangChain, LangGraph, vector databases like pgvector/Pinecone, and Python is required. Familiarity with RAG evaluation is desirable.",
        "skills": ["Python", "LangChain", "LangGraph", "Vector Databases", "pgvector", "RAG", "LLMs"],
        "application_url": "https://cognitivelabs.example.com/ai-engineer"
    },
    {
        "title": "Frontend Engineer - React",
        "company": "FinFlow Technologies",
        "location": "Mumbai, India (Hybrid)",
        "employment_type": "Full-time",
        "experience_required": "1-3 years",
        "description": "FinFlow is looking for a frontend developer to build responsive financial dashboards and interactive charts. Must have strong expertise in React, Redux Toolkit, Tailwind CSS, and WebSockets for real-time trade updates.",
        "skills": ["React", "JavaScript", "Redux", "Tailwind CSS", "WebSockets"],
        "application_url": "https://finflow.example.com/frontend-react"
    },
    {
        "title": "Machine Learning Intern",
        "company": "VisionMetrics AI",
        "location": "Remote",
        "employment_type": "Internship",
        "experience_required": "0-1 years",
        "description": "Work with our computer vision and NLP team to train and evaluate PyTorch models. You will preprocess datasets, run model benchmarks using Hugging Face transformers, and optimize inference latency using ONNX and FastAPI.",
        "skills": ["Python", "PyTorch", "Hugging Face", "NLP", "Computer Vision", "Scikit-learn"],
        "application_url": "https://visionmetrics.example.com/ml-intern"
    },
    {
        "title": "DevOps & Cloud Engineer",
        "company": "DataStream Corp",
        "location": "Pune, India (Hybrid)",
        "employment_type": "Full-time",
        "experience_required": "1-3 years",
        "description": "Manage Kubernetes clusters, CI/CD deployment pipelines with GitHub Actions, and AWS cloud infrastructure. You will automate monitoring with Prometheus and Grafana and manage PostgreSQL and Redis clusters.",
        "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Linux", "Terraform", "PostgreSQL"],
        "application_url": "https://datastream.example.com/devops-engineer"
    },
    {
        "title": "Junior Python Developer",
        "company": "AlgoTrading Partners",
        "location": "Remote",
        "employment_type": "Full-time",
        "experience_required": "0-2 years",
        "description": "Develop automated data ingestion pipelines, financial market scrapers, and data analysis algorithms. Strong command of Python, Pandas, NumPy, SQL, and asynchronous programming with AsyncIO required.",
        "skills": ["Python", "Pandas", "NumPy", "SQL", "AsyncIO", "REST APIs"],
        "application_url": "https://algotrading.example.com/python-dev"
    },
    {
        "title": "Software Engineer - Java Backend",
        "company": "Enterprise Horizon",
        "location": "Bengaluru, India (In-office)",
        "employment_type": "Full-time",
        "experience_required": "1-3 years",
        "description": "Design and build enterprise backend microservices using Java, Spring Boot, and Kafka. Experience with relational databases (PostgreSQL/Oracle), distributed caching, and clean architecture principles required.",
        "skills": ["Java", "Spring Boot", "Kafka", "PostgreSQL", "Microservices", "Docker"],
        "application_url": "https://enterprisehorizon.example.com/java-backend"
    }
]

def seed_database():
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                print(f"[INFO] Seeding {len(sample_jobs)} jobs with vector embeddings...")
                
                for idx, job in enumerate(sample_jobs, 1):
                    # Combine title, company, skills, and description into embedding text
                    embed_text = f"{job['title']} at {job['company']}. Location: {job['location']}. Skills: {', '.join(job['skills'])}. Description: {job['description']}"
                    
                    # Generate 384-dim embedding
                    embedding_vector = model.encode(embed_text).tolist()
                    embedding_str = f"[{','.join(map(str, embedding_vector))}]"

                    # Insert Job
                    cur.execute("""
                        INSERT INTO jobs (title, company, location, employment_type, experience_required, description, application_url, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        RETURNING id;
                    """, (
                        job["title"],
                        job["company"],
                        job["location"],
                        job["employment_type"],
                        job["experience_required"],
                        job["description"],
                        job["application_url"],
                        embedding_str
                    ))
                    job_id = cur.fetchone()[0]

                    # Insert Skills & Job-Skill relationships
                    for skill_name in job["skills"]:
                        # Insert skill if not exists
                        cur.execute("""
                            INSERT INTO skills (skill_name, category)
                            VALUES (%s, 'General')
                            ON CONFLICT (skill_name) DO NOTHING;
                        """, (skill_name,))

                        # Get skill id
                        cur.execute("SELECT id FROM skills WHERE skill_name = %s;", (skill_name,))
                        skill_id = cur.fetchone()[0]

                        # Link in job_skills
                        cur.execute("""
                            INSERT INTO job_skills (job_id, skill_id, importance)
                            VALUES (%s, %s, 'Required')
                            ON CONFLICT (job_id, skill_id) DO NOTHING;
                        """, (job_id, skill_id))

                    print(f"  [{idx}/{len(sample_jobs)}] Inserted: {job['title']} ({job['company']})")

                conn.commit()
                print("\n[SUCCESS] All sample jobs and embeddings seeded successfully into Neon!")

    except Exception as e:
        print(f"\n[ERROR] Seeding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    seed_database()
