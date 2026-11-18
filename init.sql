-- ====================================================================
-- CareerGraph Database Schema Definition
-- Supports relational entities, Full-Text Search, and pgvector embeddings
-- ====================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Candidates Table
CREATE TABLE IF NOT EXISTS candidates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    education JSONB,
    experience JSONB,
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Skills Table (Standardized master skills list)
CREATE TABLE IF NOT EXISTS skills (
    id SERIAL PRIMARY KEY,
    skill_name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(100) -- e.g., 'Programming Language', 'Database', 'Framework', 'Cloud'
);

-- 4. Candidate Skills Junction Table
CREATE TABLE IF NOT EXISTS candidate_skills (
    candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    proficiency VARCHAR(50), -- e.g., 'Beginner', 'Intermediate', 'Advanced'
    PRIMARY KEY (candidate_id, skill_id)
);

-- 5. Jobs Table
-- Note: Embedding dimension is 384 for all-MiniLM-L6-v2 (or 1536 for OpenAI text-embedding-3-small)
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    employment_type VARCHAR(50) DEFAULT 'Full-time', -- e.g., 'Full-time', 'Internship', 'Remote', 'Hybrid'
    experience_required VARCHAR(100),
    description TEXT NOT NULL,
    application_url TEXT,
    posted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding vector(384)
);

-- Full-Text Search index for lexical keyword matching
CREATE INDEX IF NOT EXISTS idx_jobs_fts ON jobs 
USING gin(to_tsvector('english', title || ' ' || company || ' ' || description));

-- HNSW Vector index for fast cosine distance matching
CREATE INDEX IF NOT EXISTS idx_jobs_embedding ON jobs 
USING hnsw (embedding vector_cosine_ops);

-- 6. Job Skills Junction Table
CREATE TABLE IF NOT EXISTS job_skills (
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    importance VARCHAR(50) DEFAULT 'Required', -- 'Required' or 'Preferred'
    PRIMARY KEY (job_id, skill_id)
);

-- 7. Applications and Saved Jobs
CREATE TABLE IF NOT EXISTS applications (
    id SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'Saved', -- 'Saved', 'Applied', 'Interviewing', 'Accepted', 'Rejected'
    applied_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id, job_id)
);

-- 8. Agent Conversations Table (for LangGraph state & session memory)
CREATE TABLE IF NOT EXISTS agent_conversations (
    conversation_id VARCHAR(255) PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(id) ON DELETE SET NULL,
    conversation_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
