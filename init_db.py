import os
import sys
from dotenv import load_dotenv
import psycopg

load_dotenv()

db_url = os.getenv("DATABASE_URL")

if not db_url:
    print("[ERROR] DATABASE_URL is not set in .env! Please add your Neon connection string.")
    sys.exit(1)

print("[INFO] Connecting to PostgreSQL database...")

try:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Read init.sql
            sql_path = os.path.join(os.path.dirname(__file__), "init.sql")
            with open(sql_path, "r", encoding="utf-8") as f:
                sql_script = f.read()
            
            print("[INFO] Executing init.sql migrations...")
            cur.execute(sql_script)
            conn.commit()

            # Verify tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            # Verify vector extension
            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            vector_ext = cur.fetchone()

            print("\n================ SUCCESS ================")
            print(f"[OK] pgvector extension: {vector_ext[0]} (version {vector_ext[1]})" if vector_ext else "[WARN] pgvector extension not found!")
            print(f"[OK] Created Tables ({len(tables)}):")
            for t in tables:
                print(f"  - {t}")
            print("=========================================\n")

except Exception as e:
    print(f"\n[ERROR] Migration failed: {e}\n")
    sys.exit(1)
