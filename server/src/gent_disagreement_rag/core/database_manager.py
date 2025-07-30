import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


class DatabaseManager:
    """
    Manages database connections and operations for the A Gentleman's Disagreement RAG application.
    """

    def __init__(
        self,
        host=None,
        port=None,
        user=None,
        password=None,
        database=None,
    ):
        """
        Initialize the database manager with connection parameters.
        Loads from environment variables with sensible defaults.
        """
        # Load environment variables
        load_dotenv()

        self.connection_params = {
            "host": host or os.getenv("DB_HOST", "localhost"),
            "port": port or int(os.getenv("DB_PORT", "5432")),
            "user": user or os.getenv("DB_USER", "postgres"),
            "password": password
            or os.getenv("DB_PASSWORD", ""),  # No default for security
            "database": database or os.getenv("DB_NAME", "gent_disagreement"),
        }

        if not self.connection_params["password"]:
            raise ValueError(
                "Database password must be provided via DB_PASSWORD environment variable"
            )

    def get_connection(self):
        """
        Create and return a database connection.
        """
        return psycopg2.connect(**self.connection_params)

    def setup_database(self):
        """
        Set up the database with required extensions and tables.
        """
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_segments (
                    id SERIAL PRIMARY KEY,
                    speaker TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(1536)
                );
            """
            )
            conn.commit()
            print("Database setup complete!")
        except Exception as e:
            print("Error during setup:", e)
        finally:
            cur.close()
            conn.close()

    def insert_transcript_segment_with_embedding(self, speaker, text, embedding):
        """
        Insert a single transcript segment with its embedding into the database.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO transcript_segments (speaker, text, embedding) VALUES (%s, %s, %s)",
                (speaker, text, embedding),
            )
            conn.commit()
            print(f"Stored embedding for: {text[:50]}...")
        except Exception as e:
            print("Error inserting transcript segment:", e)
        finally:
            cursor.close()
            conn.close()

    def execute_query(self, query, params=None):
        """
        Execute a query with optional parameters.
        """
        cursor = None
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
        except Exception as e:
            print("Error executing query:", e)
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
