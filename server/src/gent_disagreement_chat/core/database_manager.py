import os
from typing import List, Optional

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

    def get_max_episode_number(self) -> Optional[int]:
        """Get the maximum episode number from the database."""
        try:
            query = "SELECT MAX(episode_number) as max_episode FROM episodes"
            result = self.execute_query(query)
            return (
                result[0]["max_episode"]
                if result and result[0]["max_episode"]
                else None
            )
        except Exception as e:
            print(f"Error getting max episode number: {e}")
            return None

    def get_latest_episode_numbers(self, n: int) -> List[int]:
        """Get the top N episode numbers ordered by episode_number DESC."""
        try:
            query = """
                SELECT episode_number 
                FROM episodes 
                ORDER BY episode_number DESC 
                LIMIT %s
            """
            results = self.execute_query(query, (n,))
            return [row["episode_number"] for row in results]
        except Exception as e:
            print(f"Error getting latest episode numbers: {e}")
            return []
