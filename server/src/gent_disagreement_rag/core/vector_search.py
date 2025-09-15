from .database_manager import DatabaseManager
from .embedding_service import EmbeddingService


class VectorSearch:
    """Handles vector search similarity search operations"""

    def __init__(self, database_name="gent_disagreement"):
        self.db_manager = DatabaseManager(database=database_name)
        self.embedding_service = EmbeddingService(self.db_manager)

    def find_most_similar(self, query, limit=5):
        """Find the most similar transcript segments without threshold filtering."""
        try:
            embedding = self.embedding_service.generate_embedding(query)

            query_sql = """
                SELECT
                    ts.speaker,
                    ts.text,
                    e.episode_number,
                    e.title,
                    e.date_published,
                    1 - (ts.embedding <=> %s::vector) as similarity
                FROM transcript_segments ts
                JOIN episodes e ON ts.episode_id = e.id
                ORDER BY similarity DESC
                LIMIT %s
                """

            results = self.db_manager.execute_query(query_sql, (embedding, limit))
            return results

        except Exception as e:
            print(f"Error finding most similar transcript segments: {e}")
            raise e
