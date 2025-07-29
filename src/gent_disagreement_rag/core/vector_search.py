from .database_manager import DatabaseManager
from .embedding_service import EmbeddingService


class VectorSearch:
    """Handles vector search similarity search operations"""

    def __init__(self, database_name="gent_disagreement"):
        self.db_manager = DatabaseManager(database=database_name)
        self.embedding_service = EmbeddingService(self.db_manager)

    def search_similar_transcript_segments(
        self, query, limit=5, similarity_threshold=0.4
    ):
        """Search for similar transcript segments using vector similarity."""
        try:
            embedding = self.embedding_service.generate_embedding(query)

            query_sql = """
                SELECT
                    speaker,
                    text,
                    1 - (embedding <=> %s::vector) as similarity
                FROM transcript_segments
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY similarity DESC
                LIMIT %s
                """

            results = self.db_manager.execute_query(
                query_sql, (embedding, embedding, similarity_threshold, limit)
            )

            return results

        except Exception as e:
            print(f"Error searching transcript segments: {e}")
            raise e

    def find_similar_above_threshold(self, query, threshold=0.5, limit=5):
        """Find similar transcript segments above a specific threshold."""
        return self.search_similar_transcript_segments(query, limit, threshold)

    def find_most_similar(self, query, limit=5):
        """Find the most similar transcript segments without threshold filtering."""
        try:
            embedding = self.embedding_service.generate_embedding(query)

            query_sql = """
                SELECT 
                    speaker,
                    text,
                    1 - (embedding <=> %s::vector) as similarity
                FROM transcript_segments
                ORDER BY similarity DESC
                LIMIT %s
                """

            results = self.db_manager.execute_query(query_sql, (embedding, limit))
            return results

        except Exception as e:
            print(f"Error finding most similar transcript segments: {e}")
            raise e
