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

    def find_relevant_above_adaptive_threshold(self, query, min_docs=3, max_docs=10, similarity_threshold=0.6):
        """
        Clean two-query approach: try threshold first, fallback if needed

        Args:
            query: Search query
            min_docs: Minimum number of documents to return
            max_docs: Maximum number of documents to return
            similarity_threshold: Minimum similarity score for inclusion

        Returns:
            List of relevant documents above threshold
        """
        try:
            embedding = self.embedding_service.generate_embedding(query)

            # Query 1: Try to get results above threshold
            threshold_sql = """
                SELECT
                    ts.speaker,
                    ts.text,
                    e.episode_number,
                    e.title,
                    e.date_published,
                    1 - (ts.embedding <=> %s::vector) as similarity
                FROM transcript_segments ts
                JOIN episodes e ON ts.episode_id = e.id
                WHERE 1 - (ts.embedding <=> %s::vector) >= %s
                ORDER BY similarity DESC
                LIMIT %s
            """

            results = self.db_manager.execute_query(
                threshold_sql, (embedding, embedding, similarity_threshold, max_docs)
            )

            # If we have enough results above threshold, return them
            if len(results) >= min_docs:
                return results

            # Query 2: Fallback to top N results regardless of threshold
            fallback_sql = """
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

            fallback_results = self.db_manager.execute_query(
                fallback_sql, (embedding, min_docs)
            )

            return fallback_results

        except Exception as e:
            print(f"Error finding relevant transcript segments above adaptive threshold: {e}")
            raise e
