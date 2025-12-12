from typing import Any, Dict, List, Optional, Tuple

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
                    s.name,
                    ts.text,
                    e.episode_number,
                    e.title,
                    e.date_published,
                    1 - (ts.embedding <=> %s::vector) as similarity
                FROM transcript_segments ts
                JOIN episodes e ON ts.episode_id = e.episode_number
                JOIN speakers s ON ts.speaker_id = s.id
                ORDER BY similarity DESC
                LIMIT %s
                """

            results = self.db_manager.execute_query(query_sql, (embedding, limit))
            return results

        except Exception as e:
            print(f"Error finding most similar transcript segments: {e}")
            raise e

    def find_relevant_above_adaptive_threshold(
        self, query, min_docs=3, max_docs=10, similarity_threshold=0.6
    ):
        """Find relevant segments above threshold, fallback to top N if needed."""
        try:
            embedding = self.embedding_service.generate_embedding(query)

            # Query 1: Try to get results above threshold
            threshold_sql = """
                SELECT
                    s.name,
                    ts.text,
                    e.episode_number,
                    e.title,
                    e.date_published,
                    1 - (ts.embedding <=> %s::vector) as similarity
                FROM transcript_segments ts
                JOIN episodes e ON ts.episode_id = e.episode_number
                JOIN speakers s ON ts.speaker_id = s.id
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
                    s.name,
                    ts.text,
                    e.episode_number,
                    e.title,
                    e.date_published,
                    1 - (ts.embedding <=> %s::vector) as similarity
                FROM transcript_segments ts
                JOIN episodes e ON ts.episode_id = e.episode_number
                JOIN speakers s ON ts.speaker_id = s.id
                ORDER BY similarity DESC
                LIMIT %s
            """

            fallback_results = self.db_manager.execute_query(
                fallback_sql, (embedding, min_docs)
            )

            return fallback_results

        except Exception as e:
            print(
                f"Error finding relevant transcript segments above adaptive threshold: {e}"
            )
            raise e

    def find_relevant_with_filters(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        min_docs: int = 3,
        max_docs: int = 10,
        similarity_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant transcript segments with optional metadata filtering.

        Extends the adaptive threshold approach with dynamic WHERE clauses
        based on extracted filters (episode, speaker, date range). This enables
        precise retrieval for scoped queries while maintaining fallback behavior.

        Args:
            query: Search query text
            filters: Dict with 'episode_number', 'speaker', 'date_range' keys
            min_docs: Minimum documents to return
            max_docs: Maximum documents to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of transcript segment dictionaries
        """
        try:
            embedding = self.embedding_service.generate_embedding(query)

            # Build dynamic WHERE clause and parameters based on filters.
            # This allows flexible filtering without SQL injection risks.
            where_clauses, filter_params = self._build_filter_clauses(filters)

            # Combine similarity threshold with metadata filters
            # Base WHERE clause always includes the similarity threshold.
            base_where = "WHERE 1 - (ts.embedding <=> %s::vector) >= %s"
            if where_clauses:
                full_where = f"{base_where} AND {' AND '.join(where_clauses)}"
            else:
                full_where = base_where

            # Query 1: Try to get results above threshold with filters
            threshold_sql = f"""
                SELECT
                    s.name,
                    ts.text,
                    e.episode_number,
                    e.title,
                    e.date_published,
                    1 - (ts.embedding <=> %s::vector) as similarity
                FROM transcript_segments ts
                JOIN episodes e ON ts.episode_id = e.episode_number
                JOIN speakers s ON ts.speaker_id = s.id
                {full_where}
                ORDER BY similarity DESC
                LIMIT %s
            """

            # Build parameter tuple: (embedding for SELECT, embedding for WHERE, threshold, filters..., limit)
            threshold_params = (
                embedding,
                embedding,
                similarity_threshold,
                *filter_params,
                max_docs,
            )

            results = self.db_manager.execute_query(threshold_sql, threshold_params)

            # If we have enough results above threshold, return them
            if len(results) >= min_docs:
                return results

            # Query 2: Fallback to top N results with filters (no threshold)
            # This ensures we always return some results even for niche queries.
            fallback_where = (
                f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            )

            fallback_sql = f"""
                SELECT
                    s.name,
                    ts.text,
                    e.episode_number,
                    e.title,
                    e.date_published,
                    1 - (ts.embedding <=> %s::vector) as similarity
                FROM transcript_segments ts
                JOIN episodes e ON ts.episode_id = e.episode_number
                JOIN speakers s ON ts.speaker_id = s.id
                {fallback_where}
                ORDER BY similarity DESC
                LIMIT %s
            """

            fallback_params = (embedding, *filter_params, min_docs)
            fallback_results = self.db_manager.execute_query(
                fallback_sql, fallback_params
            )

            return fallback_results

        except Exception as e:
            print(f"Error finding relevant transcript segments with filters: {e}")
            raise e

    def _build_filter_clauses(
        self, filters: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], List[Any]]:
        """
        Build SQL WHERE clauses and parameters from a filter dictionary.

        Converts the filter dictionary into SQL-safe WHERE clause fragments
        and corresponding parameter values for parameterized queries.

        Args:
            filters: Filter dictionary from QueryParser.extract_filters().
                     Example: {'episode_number': 180, 'speaker': 'Ricky Ghoshroy', 'date_range': None}

        Returns:
            Tuple of (where_clauses, parameters):
                - where_clauses: List of SQL WHERE clause fragments (without AND).
                  Example: ['e.episode_number = %s', 's.name = %s']
                - parameters: List of values for parameterized query.
                  Example: [180, 'Ricky Ghoshroy']

        Data transformation:
            Input: {'episode_number': 180, 'speaker': 'Ricky Ghoshroy', 'date_range': None}
                   ↓
            Output: (
                ['e.episode_number = %s', 's.name = %s'],
                [180, 'Ricky Ghoshroy']
            )

        """
        if not filters:
            return [], []

        where_clauses = []
        params = []

        if filters.get("episode_number") is not None:
            where_clauses.append("e.episode_number = %s")
            params.append(filters["episode_number"])

        # Note: "guest" and "professor" are special values that may need
        # different handling in future iterations (partial match, etc.)
        if filters.get("speaker") is not None:
            where_clauses.append("s.name = %s")
            params.append(filters["speaker"])

        if filters.get("date_range") is not None:
            start_date, end_date = filters["date_range"]
            where_clauses.append("e.date_published BETWEEN %s AND %s")
            params.append(start_date)
            params.append(end_date)

        return where_clauses, params
