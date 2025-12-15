"""
Query Parser Module for RAG Retrieval System.

Extracts structured information from natural language queries to enable
metadata-based filtering in vector search.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

SPEAKER_MAPPINGS = {
    "ricky": "Ricky Ghoshroy",
    "brendan": "Brendan Kelly",
    "ricky ghoshroy": "Ricky Ghoshroy",
    "brendan kelly": "Brendan Kelly",
}

SPECIAL_SPEAKER_KEYWORDS = {
    "hosts": None,
    "host": None,
    "both": None,
    "guest": "guest",
    "professor": "professor",
}

# Keywords used for question type classification.
# Each category has patterns that indicate the query intent.
QUESTION_TYPE_KEYWORDS = {
    "comparative": [
        "compare",
        "comparison",
        "versus",
        "vs",
        "differ",
        "difference",
        "similar",
        "contrast",
        "both",
        "each",
    ],
    "factual": [
        "what is",
        "what's",
        "who is",
        "who's",
        "when did",
        "where did",
        "how many",
        "how much",
        "define",
        "explain what",
    ],
    "analytical": [
        "why",
        "how does",
        "what causes",
        "analyze",
        "analysis",
        "implications",
        "impact",
        "effect",
        "significance",
    ],
}


class QueryParser:
    """Parses natural language queries to extract structured filters for vector search."""

    def __init__(self):
        """Initialize QueryParser."""
        pass

    def extract_episode_number(self, query: str) -> Optional[int]:
        """
        Extract episode number from a natural language query.

        Args:
            query: User's natural language question.

        Returns:
            Extracted episode number if found, None otherwise.
        """
        patterns = [
            r"episode\s+number\s+(\d+)",  # "episode number 180"
            r"episode\s+(\d+)",  # "episode 180"
            r"ep\.\s*(\d+)",  # "ep. 180" or "ep.180"
            r"ep\s+(\d+)",  # "ep 180"
            r"#(\d+)",  # "#180"
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def extract_speaker(self, query: str) -> Optional[str]:
        """
        Extract speaker name from a natural language query.

        Args:
            query: User's natural language question.

        Returns:
            Full speaker name for database filtering, or None if no specific speaker mentioned.
        """
        query_lower = query.lower()

        # First check for special keywords (hosts, guest, etc.)
        # These take precedence over individual speaker names.
        for keyword, value in SPECIAL_SPEAKER_KEYWORDS.items():
            # Use word boundary matching to avoid partial matches
            if re.search(rf"\b{keyword}\b", query_lower):
                return value

        # Then check for individual speaker name mappings
        for name_variant, full_name in SPEAKER_MAPPINGS.items():
            if re.search(rf"\b{name_variant}\b", query_lower):
                return full_name

        print("result", None)
        print("+++++++++++++++++++++++++++++++++++++++")
        return None

    def extract_date_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Extract date range from a natural language query.

        Args:
            query: User's natural language question.

        Returns:
            Tuple of (start_date, end_date) as datetime objects, or None if no temporal reference detected.
        """
        query_lower = query.lower()
        today = datetime.now()

        # Pattern: "between YYYY and YYYY"
        # Example: "between 2023 and 2024" → (2023-01-01, 2024-12-31)
        between_match = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", query_lower)
        if between_match:
            start_year = int(between_match.group(1))
            end_year = int(between_match.group(2))
            return (datetime(start_year, 1, 1), datetime(end_year, 12, 31))
        # Pattern: "in YYYY"
        # Example: "in 2023" → (2023-01-01, 2023-12-31)
        elif in_year_match := re.search(r"\bin\s+(\d{4})\b", query_lower):
            year = int(in_year_match.group(1))
            return (datetime(year, 1, 1), datetime(year, 12, 31))
        # Pattern: "since YYYY"
        # Example: "since 2024" → (2024-01-01, today)
        elif since_match := re.search(r"since\s+(\d{4})", query_lower):
            year = int(since_match.group(1))
            return (datetime(year, 1, 1), today)
        # Pattern: "last N months"
        # Example: "last 3 months" → (3 months ago, today)
        elif last_months_match := re.search(r"last\s+(\d+)\s+months?", query_lower):
            months = int(last_months_match.group(1))
            # Approximate months as 30 days each
            start_date = today - timedelta(days=months * 30)
            return (start_date, today)
        # Pattern: "last N weeks"
        # Example: "last 2 weeks" → (2 weeks ago, today)
        elif last_weeks_match := re.search(r"last\s+(\d+)\s+weeks?", query_lower):
            weeks = int(last_weeks_match.group(1))
            start_date = today - timedelta(weeks=weeks)
            return (start_date, today)
        # Pattern: "this year"
        # Example: "this year" → (current year Jan 1, today)
        elif "this year" in query_lower:
            return (datetime(today.year, 1, 1), today)
        # Pattern: "last year"
        # Example: "last year" → (last year Jan 1, last year Dec 31)
        elif "last year" in query_lower:
            last_year = today.year - 1
            return (datetime(last_year, 1, 1), datetime(last_year, 12, 31))

        return None

    def classify_question_type(self, query: str) -> str:
        """
        Classify the question type based on extracted filters and keywords.

        Args:
            query: User's natural language question.

        Returns:
            Question type: "episode_scoped", "speaker_specific", "temporal",
            "comparative", "factual", "analytical", or "topical".
        """
        query_lower = query.lower()

        # Priority 1: Check for episode-scoped queries
        # These are the most constrained and should be handled first.
        if self.extract_episode_number(query) is not None:
            return "episode_scoped"

        # Priority 2: Check for speaker-specific queries
        # Note: We check for actual speaker names, not "hosts" which implies both.
        speaker = self.extract_speaker(query)
        if speaker is not None and speaker not in ["guest", "professor"]:
            return "speaker_specific"

        # Priority 3: Check for temporal queries
        if self.extract_date_range(query) is not None:
            return "temporal"

        # Priority 4: Keyword-based classification
        # Check for comparative, factual, or analytical patterns.
        for question_type, keywords in QUESTION_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return question_type

        # Default: topical query (general topic exploration)
        return "topical"

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract all filters from a query in a single call.

        Args:
            query: User's natural language question.

        Returns:
            Dictionary with 'episode_number', 'speaker', and 'date_range' keys.
        """
        return {
            "episode_number": self.extract_episode_number(query),
            "speaker": self.extract_speaker(query),
            "date_range": self.extract_date_range(query),
        }
