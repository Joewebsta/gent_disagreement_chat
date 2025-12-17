import os
from typing import Any, Dict

from openai import OpenAI

from .query_parser import QueryParser
from .vector_search import VectorSearch


class RAGService:
    """
    Handles RAG (Retrieval-Augmented Generation) operations for the podcast chat.

    This service orchestrates the retrieval pipeline:
    1. Parse user query to extract filters (episode, speaker, date)
    2. Search vector database with filters applied
    3. Format results into context for the LLM
    4. Stream LLM response to the user

    Data flow:
        User Question → QueryParser → VectorSearch → Context Formatting → LLM → Response

    Example usage:
        >>> service = RAGService()
        >>> for chunk in service.ask_question_text_stream("What did Ricky say about tariffs?"):
        ...     print(chunk, end="")
    """

    # Default search parameters for queries without specific constraints.
    # These values balance coverage (enough context) with precision (relevant results).
    DEFAULT_THRESHOLD = 0.6  # Minimum similarity score for inclusion
    DEFAULT_MIN_DOCS = 3  # Minimum documents to return (triggers fallback if not met)
    DEFAULT_MAX_DOCS = 10  # Maximum documents to return

    # Adaptive parameters by question type.
    # Different query types benefit from different retrieval strategies.
    # Format: {question_type: {'min_docs': int, 'max_docs': int, 'threshold': float}}
    ADAPTIVE_PARAMS = {
        # Episode-scoped: Need more docs to cover the full episode context
        "episode_scoped": {"min_docs": 5, "max_docs": 15, "threshold": 0.2},
        # Speaker-specific: Moderate coverage for speaker's views
        "speaker_specific": {"min_docs": 3, "max_docs": 10, "threshold": 0.6},
        # Temporal: Search across time period, need decent coverage
        "temporal": {"min_docs": 3, "max_docs": 10, "threshold": 0.55},
        # Comparative: Need diverse results to compare viewpoints
        "comparative": {"min_docs": 5, "max_docs": 12, "threshold": 0.55},
        # Factual: Fewer, more precise results
        "factual": {"min_docs": 2, "max_docs": 5, "threshold": 0.65},
        # Analytical: More context for deep analysis
        "analytical": {"min_docs": 4, "max_docs": 12, "threshold": 0.55},
        # Topical: Default balanced approach
        "topical": {"min_docs": 3, "max_docs": 10, "threshold": 0.6},
    }

    def __init__(self, database_name: str = "gent_disagreement"):
        """
        Initialize RAGService with database connection and query parser.

        Args:
            database_name: PostgreSQL database name containing podcast transcripts.
                           Default: "gent_disagreement"
        """
        self.vector_search = VectorSearch(database_name)
        self.query_parser = QueryParser()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ask_question_text_stream(
        self, question: str, model: str = "gpt-4o-mini-2024-07-18"
    ):
        """
        Process a user question through the RAG pipeline with text streaming.

        This method orchestrates the full retrieval-augmented generation flow:
        1. Parse the question to extract metadata filters (episode, speaker, date)
        2. Classify the question type for adaptive retrieval parameters
        3. Search the vector database with filters applied
        4. Format results into context for the LLM
        5. Stream the LLM response

        Args:
            question: User's natural language question.
                      Example: "What did Ricky say about tariffs in episode 180?"

            model: OpenAI model identifier for response generation.
                   Default: "gpt-4o-mini-2024-07-18"

        Returns:
            Generator[str]: Yields text chunks as the LLM generates them.
        """
        try:
            # 1. Parse query to extract metadata filters.
            # This enables episode-scoped, speaker-specific, and temporal filtering.
            filters = self.query_parser.extract_filters(question)

            # 2. Classify question type for adaptive retrieval parameters.
            # Different question types benefit from different retrieval strategies.
            question_type = self.query_parser.classify_question_type(question)

            # 3. Get adaptive parameters based on question type.
            # This adjusts min_docs, max_docs, and threshold for optimal retrieval.
            params = self._get_adaptive_params(question_type, filters)

            # 4. Find relevant transcript segments with filters applied.
            # Filters narrow down results to specific episodes/speakers/dates.
            search_results = self.vector_search.find_relevant_with_filters(
                question,
                filters=filters,
                min_docs=params["min_docs"],
                max_docs=params["max_docs"],
                similarity_threshold=params["threshold"],
            )

            # Check if no segments were found
            if len(search_results) == 0:
                return self._generate_no_information_response()

            # 5. Format context from search results into structured text.
            # Groups by episode and includes metadata for LLM context.
            formatted_results = self._format_search_results(search_results)

            # 6. Create prompt with context for the LLM.
            prompt = self._create_prompt(formatted_results, question)

            # 7. Generate and stream LLM response.
            return self._generate_simple_text_stream(prompt, model)

        except Exception as e:
            print(f"Error in RAG service: {e}")
            raise e

    def _get_adaptive_params(self, question_type: str) -> Dict[str, Any]:
        """
        Get retrieval parameters adapted to the question type and filters.

        Different question types benefit from different retrieval strategies:
        - Episode-scoped queries need more documents to cover the full episode
        - Factual queries need fewer, more precise results
        - Analytical queries need more context for deep analysis

        Args:
            question_type: Classification from QueryParser.classify_question_type().
                           One of: 'episode_scoped', 'speaker_specific', 'temporal',
                           'comparative', 'factual', 'analytical', 'topical'

        Returns:
            dict: Retrieval parameters with structure:
                  {
                      'min_docs': int,   # Minimum documents to return
                      'max_docs': int,   # Maximum documents to return
                      'threshold': float  # Similarity threshold (0.0 to 1.0)
                  }
        """
        # Look up parameters for the question type, falling back to defaults.
        params = self.ADAPTIVE_PARAMS.get(
            question_type,
            {
                "min_docs": self.DEFAULT_MIN_DOCS,
                "max_docs": self.DEFAULT_MAX_DOCS,
                "threshold": self.DEFAULT_THRESHOLD,
            },
        )

        # Return a copy to avoid mutating the class constant
        return dict(params)

    def _generate_simple_text_stream(self, prompt, model):
        """Generate simple text stream compatible with AI SDK TextStreamChatTransport"""
        try:
            # Create OpenAI stream
            stream = self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    # Simply yield the text content directly
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error in simple text streaming: {e}")
            yield f"Error: {str(e)}"

    def _generate_no_information_response(self):
        """
        Generate a response when no transcript segments are found.

        This method is called when the vector search returns 0 segments,
        preventing the LLM from hallucinating answers when no context exists.

        Returns:
            Generator[str]: Yields formatted Markdown message chunks
        """
        message = """I couldn't find any relevant information to answer your question.

Try rephrasing your question, or ask about a broader topic from the podcast.
"""
        # Yield the complete message (streaming response will handle chunking)
        yield message

    def _group_by_episode(self, search_results):
        """Group search results by episode with metadata"""
        episode_groups = {}

        for result in search_results:
            episode_key = result["episode_number"]

            if episode_key not in episode_groups:
                episode_groups[episode_key] = {
                    "episode": result["episode_number"],
                    "title": result["title"],
                    "date_published": result["date_published"],
                    "segments": [],
                    "similarities": [],
                }

            episode_groups[episode_key]["segments"].append(result)
            episode_groups[episode_key]["similarities"].append(result["similarity"])

        # Calculate average similarity for each episode and sort segments
        for group in episode_groups.values():
            group["avg_similarity"] = sum(group["similarities"]) / len(
                group["similarities"]
            )
            group["segments"].sort(key=lambda x: x["similarity"], reverse=True)

        # Convert to list and sort by average similarity
        grouped_results = list(episode_groups.values())
        grouped_results.sort(key=lambda x: x["avg_similarity"], reverse=True)

        return grouped_results

    def _format_search_results(self, search_results):
        """Enhanced context formatting with hierarchical organization"""

        grouped_results = self._group_by_episode(search_results)

        formatted_result = ""
        for episode_group in grouped_results:
            formatted_result += (
                f"## Episode {episode_group['episode']}: {episode_group['title']}\n"
            )
            formatted_result += f"**Relevance**: {episode_group['avg_similarity']:.2f} (Published: {episode_group['date_published']})\n\n"

            for result in episode_group["segments"]:
                formatted_result += f"**{result['name']}**: {result['text']}\n"
                formatted_result += f"*Similarity: {result['similarity']:.2f}*\n\n"

        return formatted_result

    def _create_prompt(self, formatted_results, question):
        """Create the prompt for the LLM."""
        return f"""# A Gentleman's Disagreement Podcast Analysis

You are an expert analyst of **A Gentleman's Disagreement Podcast**. Your task is to provide insightful answers based on the provided transcript segments.

## Instructions
- Use the relevant transcript segments below to answer the user's question
- If the segments aren't relevant to the question, clearly state this
- Maintain the conversational tone of the podcast in your analysis
- **Format your response in clean, well-structured Markdown**
- Use proper headings (## or ###), and paragraph breaks for readability
- Bold important points and use quotes for direct transcript references
- **When referencing episodes, ALWAYS use the complete episode name format: "Episode {{number}}: {{full title}}" as shown in the transcript segments. Never use partial names like "Episode 182: Summary" - always include the complete title. For example, use "Episode 180: A SCOTUS '24-'25 Term Review with Professor Jack Beermann" not "Episode 180: Summary"**
- Always refer to Ricky Ghoshroy as "Ricky" and Brendan Kelly as "Brendan". Do refer to them using solely their last names e.g. Ghoshroy or Kelly.

## Available Transcript Segments
{formatted_results}

## User Question
**{question}**

## Your Response
Please provide a comprehensive answer in Markdown format based on the transcript segments and your knowledge of the podcast:"""
