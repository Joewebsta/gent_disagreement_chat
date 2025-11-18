from .vector_search import VectorSearch
from openai import OpenAI
import os


class RAGService:
    """Handles RAG operations"""

    # Default search parameters
    DEFAULT_THRESHOLD = 0.6
    DEFAULT_MIN_DOCS = 3
    DEFAULT_MAX_DOCS = 10

    def __init__(self, database_name="gent_disagreement"):
        self.vector_search = VectorSearch(database_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ask_question_text_stream(self, question, model="gpt-4o-mini-2024-07-18"):
        """Implement RAG with simple text streaming for AI SDK compatibility"""
        try:
            # 1. Find relevant transcript segments
            search_results = self.vector_search.find_relevant_above_adaptive_threshold(
                question,
                min_docs=self.DEFAULT_MIN_DOCS,
                max_docs=self.DEFAULT_MAX_DOCS,
                similarity_threshold=self.DEFAULT_THRESHOLD,
            )

            # 2. Format context from search results
            formatted_results = self._format_search_results(search_results)

            # 3. Create prompt with context
            prompt = self._create_prompt(formatted_results, question)

            # 4. Generate simple text stream
            return self._generate_simple_text_stream(prompt, model)

        except Exception as e:
            print(f"Error in RAG service: {e}")
            raise e

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

    def group_by_episode(self, search_results):
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

        # Group by episode
        grouped_results = self.group_by_episode(search_results)
        # print("grouped_results", grouped_results)

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

## Available Transcript Segments
{formatted_results}

## User Question
**{question}**

## Your Response
Please provide a comprehensive answer in Markdown format based on the transcript segments and your knowledge of the podcast:"""
