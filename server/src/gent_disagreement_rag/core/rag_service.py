from .vector_search import VectorSearch
from .query_enhancer import QueryEnhancer
from openai import OpenAI
import os
import json


class RAGService:
    """Handles RAG operations"""

    # Default search parameters
    DEFAULT_THRESHOLD = 0.6
    DEFAULT_MIN_DOCS = 3
    DEFAULT_MAX_DOCS = 10

    def __init__(self, database_name="gent_disagreement", use_query_enhancement=True):
        self.vector_search = VectorSearch(database_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.use_query_enhancement = use_query_enhancement

        if self.use_query_enhancement:
            self.query_enhancer = QueryEnhancer()

    def ask_question_text_stream(self, question, model="gpt-4o-mini-2024-07-18"):
        """Implement RAG with simple text streaming for AI SDK compatibility"""
        try:
            # 1. Find relevant transcript segments with optional query enhancement
            if self.use_query_enhancement:
                enhanced_query = self.query_enhancer.enhance_query(question)
                search_results = self._multi_query_search(enhanced_query)
            else:
                search_results = (
                    self.vector_search.find_relevant_above_adaptive_threshold(
                        question,
                        min_docs=self.DEFAULT_MIN_DOCS,
                        max_docs=self.DEFAULT_MAX_DOCS,
                        similarity_threshold=self.DEFAULT_THRESHOLD,
                    )
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

    def _multi_query_search(self, enhanced_query):
        """Perform multi-query search using enhanced query variations"""
        all_results = []
        seen_texts = set()  # To avoid duplicates

        # Search with original query
        original_results = self.vector_search.find_relevant_above_adaptive_threshold(
            enhanced_query["original"],
            min_docs=self.DEFAULT_MIN_DOCS,
            max_docs=self.DEFAULT_MAX_DOCS,
            similarity_threshold=self.DEFAULT_THRESHOLD,
        )
        for result in original_results:
            if result["text"] not in seen_texts:
                all_results.append(result)
                seen_texts.add(result["text"])

        # Search with HyDE hypothetical answer if different from original
        if enhanced_query["hyde"] != enhanced_query["original"]:
            hyde_results = self.vector_search.find_relevant_above_adaptive_threshold(
                enhanced_query["hyde"],
                min_docs=2,
                max_docs=5,
                similarity_threshold=self.DEFAULT_THRESHOLD,
            )
            for result in hyde_results:
                if result["text"] not in seen_texts:
                    all_results.append(result)
                    seen_texts.add(result["text"])

        # Search with expanded query if significantly different
        if (
            len(enhanced_query["expanded"].split())
            > len(enhanced_query["original"].split()) * 1.5
        ):
            expanded_results = (
                self.vector_search.find_relevant_above_adaptive_threshold(
                    enhanced_query["expanded"],
                    min_docs=2,
                    max_docs=5,
                    similarity_threshold=self.DEFAULT_THRESHOLD,
                )
            )
            for result in expanded_results:
                if result["text"] not in seen_texts:
                    all_results.append(result)
                    seen_texts.add(result["text"])

        # Sort by similarity and limit results
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[: self.DEFAULT_MAX_DOCS]

    def _format_search_results(self, search_results):
        """Format the search results into readable string"""

        formatted_result = ""
        for result in search_results:
            formatted_result += f"Speaker: {result['speaker']}\n"
            formatted_result += f"Text: {result['text']}\n"
            formatted_result += f"Similarity: {result['similarity']}\n  "
            formatted_result += f"Episode: {result['episode_number']}\n"
            formatted_result += f"Title: {result['title']}\n"
            formatted_result += f"Date Published: {result['date_published']}\n"
            formatted_result += f"--------------------------------\n"

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
