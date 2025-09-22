from .vector_search import VectorSearch
from openai import OpenAI
import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


class RAGService:
    """Handles RAG operations"""

    # Default search parameters
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_LIMIT = 5

    def __init__(self, database_name="gent_disagreement"):
        self.vector_search = VectorSearch(database_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ask_question(self, question, model="gpt-4o-mini"):
        """Implement RAG to answer questions using retrieved context"""
        try:
            # 1. Find relevant transcript segments
            # search_results = self.vector_search.find_similar_above_threshold(
            #     question, threshold=self.DEFAULT_THRESHOLD, limit=self.DEFAULT_LIMIT
            # )

            search_results = self.vector_search.find_most_similar(
                question, limit=self.DEFAULT_LIMIT
            )

            # 2. Format context from search results
            formatted_results = self._format_search_results(search_results)

            # 3. Create prompt with context
            prompt = self._create_prompt(formatted_results, question)

            # 4. Generate response
            return self._generate_streaming_response(prompt, model)

        except Exception as e:
            print(f"Error in RAG service: {e}")
            raise e

    def ask_question_ai_sdk_format(self, question, model="gpt-4o-mini"):
        """Implement RAG with AI SDK compatible streaming format"""
        try:
            # 1. Find relevant transcript segments
            search_results = self.vector_search.find_most_similar(
                question, limit=self.DEFAULT_LIMIT
            )

            # 2. Format context from search results
            formatted_results = self._format_search_results(search_results)

            # 3. Create prompt with context
            prompt = self._create_prompt(formatted_results, question)

            # 4. Generate response in AI SDK format
            return self._generate_ai_sdk_streaming_response(prompt, model)

        except Exception as e:
            print(f"Error in RAG service: {e}")
            raise e

    def ask_question_text_stream(self, question, model="gpt-4o-mini-2024-07-18"):
        """Implement RAG with simple text streaming for AI SDK compatibility"""
        try:
            # 1. Find relevant transcript segments
            search_results = self.vector_search.find_most_similar(
                question, limit=self.DEFAULT_LIMIT
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

    def _generate_streaming_response(self, prompt, model):
        """Generate streaming response using OpenAI LLM"""
        try:
            stream = self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Prefix each line with 'data: ' per SSE spec, preserving empty lines
                    for line in content.split("\n"):
                        yield f"data: {line}\n"
                    # End of event
                    yield "\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Error in streaming response: {e}")
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    def _generate_ai_sdk_streaming_response(self, prompt, model):
        """Generate AI SDK compatible streaming response"""
        import uuid

        try:
            # Generate a unique message ID for this response
            message_id = str(uuid.uuid4())

            # Send the initial message metadata
            initial_chunk = {"id": message_id, "role": "assistant", "parts": []}
            yield f"0:{json.dumps(initial_chunk)}\n"

            # Create OpenAI stream
            stream = self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], stream=True
            )

            text_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    text_content += content

                    # Send text delta in AI SDK format
                    text_chunk = {"type": "text-delta", "textDelta": content}
                    yield f"0:{json.dumps(text_chunk)}\n"

            # Send final message with complete content
            final_chunk = {
                "id": message_id,
                "role": "assistant",
                "parts": [{"type": "text", "text": text_content}],
            }
            yield f"0:{json.dumps(final_chunk)}\n"

            # Send done marker
            yield f"d\n"

        except Exception as e:
            print(f"Error in AI SDK streaming response: {e}")
            # Send error in AI SDK format
            error_chunk = {"type": "error", "error": str(e)}
            yield f"3:{json.dumps(error_chunk)}\n"

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
