import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from .database_manager import DatabaseManager


class EmbeddingService:
    """Handles embedding generation and storage for transcript segments."""
    
    def __init__(self, database_manager=None):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.database_manager = database_manager or DatabaseManager()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding for the given text."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    
    def generate_and_store_embeddings(self, segments: List[Dict[str, Any]]) -> None:
        """Generate embeddings for all segments and store them in the database."""
        try:
            for segment in segments:
                embedding = self.generate_embedding(segment["text"])
                self.database_manager.insert_transcript_segment_with_embedding(
                    segment["speaker"], segment["text"], embedding
                )
            print("All embeddings stored successfully!")
        except Exception as e:
            print("Error generating embeddings:", e)
            raise