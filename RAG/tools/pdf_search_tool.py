import os
import re
import numpy as np
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from markitdown import MarkItDown
from chonkie import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    
    def __init__(self, file_path: str, chat_title: str, chat_id: str):
        super().__init__()
        self.file_path = file_path
        self.chat_title = chat_title
        self.chat_id = chat_id
        self.collection_name = self._generate_collection_name()
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF file not found at {self.file_path}")
        
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Process document only if collection doesn't exist
        if self.collection_name not in [col.name for col in self.client.get_collections().collections]:
            self._process_document()

    def _generate_collection_name(self):
        """Generate a unique collection name based on chat_title and chat_id."""
        sanitized_title = re.sub(r'[^a-zA-Z0-9_]', '_', self.chat_title)[:50]
        return f"{sanitized_title}_{self.chat_id}"

    def _process_document(self):
        try:
            print(f"Processing PDF for collection {self.collection_name}")
            # Delete existing collection if present (optional, based on your needs)
            if self.collection_name in [col.name for col in self.client.get_collections().collections]:
                self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            raw_text = self._extract_text()
            chunks = self._create_chunks(raw_text)
            embeddings = [self._generate_embeddings(chunk.text) for chunk in chunks]
            points = [
                {
                    "id": idx, 
                    "vector": embedding.tolist(),
                    "payload": {
                        "text": chunk.text,
                        "source": os.path.basename(self.file_path)
                    }
                }
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                if embedding.size > 0
            ]
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Inserted {len(points)} chunks into {self.collection_name}")
            else:
                print("No valid points to insert")
        except Exception as e:
            print(f"Document processing failed: {str(e)}")
            raise

    def _generate_embeddings(self, text: str) -> np.ndarray:
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.array([])

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        try:
            md = MarkItDown()
            result = md.convert(self.file_path)
            return result.text_content
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def _create_chunks(self, raw_text: str) -> list:
        try:
            chunker = SemanticChunker(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                threshold=0.5,
                chunk_size=512,
                min_sentences=1
            )
            return chunker.chunk(raw_text)
        except Exception as e:
            print(f"Error creating chunks: {e}")
            return []

    def _run(self, query: str) -> str:
        try:
            query_embedding = self._generate_embeddings(query)
            if query_embedding.size == 0:
                return "Failed to process query"
            if self.collection_name not in [col.name for col in self.client.get_collections().collections]:
                return "Document index not found"
            relevant_chunks = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=5
            )
            docs = [chunk.payload["text"] for chunk in relevant_chunks]
            return "\n___\n".join(docs) if docs else "No relevant information found."
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return f"Search error: {str(e)}"