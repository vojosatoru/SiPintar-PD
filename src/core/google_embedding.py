from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding
from google import genai
from google.genai import types

class GeminiNewEmbedding(BaseEmbedding):
    client: Any = None
    model_name: str = "gemini-embedding-001"
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gemini-embedding-001", 
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Inisialisasi client API Google versi terbaru
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _get_query_embedding(self, query: str) -> List[float]:
        # Khusus untuk memproses pertanyaan user saat chatting
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return result.embeddings[0].values

    def _get_text_embedding(self, text: str) -> List[float]:
        # Khusus untuk menyimpan teks tunggal ke database
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return result.embeddings[0].values

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Khusus untuk memproses ratusan batch sekaligus di ingest.py
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return [e.values for e in result.embeddings]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)