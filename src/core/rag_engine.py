import logging
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.groq import Groq
from pinecone import Pinecone

# Import kedua jenis mesin embedding
from src.core.google_embedding import GeminiNewEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config.settings import Settings as AppSettings
from src.config.prompts import QA_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class RAGEngine:    
    def __init__(self):
        self.chat_engine = None
        self._validate_api_keys()
        self._initialize()
    
    def _validate_api_keys(self):
        pinecone_key = AppSettings.get_pinecone_api_key()
        groq_key = AppSettings.get_groq_api_key()
        google_key = AppSettings.get_google_api_key()
        
        # Jika menggunakan Google, wajib ada API Key Google
        if AppSettings.EMBEDDING_PROVIDER == "google" and not google_key:
            raise ValueError("Missing Google API key for Google Embeddings.")
            
        if not all([pinecone_key, groq_key]):
            raise ValueError("Missing required API keys (Pinecone atau Groq)...")
    
    def _initialize(self):
        logger.info("Initializing RAG engine...")

        # --- LOGIKA SAKLAR EMBEDDING ---
        if AppSettings.EMBEDDING_PROVIDER == "local":
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=AppSettings.LOCAL_EMBEDDING_MODEL
            )
            logger.info(f"Embedding model configured: LOCAL ({AppSettings.LOCAL_EMBEDDING_MODEL})")
        else:
            Settings.embed_model = GeminiNewEmbedding(
                api_key=AppSettings.get_google_api_key(),
                model_name=AppSettings.GOOGLE_EMBEDDING_MODEL
            )
            logger.info(f"Embedding model configured: GOOGLE ({AppSettings.GOOGLE_EMBEDDING_MODEL})")
        
        # LLM (Tetap menggunakan Llama-3 via Groq)
        Settings.llm = Groq(
            model=AppSettings.LLM_MODEL,
            api_key=AppSettings.get_groq_api_key(),
            temperature=AppSettings.LLM_TEMPERATURE
        )
        logger.info(f"LLM configured: {AppSettings.LLM_MODEL}")
        
        # Connect to Pinecone
        pc = Pinecone(api_key=AppSettings.get_pinecone_api_key())
        vector_store = PineconeVectorStore(
            pinecone_index=pc.Index(AppSettings.INDEX_NAME)
        )
        logger.info(f"Connected to Pinecone index: {AppSettings.INDEX_NAME}")
        
        # Load index from vector store
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # Create chat engine with SYSTEM PROMPT (mengganti text_qa_template)
        self.chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=QA_PROMPT_TEMPLATE,
            similarity_top_k=AppSettings.SIMILARITY_TOP_K
        )
        
        logger.info("RAG engine initialized successfully")
    
    def get_engine(self):
        return self.chat_engine
    
    def reset_memory(self):
        """Reset chat engine memory to free up context window"""
        if self.chat_engine and hasattr(self.chat_engine, 'reset'):
            self.chat_engine.reset()