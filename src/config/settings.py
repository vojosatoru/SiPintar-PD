import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


class Settings:    
    # API Keys
    @staticmethod
    def get_google_api_key():
        try:
            return st.secrets["GOOGLE_API_KEY"]
        except:
            return os.getenv("GOOGLE_API_KEY")
    
    @staticmethod
    def get_pinecone_api_key():
        try:
            return st.secrets["PINECONE_API_KEY"]
        except:
            return os.getenv("PINECONE_API_KEY")
    
    @staticmethod
    def get_groq_api_key():
        try:
            return st.secrets["GROQ_API_KEY"]
        except:
            return os.getenv("GROQ_API_KEY")
    
    # --- PENGATURAN FLEKSIBILITAS EMBEDDING ---
    # Ubah "local" atau "google" sesuai dengan proses ingest terakhir yang Anda lakukan
    EMBEDDING_PROVIDER = "local" 
    
    GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"
    LOCAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    # ------------------------------------------
    
    # Vector Store Configuration
    INDEX_NAME = "sipintar-pd"
    
    # LLM Configuration with Fallback
    LLM_MODEL = "llama-3.3-70b-versatile"  # Primary model
    LLM_TEMPERATURE = 0.0
    SIMILARITY_TOP_K = 20
    
    # Fallback models (ordered by priority when primary hits rate limit)
    # Format: (model_name, TPM_limit, description, note)
    FALLBACK_MODELS = [
        ("meta-llama/llama-4-scout-17b-16e-instruct", 30000, "Llama 4 Scout", "mid 🙂"),
        ("llama-3.1-8b-instant", 6000, "Llama 3.1 8B", "agak kocaks 😹"),
    ]
    
    @staticmethod
    def get_all_available_models():
        """
        Get list of all available models (primary + fallbacks)
        
        Returns:
            list: List of dicts with model metadata
                - model: model identifier
                - description: human-readable name
                - tpm: tokens per minute limit
                - note: fun description
        """
        models = [
            {
                "model": Settings.LLM_MODEL,
                "description": "Llama 3.3 70B",
                "tpm": "12,000",
                "note": "paling bagus 🔥"
            }
        ]
        
        # Add fallback models
        for model_name, tpm_limit, description, note in Settings.FALLBACK_MODELS:
            models.append({
                "model": model_name,
                "description": description,
                "tpm": f"{tpm_limit:,}",
                "note": note
            })
        
        return models
    
    # Paths
    DATASET_DIR = "dataset"
    
    # UI Configuration
    PAGE_TITLE = "SiPintar-PD - Bapperida Kudus"
    PAGE_ICON = "🏛️"
    LAYOUT = "centered"
    
    # Chat Configuration
    MAX_RETRIES = 3
    RETRY_WAIT_BASE = 25
    TOP_SOURCES_TO_DISPLAY = 3
    PDF_RENDER_DPI = 120
