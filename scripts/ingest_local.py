import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import nest_asyncio
import time
from tqdm import tqdm

# Apply nest_asyncio for LlamaParse async operations
nest_asyncio.apply()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, get_leaf_nodes
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec
from src.utils.metadata import get_meta

# Config Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = "sipintar-pd"

def init_local_settings():
    """Menggunakan model HuggingFace lokal, tanpa LLM API"""
    logger.info("Memuat model embedding lokal (multilingual-e5-base)...")
    
    # Model ini akan diunduh ke cache komputer pada percobaan pertama (sekitar 1-2 GB)
    # Menghasilkan 768 dimensi, cocok dengan Pinecone Anda
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base"
    )
    
    # Kita nonaktifkan LLM karena proses chunking hirarkis tidak membutuhkan LLM
    # Ini mencegah error koneksi ke Google API
    Settings.llm = None

def main():
    if not PINECONE_API_KEY or not LLAMA_CLOUD_API_KEY:
        logger.error("API Keys (Pinecone / LlamaCloud) missing in .env")
        return

    init_local_settings()

    logger.info("Loading Documents with LlamaParse...")
    
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True,
        language="id",
        num_workers=4, 
        parsing_instruction="""
        This is an Indonesian bureaucratic and public policy document (RPJPD, RPJMD, RENSTRA, RKPD).
        Please:
        - Extract all text content with proper structure
        - Pay extremely close attention to budget tables, performance indicators (Indikator Kinerja), and targets. Convert all tables to markdown format accurately.
        - Preserve document hierarchy (Bab, Pasal, Bagian)
        """
    )
    
    pdf_files = [str(p) for p in Path("./dataset").rglob("*.pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    documents = []
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            parsed_docs = parser.load_data(pdf_file)
            file_metadata = get_meta(pdf_file)
            
            for page_idx, doc in enumerate(parsed_docs):
                doc.metadata.update(file_metadata)
                doc.metadata['page_label'] = str(page_idx + 1)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Failed to parse {pdf_file}: {e}")
            continue
    
    logger.info(f"Parsed {len(documents)} document pages.")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Cek dan Reset Index
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME in existing_indexes:
        logger.info(f"Deleting existing index {INDEX_NAME} to reset...")
        pc.delete_index(INDEX_NAME)
        time.sleep(10) # Beri waktu Pinecone untuk menghapus index
        
    logger.info(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768, # Konsisten dengan model e5-base
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    
    vector_store = PineconeVectorStore(pinecone_index=pc.Index(INDEX_NAME))
    
    # CHUNKING LOKAL
    logger.info("Initializing Standard chunking...")
    # Gunakan ukuran chunk besar (1024) dengan irisan (200) 
    # agar tabel dan judul tidak pernah terputus!
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    
    # Ubah leaf_nodes menjadi nodes
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    logger.info(f"Final chunks: {len(nodes)}")
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    BATCH_SIZE = 50  # Bisa lebih besar karena proses lokal tidak kena limit API (kecuali limit Pinecone)
    DELAY_SECONDS = 1 

    logger.info(f"Indexing in batches of {BATCH_SIZE}...")
    index = None

    for i in tqdm(range(0, len(nodes), BATCH_SIZE), desc="Uploading to Pinecone"):
        batch_nodes = nodes[i : i + BATCH_SIZE]
        try:
            if index is None:
                index = VectorStoreIndex(batch_nodes, storage_context=storage_context)
            else:
                index.insert_nodes(batch_nodes)
            
            time.sleep(DELAY_SECONDS)
        except Exception as e:
            logger.error(f"Error indexing batch starting at {i}: {e}")
            time.sleep(5)
    
    logger.info("SUCCESS: Documents inserted to Pinecone using LOCAL EMBEDDINGS.")

if __name__ == "__main__":
    main()