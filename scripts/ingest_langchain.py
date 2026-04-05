import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Ekosistem LangChain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Config Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = "ordal-filkom"

def main():
    if not GOOGLE_API_KEY or not PINECONE_API_KEY:
        logger.error("API Keys missing in .env")
        return

    logger.info("Mencari dokumen PDF...")
    pdf_files = [str(p) for p in Path("./dataset").rglob("*.pdf")]
    logger.info(f"Ditemukan {len(pdf_files)} file PDF.")

    # 1. Ekstraksi Teks dari PDF
    documents = []
    for pdf_path in tqdm(pdf_files, desc="Membaca PDF"):
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Gagal membaca {pdf_path}: {e}")

    logger.info(f"Berhasil mengekstrak {len(documents)} halaman dokumen.")

    # 2. Proses Chunking (Memecah teks menjadi potongan kecil)
    logger.info("Memecah teks (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Total chunks yang dihasilkan: {len(chunks)}")

    # 3. Setup Model Embedding Google
    logger.info("Inisialisasi Model Embedding...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", # Menggunakan model paling stabil
        google_api_key=GOOGLE_API_KEY
    )

    # 4. Setup Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        logger.info(f"Menghapus index lama '{INDEX_NAME}'...")
        pc.delete_index(INDEX_NAME)
        time.sleep(10) # Jeda untuk memastikan index benar-benar terhapus dari server Pinecone

    # SELALU buat index baru (karena jika sebelumnya ada, pasti sudah dihapus di atas)
    logger.info(f"Membuat index baru '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768, # Dimensi standar untuk embedding-001
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    time.sleep(10) # Jeda agar index siap menerima data

    # 5. Insert ke Pinecone via LangChain
    logger.info("Memasukkan data ke Pinecone (Batching otomatis oleh LangChain)...")
    try:
        # LangChain akan otomatis memotong chunks menjadi batch kecil dan mengunggahnya
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        logger.info("SUCCESS: Semua dokumen berhasil masuk ke Pinecone!")
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat memasukkan data: {e}")

if __name__ == "__main__":
    main()