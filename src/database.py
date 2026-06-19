import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY

def build_or_load_vector_db():
    """Reads support docs inside the target directory and converts them into an indexed embedding store."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        raise FileNotFoundError(f"Created '{DATA_DIR}' directory. Please drop your 10-20 support articles there.")
    
    all_documents = []
    
    # Track file processing loops
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file_path, encoding="utf-8")
            all_documents.extend(loader.load())
            
    if not all_documents:
        raise ValueError(f"No documents found inside {DATA_DIR}. Drop your required reference assets here.")
        
    # Split documents cleanly
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_chunks = splitter.split_documents(all_documents)
    
    # Standardize metadata naming convention
    for chunk in split_chunks:
        if "source" in chunk.metadata:
            chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(split_chunks, embeddings)
    return vector_store
