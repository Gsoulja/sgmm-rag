# app/main.py
import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from .pdf_processor import PDFProcessor
from .embeddings import DeepSeekEmbedder
from .vector_store import FAISSVectorStore
from .utils import get_pdf_files, extract_metadata_from_filename, timing_decorator

# Constants
DATA_DIR = os.environ.get("DATA_DIR", "data/pdfs")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
EMBEDDING_DIM = 4096  # DeepSeek-coder-r1-8b embedding dimension

# Initialize FastAPI
app = FastAPI(
    title="Book Search RAG API",
    description="A RAG-based system for searching through books in PDF format",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
embedder = None  # Initialize later to save memory during startup
vector_store = None  # Initialize later


# Pydantic models
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str
    source: str
    chunk_id: int
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    processing_time_ms: float


# Initialization
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vector_store

    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Try to load existing vector store
    index_path = os.path.join(MODELS_DIR, "faiss_index.index")
    if os.path.exists(index_path):
        try:
            print("Loading existing FAISS index...")
            vector_store = FAISSVectorStore.load(MODELS_DIR)
            print(f"Loaded index with {len(vector_store.documents)} documents")
        except Exception as e:
            print(f"Error loading index: {e}")
            vector_store = FAISSVectorStore(dimension=EMBEDDING_DIM)
    else:
        print("Creating new FAISS index")
        vector_store = FAISSVectorStore(dimension=EMBEDDING_DIM)


# Lazy load embedder to save memory
def get_embedder():
    """Get or initialize the embedder"""
    global embedder
    if embedder is None:
        embedder = DeepSeekEmbedder()
    return embedder


# API endpoints
@app.post("/upload/", response_class=JSONResponse)
async def upload_pdf(file: UploadFile = File(...),
                     background_tasks: BackgroundTasks = None):
    """Upload a PDF file and add it to the index"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Save the file
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Process in background if background_tasks provided
    if background_tasks:
        background_tasks.add_task(process_pdf_file, file_path)
        return {"message": f"PDF uploaded successfully. Processing in background."}
    else:
        # Process immediately
        result = process_pdf_file(file_path)
        return result


@app.post("/process/", response_class=JSONResponse)
async def process_directory():
    """Process all PDFs in the data directory"""
    pdf_files = get_pdf_files(DATA_DIR)

    if not pdf_files:
        return {"message": "No PDF files found in the data directory"}

    results = []
    for pdf_file in pdf_files:
        try:
            result = process_pdf_file(pdf_file)
            results.append(result)
        except Exception as e:
            results.append({"file": pdf_file, "error": str(e)})

    # Save the index
    vector_store.save(MODELS_DIR)

    return {
        "message": f"Processed {len(results)} PDF files",
        "results": results
    }


@app.post("/search/", response_model=SearchResponse)
@timing_decorator
async def search(query: SearchQuery):
    """Search for documents matching the query"""
    if vector_store is None or len(vector_store.documents) == 0:
        raise HTTPException(status_code=404, detail="No documents in the index")

    start_time = time.time()

    # Get embedder
    embedder = get_embedder()

    # Generate embedding for the query
    query_embedding = embedder.generate_embedding(query.query)

    # Search the vector store
    distances, docs = vector_store.search(query_embedding, k=query.top_k)

    # Prepare results
    results = []
    for i, (doc, distance) in enumerate(zip(docs, distances)):
        # Convert distance to score (lower distance = higher score)
        score = 1.0 / (1.0 + distance)

        result = SearchResult(
            text=doc["text"],
            source=doc.get("source", "Unknown"),
            chunk_id=doc.get("chunk_id", -1),
            score=score,
            metadata={k: v for k, v in doc.items() if k not in ["text", "chunk_id", "source"]}
        )
        results.append(result)

    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000

    return SearchResponse(
        results=results,
        query=query.query,
        total_results=len(results),
        processing_time_ms=processing_time_ms
    )


@app.get("/status/")
async def status():
    """Get the status of the index"""
    if vector_store is None:
        return {"status": "not_initialized"}

    return {
        "status": "ready",
        "documents_count": len(vector_store.documents),
        "unique_sources": len(set(doc.get("source", "") for doc in vector_store.documents))
    }


@app.delete("/reset/")
async def reset_index():
    """Reset the index and delete all documents"""
    global vector_store

    # Create a new index
    vector_store = FAISSVectorStore(dimension=EMBEDDING_DIM)

    # Delete index files
    index_path = os.path.join(MODELS_DIR, "faiss_index.index")
    docs_path = os.path.join(MODELS_DIR, "faiss_index_docs.pkl")

    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(docs_path):
            os.remove(docs_path)
    except Exception as e:
        return {"message": f"Error deleting index files: {str(e)}"}

    return {"message": "Index reset successfully"}


# Helper functions
def process_pdf_file(pdf_path: str) -> Dict[str, Any]:
    """Process a PDF file and add it to the index"""
    filename = os.path.basename(pdf_path)

    # Extract metadata
    metadata = extract_metadata_from_filename(filename)

    # Process the PDF
    documents = pdf_processor.process_pdf(pdf_path, metadata)

    if not documents:
        return {
            "file": filename,
            "status": "error",
            "message": "No text extracted from PDF"
        }

    # Get embedder
    embedder = get_embedder()

    # Generate embeddings
    embeddings = embedder.embed_documents(documents)

    # Add to vector store
    vector_store.add_documents(documents, embeddings)

    # Save the index
    vector_store.save(MODELS_DIR)

    return {
        "file": filename,
        "status": "success",
        "chunks": len(documents)
    }


# Run the app
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
