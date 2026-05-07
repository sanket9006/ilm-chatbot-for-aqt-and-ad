from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from rag_pipeline import RAGPipeline
import os
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("API_MAIN")

app = FastAPI(title="E-commerce RAG System")

# Initialize pipeline on startup
logger.info("Initializing RAG Pipeline...")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")
pipeline = RAGPipeline(DATA_PATH)
logger.info("RAG Pipeline successfully initialized.")

# Serve static files (if any)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []

@app.get("/")
async def read_root():
    # Serve the index.html from the static directory
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.post("/ask")
async def ask_question(request: ChatRequest):
    """
    Endpoint to ask questions about products with chat history context.
    """
    logger.info(f"Received request: query='{request.query}', history_length={len(request.history)}")
    result = pipeline.ask(request.query, request.history)
    logger.info("Successfully generated response for request.")
    return result

@app.post("/clear-cache")
async def clear_cache():
    """
    Endpoint to clear the RAG pipeline cache.
    """
    logger.info("Clearing RAG Pipeline cache via API request.")
    pipeline.cache.clear()
    return {"status": "success", "message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
