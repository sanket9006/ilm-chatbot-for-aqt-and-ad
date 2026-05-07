from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from rag_pipeline import RAGPipeline
import os

app = FastAPI(title="E-commerce RAG System")

# Initialize pipeline on startup
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")
pipeline = RAGPipeline(DATA_PATH)

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
    result = pipeline.ask(request.query, request.history)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
