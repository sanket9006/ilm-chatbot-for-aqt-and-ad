from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rag_pipeline import RAGPipeline
import os

app = FastAPI(title="E-commerce RAG System")

# Initialize pipeline on startup
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")
pipeline = RAGPipeline(DATA_PATH)

# Serve static files (if any)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
async def read_root():
    # Serve the index.html from the static directory
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/ask")
async def ask_question(query: str = Query(..., description="The user's question")):
    """
    Endpoint to ask questions about products.
    """
    result = pipeline.ask(query)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
