import os
import logging
from typing import List, Dict
from embeddings import EmbeddingModel
from vector_store import VectorStore
from data_loader import load_products, prepare_chunks

logger = logging.getLogger("RAG_PIPELINE")

# Use the new google-genai library
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Toggle this flag to True to enable query rewriting (uses 2x API quota)
ENABLE_QUERY_REWRITE = False

class RAGPipeline:
    def __init__(self, data_path: str):
        logger.info("Initializing Embedding Model...")
        self.embedding_model = EmbeddingModel()
        
        # Load and prepare data
        logger.info(f"Loading data from {data_path}...")
        products = load_products(data_path)
        chunks = prepare_chunks(products)
        logger.info(f"Prepared {len(chunks)} chunks from {len(products)} products.")
        
        # Initialize Vector Store
        sample_embedding = self.embedding_model.get_embeddings([chunks[0]["text"]])
        dimension = sample_embedding.shape[1]
        self.vector_store = VectorStore(dimension)
        
        # Add chunks to vector store
        logger.info("Generating embeddings for chunks and adding to Vector Store...")
        embeddings = self.embedding_model.get_embeddings([c["text"] for c in chunks])
        self.vector_store.add_chunks(embeddings, chunks)
        logger.info("Vector Store indexing complete.")
        
        # Simple in-memory cache for queries (Bonus)
        self.cache = {}

    def get_llm_response(self, context: str, query: str, history: List[Dict] = None) -> str:
        """
        Calls Gemini or a Mock LLM using the new google-genai library.
        """
        # Priority: Environment variable -> Hardcoded fallback (not recommended for prod)
        api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyDBdW_kv7K4auMHqnMsejCfqMdi4y4X0_w"
        
        history_str = ""
        if history:
            history_str = "CHAT HISTORY:\n"
            for msg in history:
                history_str += f"{msg['role'].upper()}: {msg['content']}\n"
        
        prompt = f"""SYSTEM:
You are a specialized product assistant for AirDoctor (air purifiers) and AquaTru (water purifiers). 
1. For product questions, answer ONLY using the provided context. If the answer is not in the context, say "I don't know".
2. For general greetings (like 'Hello') or questions about your purpose, respond naturally and politely.
3. For IRRELEVANT questions (e.g., sports, politics, general world knowledge), politely state that you are an expert on AirDoctor and AquaTru systems and are unable to assist with unrelated topics.
4. Use the Chat History to understand context, but base factual answers ONLY on the Context provided below.

{history_str}

USER:
Context:
{context}

Question:
{query}
"""

        if HAS_GEMINI and api_key:
            try:
                # Initialize the new GenAI client
                client = genai.Client(api_key=api_key)
                
                # Use gemini-2.0-flash based on availability list
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    logger.warning("Gemini API Rate Limit Exceeded (429).")
                    return "⚠️ **API Rate Limit Exceeded** ⚠️\n\nYou are using the Free Tier of the Gemini API, which allows 15 requests per minute. Please wait about 30-60 seconds and try your question again."
                
                # Secondary fallback to flash-latest
                try:
                    logger.warning(f"Error with 2.0-flash: {error_msg}. Falling back to 1.5-flash...")
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt
                    )
                    return response.text
                except Exception as inner_e:
                    inner_err = str(inner_e)
                    if "429" in inner_err or "RESOURCE_EXHAUSTED" in inner_err:
                        logger.warning("Gemini API Rate Limit Exceeded (429) on fallback.")
                        return "⚠️ **API Rate Limit Exceeded** ⚠️\n\nPlease wait about 30-60 seconds and try your question again."
                    logger.error(f"Failed to generate LLM response. Error: {inner_err}")
                    return f"Sorry, I encountered an error generating a response. Error: {inner_err}"
        else:
            # Simple Mock LLM for demo purposes
            logger.warning("HAS_GEMINI is false or API key missing. Using mock response.")
            return f"Mock Answer: Based on the context provided, I found relevant information about your query '{query}'. However, a GEMINI_API_KEY was not found to generate a natural response."

    def rewrite_query(self, query: str, history: List[Dict]) -> str:
        """
        Rewrites the query to be standalone, resolving pronouns based on history.
        """
        if not ENABLE_QUERY_REWRITE or not HAS_GEMINI or not history:
            return query
            
        api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyDBdW_kv7K4auMHqnMsejCfqMdi4y4X0_w"
        client = genai.Client(api_key=api_key)
        
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-4:]]) # last 2 turns
        
        prompt = f"""Given the following conversation history and a new user question, rewrite the user question to be a standalone query that can be understood without the history. Do not answer the question, just rewrite it. If it is already standalone, return it as is.
        
History:
{history_str}

New Question: {query}
Standalone Question:"""
        try:
            res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return res.text.strip()
        except:
            return query

    def ask(self, query: str, history: List[Dict] = None) -> Dict:
        history = history or []
        logger.info(f"Processing query: '{query}'")
        
        # 0. Rewrite query if history exists
        standalone_query = self.rewrite_query(query, history)
        if standalone_query != query:
            logger.info(f"Query rewritten to: '{standalone_query}'")
        
        # Cache uses the full context to avoid bad cache hits
        cache_key = f"{standalone_query}_{len(history)}"
        if cache_key in self.cache:
            logger.info("Cache hit! Returning cached response.")
            return self.cache[cache_key]
            
        # 1. Embed query (using standalone)
        logger.debug("Generating query embedding...")
        query_vec = self.embedding_model.get_query_embedding(standalone_query)
        
        # 2. Retrieve top chunks
        logger.info("Searching Vector Store...")
        retrieved_chunks = self.vector_store.search(query_vec, top_k=3)
        
        # 3. Build context
        context = "\n---\n".join([c["text"] for c in retrieved_chunks])
        sources = list(set([c["metadata"]["product_name"] for c in retrieved_chunks]))
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks. Sources: {sources}")
        
        # 4. Get response from LLM (passing original query and history for natural conversation)
        logger.info("Generating LLM Response...")
        answer = self.get_llm_response(context, query, history)
        
        result = {
            "answer": answer,
            "sources": sources
        }
        
        # Store in cache
        self.cache[cache_key] = result
        
        return result
