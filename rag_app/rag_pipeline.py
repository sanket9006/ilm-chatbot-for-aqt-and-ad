import os
from typing import List, Dict
from embeddings import EmbeddingModel
from vector_store import VectorStore
from data_loader import load_products, prepare_chunks

# Use the new google-genai library
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class RAGPipeline:
    def __init__(self, data_path: str):
        self.embedding_model = EmbeddingModel()
        
        # Load and prepare data
        products = load_products(data_path)
        chunks = prepare_chunks(products)
        
        # Initialize Vector Store
        sample_embedding = self.embedding_model.get_embeddings([chunks[0]["text"]])
        dimension = sample_embedding.shape[1]
        self.vector_store = VectorStore(dimension)
        
        # Add chunks to vector store
        embeddings = self.embedding_model.get_embeddings([c["text"] for c in chunks])
        self.vector_store.add_chunks(embeddings, chunks)
        
        # Simple in-memory cache for queries (Bonus)
        self.cache = {}

    def get_llm_response(self, context: str, query: str, history: List[Dict] = None) -> str:
        """
        Calls Gemini or a Mock LLM using the new google-genai library.
        """
        # Priority: Environment variable -> Hardcoded fallback (not recommended for prod)
        api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyBl0JjZm243KSd634AywFcohhtsu_O9f-s"
        
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
                # Secondary fallback to flash-latest
                try:
                    response = client.models.generate_content(
                        model="gemini-flash-latest",
                        contents=prompt
                    )
                    return response.text
                except:
                    return f"[LLM ERROR: {str(e)}] (Fallback to mock) Mock response for: {query}"
        else:
            # Simple Mock LLM for demo purposes
            return f"Mock Answer: Based on the context provided, I found relevant information about your query '{query}'. However, a GEMINI_API_KEY was not found to generate a natural response."

    def rewrite_query(self, query: str, history: List[Dict]) -> str:
        """
        Rewrites the query to be standalone, resolving pronouns based on history.
        """
        if not HAS_GEMINI or not history:
            return query
            
        api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyBl0JjZm243KSd634AywFcohhtsu_O9f-s"
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
        
        # 0. Rewrite query if history exists
        standalone_query = self.rewrite_query(query, history)
        
        # Cache uses the full context to avoid bad cache hits
        cache_key = f"{standalone_query}_{len(history)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # 1. Embed query (using standalone)
        query_vec = self.embedding_model.get_query_embedding(standalone_query)
        
        # 2. Retrieve top chunks
        retrieved_chunks = self.vector_store.search(query_vec, top_k=3)
        
        # 3. Build context
        context = "\n---\n".join([c["text"] for c in retrieved_chunks])
        sources = list(set([c["metadata"]["product_name"] for c in retrieved_chunks]))
        
        # 4. Get response from LLM (passing original query and history for natural conversation)
        answer = self.get_llm_response(context, query, history)
        
        result = {
            "answer": answer,
            "sources": sources
        }
        
        # Store in cache
        self.cache[cache_key] = result
        
        return result
