# 🤖 E-commerce RAG System (AirDoctor & AquaTru)

Welcome to the **E-commerce RAG (Retrieval-Augmented Generation) System**. This project is designed to help users get instant, accurate answers about AirDoctor (air purifiers) and AquaTru (water purifiers) products.

---

## 🌟 What is RAG? (For Freshers)
Imagine you have a giant library (our `data.json`) and a super-smart assistant (Gemini AI). 
1. When a user asks a question, we don't just ask the assistant from its memory (which might be old or wrong).
2. Instead, we first **search** the library for the most relevant pages.
3. We give those pages to the assistant and say: *"Read these pages and answer the user's question using ONLY this info."*

This makes the AI much more accurate and prevents it from making things up (hallucinating).

---

## 🏗️ Project Structure
The project is broken into small, simple pieces:

*   **`data.json`**: Our database. Contains product names, specs, and FAQs.
*   **`chunking.py`**: A tool that cuts long product descriptions into smaller, bite-sized pieces (chunks) so the AI can read them easily.
*   **`embeddings.py`**: Converts human text into numbers (vectors). Computers understand numbers better than words for finding "similar" meanings.
*   **`vector_store.py`**: A special "number-database" (using FAISS) that helps us find the most relevant chunks in milliseconds.
*   **`rag_pipeline.py`**: The "Brain." It coordinates everything: gets the question, finds the chunks, and talks to Gemini AI.
*   **`main.py`**: The "Server." It creates the website API so the UI can talk to the Brain.
*   **`static/index.html`**: The "Face." A beautiful chat interface for the user.

---

## 🚀 Getting Started

### 1. Install Dependencies
Open your terminal and run:
```powershell
pip install -r rag_app/requirements.txt
```

### 2. Set your API Key
The project uses Google Gemini. You should set your API key in your terminal:
```powershell
# Windows
$env:GEMINI_API_KEY="your_api_key_here"
```
*(If you don't have a key, the system will use a "Mock Response" so you can still see how it works!)*

### 3. Run the App
```powershell
python rag_app/main.py
```
Now, open your browser and go to: **`http://localhost:8000`**

---

## 🛠️ How it works (Step-by-Step)
1. **Loading**: We read `data.json` and split it into chunks.
2. **Embedding**: We turn those chunks into mathematical vectors.
3. **Storage**: We put those vectors into `FAISS` (our vector store).
4. **Asking**: When you type "Is AirDoctor 3000 good for allergies?":
   - We turn your question into a vector.
   - We find the top 3 closest chunks in FAISS.
   - We send those 3 chunks + your question to Gemini.
5. **Answering**: Gemini writes a polite answer based *only* on those chunks.

---

## 🛡️ Guardrails
*   **Product Only**: If you ask about "Pizza recipes," the bot will politely tell you it's only an expert in AirDoctor/AquaTru.
*   **Strict Fact-Checking**: If the answer isn't in our data, it says "I don't know" instead of guessing.

---
**Happy Coding!** 🚀
