# RAG Pipeline

A local RAG pipeline that lets you query your own documents (make sre they are in Mardown .md format) using natural language.  
Uses **Gemini** for embeddings and **Groq** for answer generation, with **ChromaDB** as the vector database.

---
## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shridev-Kandari/rag_pipeline.git
cd rag_pipeline
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
then,
```bash
pip install "unstructured[md]"
```

### 4. Set up your API keys

Create a `.env` file in the project root:

```bash
nano .env
```

Add your keys:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add your documents

Place your `.md` files inside the `data/books/` directory:

### 6. Build the database
```bash
python3 create_database.py
```
> ⚠️ This will take a while if you have many documents — there is a 65-second delay between batches to respect Gemini's rate limits.

### 7. Query your documents

```bash
python3 query_data.py "Your question here"
```
Example:

```bash
python3 query_data.py "How was Alice feeling?"
```
---
## Deactivating the virtual environment

When you're done:

```bash
deactivate
```

---
- Gemini free tier has rate limits. If you hit quota errors during `create_database.py`, the 65-second sleep between batches should handle it. For `query_data.py`, switching to a new Google project/API key resolves quota issues.
- Groq free tier is generous and requires no billing setup.
