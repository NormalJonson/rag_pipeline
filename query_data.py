from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from groq import Groq
import chromadb
import os, sys
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHROMA_PATH = "chroma"

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
    def __call__(self, input: Documents) -> Embeddings:
        result = gemini_client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=input,
        )
        return [e.values for e in result.embeddings]

def query(query_text: str):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(
        name="documents",
        embedding_function=GeminiEmbeddingFunction()
    )

    results = collection.query(query_texts=[query_text], n_results=3)

    if not results["documents"][0]:
        print("No matching results found.")
        return

    context_text = "\n\n---\n\n".join(results["documents"][0])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    sources = [m.get("source") for m in results["metadatas"][0]]
    print(f"\nResponse: {response.choices[0].message.content}")
    print(f"Sources:  {sources}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 query_data.py "your question here"')
    else:
        query(" ".join(sys.argv[1:]))