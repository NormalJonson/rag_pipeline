from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)

def compare(word1: str, word2: str):
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_query"
    )
    vec1 = embeddings.embed_query(word1)
    vec2 = embeddings.embed_query(word2)

    print(f"\nVector for '{word1}': {vec1[:5]}... (length: {len(vec1)})")
    print(f"Vector for '{word2}': {vec2[:5]}... (length: {len(vec2)})")
    print(f"\nCosine similarity ({word1}, {word2}): {cosine_similarity(vec1, vec2):.4f}")
    print(f"Cosine distance   ({word1}, {word2}): {cosine_distance(vec1, vec2):.4f}")

if __name__ == "__main__":
    compare("apple", "iphone")
    compare("apple", "banana")