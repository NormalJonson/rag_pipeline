from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
import chromadb
import os, shutil, time
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

client = genai.Client(api_key=GOOGLE_API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
    def __call__(self, input: Documents) -> Embeddings:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=input,
        )
        return [e.values for e in result.embeddings]

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    return loader.load()

def split_text(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100,
        length_function=len, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    print(f"\nSample chunk:\n{chunks[10].page_content}")
    print(f"Metadata: {chunks[10].metadata}")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.create_collection(
        name="documents",
        embedding_function=GeminiEmbeddingFunction()
    )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    batch_size = 50
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"Added chunks {i} to {min(i+batch_size, len(texts))}...")
        print("Waiting 65 seconds for rate limit...")
        time.sleep(65)

    print(f"\nDone! Saved {len(chunks)} chunks to '{CHROMA_PATH}'.")

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_text(docs)
    save_to_chroma(chunks)
