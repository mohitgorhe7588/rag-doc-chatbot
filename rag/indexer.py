"""Step B - Chunking + embedding pipeline, stored in a FAISS vector index."""
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Small free embedding model that runs on your own PC (no API key, ~70 MB, downloaded once).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def split_into_chunks(docs, chunk_size=800, chunk_overlap=100):
    """Cut pages into small overlapping pieces so search can find exact passages."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_index(chunks):
    """Turn every chunk into an embedding (numbers = meaning) and store them in FAISS."""
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def search(index, question, k=4):
    """Return the k chunks whose meaning is closest to the question."""
    return index.similarity_search(question, k=k)
