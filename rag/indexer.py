"""Step B - Chunking + embedding pipeline, and hybrid (meaning + keyword) search."""
import re
from dataclasses import dataclass

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Small free embedding model that runs on your own PC (no API key, ~70 MB, downloaded once).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BM25_WEIGHT = 0.8  # keyword votes count slightly less than meaning votes


@dataclass
class HybridIndex:
    """FAISS finds chunks with similar *meaning*; BM25 finds chunks with the same *keywords*."""
    faiss: FAISS
    bm25: BM25Okapi
    chunks: list


def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


STOPWORDS = set("a an the is are was were be to of in on at for and or what how when who which do does did can i you it this that with from by".split())


def _stem(word):
    """Tiny stemmer so 'Sundays' matches 'Sunday' and 'opening' matches 'open'."""
    for suffix in ("ing", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _tokens(text):
    return [_stem(w) for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


def split_into_chunks(docs, chunk_size=800, chunk_overlap=100):
    """Cut pages into small overlapping pieces so search can find exact passages."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_index(chunks, embeddings=None):
    """Embed every chunk into FAISS, and build a BM25 keyword index next to it."""
    embeddings = embeddings or get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    bm25 = BM25Okapi([_tokens(c.page_content) for c in chunks])
    return HybridIndex(vector_store, bm25, chunks)


def search(index, question, k=4):
    """Hybrid search: merge FAISS and BM25 rankings with Reciprocal Rank Fusion (RRF).

    Each list votes 1 / (60 + rank) for its chunks; chunks ranked high by both win.
    This catches exact terms (names, codes, numbers) that pure vector search can miss.
    """
    pool = max(k * 3, 10)
    scores = {}

    for rank, doc in enumerate(index.faiss.similarity_search(question, k=pool)):
        key = doc.page_content
        scores[key] = scores.get(key, (0.0, doc))
        scores[key] = (scores[key][0] + 1 / (60 + rank), doc)

    bm25_scores = index.bm25.get_scores(_tokens(question))
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool]
    for rank, i in enumerate(top_bm25):
        if bm25_scores[i] <= 0:
            break  # no shared keywords at all
        doc = index.chunks[i]
        key = doc.page_content
        prev = scores.get(key, (0.0, doc))
        scores[key] = (prev[0] + BM25_WEIGHT / (60 + rank), prev[1])

    ranked = sorted(scores.values(), key=lambda s: s[0], reverse=True)
    return [doc for _, doc in ranked[:k]]
