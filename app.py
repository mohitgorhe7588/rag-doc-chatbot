"""Step D - Streamlit web page: upload documents, then chat with them."""
import os

import streamlit as st

from rag.indexer import build_index, get_embeddings, search, split_into_chunks
from rag.loader import load_documents
from rag.qa import DEFAULT_MODEL, list_chat_models, make_standalone, stream_answer

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A Chatbot (RAG)")
st.caption("Upload documents, then ask questions. Answers come only from your files, with sources.")


# ---- Cached helpers (run once, then reused) ----
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def cached_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False, max_entries=3)
def cached_index(files, chunk_size, chunk_overlap):
    """Same files + same settings = reuse the index instead of re-embedding everything."""
    docs = load_documents(files)
    if not docs:
        raise ValueError("No readable text found. Scanned/image-only PDFs aren't supported yet.")
    chunks = split_into_chunks(docs, chunk_size, chunk_overlap)
    return build_index(chunks, cached_embeddings()), len(docs), len(chunks)


@st.cache_data(show_spinner=False)
def cached_models(key):
    return list_chat_models(key)


def saved_key():
    """Key from env var or .streamlit/secrets.toml, so you don't paste it every time."""
    try:
        return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:  # no secrets file
        return os.getenv("GROQ_API_KEY", "")


# ---- Sidebar: API key + settings ----
with st.sidebar:
    st.header("Settings")
    default_key = saved_key()
    api_key = st.text_input("Groq API key", type="password", value=default_key).strip()
    st.caption("Saved key loaded ✓" if default_key else "Free key: console.groq.com/keys (starts with 'gsk_')")
    model = DEFAULT_MODEL
    if api_key:
        try:
            models = cached_models(api_key)
            model = st.selectbox("Model", models, index=models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0)
        except Exception as e:
            st.error(f"Key problem: {e}")
    with st.expander("Advanced"):
        chunk_size = st.slider("Chunk size", 300, 2000, 800, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, 50)
        top_k = st.slider("Chunks retrieved (k)", 1, 8, 4)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []

if api_key:
    os.environ["GROQ_API_KEY"] = api_key  # LangChain's Groq class reads the key from here
else:
    st.info("Paste your Groq API key in the sidebar to begin.")

# ---- Upload + process ----
uploads = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Process documents", type="primary", disabled=not uploads):
    try:
        files = tuple((f.name, f.getvalue()) for f in uploads)
        with st.spinner("Reading, chunking and embedding..."):
            st.session_state.index, n_pages, n_chunks = cached_index(files, chunk_size, chunk_overlap)
        st.session_state.history = []
        st.session_state.file_names = [f.name for f in uploads]
        st.success(f"Indexed {n_pages} pages into {n_chunks} chunks. Ask away below!")
    except Exception as e:
        st.error(f"Could not process documents: {e}")


def show_sources(srcs):
    with st.expander(f"Sources ({len(srcs)})"):
        for i, s in enumerate(srcs, 1):
            page = s.metadata.get("page")
            where = f"page {page + 1}" if page is not None else "text file"
            st.markdown(f"**[{i}] {s.metadata.get('source')}** · {where}")
            st.caption(s.page_content[:400] + ("..." if len(s.page_content) > 400 else ""))


# ---- Chat ----
if "index" in st.session_state:
    st.caption("Chatting with: " + ", ".join(st.session_state.get("file_names", [])))
    history = st.session_state.setdefault("history", [])

    for q, a, srcs in history:
        st.chat_message("user").write(q)
        with st.chat_message("assistant"):
            st.write(a)
            show_sources(srcs)

    question = st.chat_input("Ask a question about your documents", disabled=not api_key)
    if question:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents..."):
                    query = make_standalone(question, history, model)  # handles follow-up questions
                    srcs = search(st.session_state.index, query, top_k)
                reply = st.write_stream(stream_answer(query, srcs, history, model))
                show_sources(srcs)
                history.append((question, reply, srcs))
            except Exception as e:
                st.error(f"Could not answer: {e}")
