"""Step D - Streamlit web page: upload documents, then chat with them."""
import os

import streamlit as st

from rag.indexer import build_index, search, split_into_chunks
from rag.loader import load_documents
from rag.qa import DEFAULT_MODEL, answer_question, list_chat_models

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A Chatbot (RAG)")


@st.cache_data(show_spinner=False)
def get_models(key):
    return list_chat_models(key)


# ---- Sidebar: API key + settings ----
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", "")).strip()
    st.caption("Free key: console.groq.com/keys (starts with 'gsk_')")
    model = DEFAULT_MODEL
    if api_key:
        try:
            models = get_models(api_key)
            model = st.selectbox("Model", models, index=models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0)
        except Exception as e:
            st.error(f"Key problem: {e}")
    chunk_size = st.slider("Chunk size", 300, 2000, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, 50)
    top_k = st.slider("Chunks retrieved (k)", 1, 8, 4)

if api_key:
    os.environ["GROQ_API_KEY"] = api_key  # LangChain's Groq class reads the key from here
else:
    st.info("Paste your Groq API key in the sidebar to begin.")

# ---- Upload + process ----
files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Process documents", disabled=not files):
    try:
        with st.spinner("Reading, chunking and embedding..."):
            docs = load_documents(files)
            chunks = split_into_chunks(docs, chunk_size, chunk_overlap)
            st.session_state.index = build_index(chunks)
            st.session_state.history = []
        st.success(f"Indexed {len(docs)} pages into {len(chunks)} chunks. Ask away below!")
    except Exception as e:
        st.error(f"Could not process documents: {e}")

# ---- Chat ----
if "index" in st.session_state:
    for q, a, srcs in st.session_state.history:
        st.chat_message("user").write(q)
        with st.chat_message("assistant"):
            st.write(a)
            with st.expander("Sources"):
                for s in srcs:
                    page = s.metadata.get("page")
                    st.caption(f"{s.metadata.get('source')} (page {page + 1 if page is not None else '-'})")
                    st.write(s.page_content[:300] + "...")

    question = st.chat_input("Ask a question about your documents", disabled=not api_key)
    if question:
        try:
            with st.spinner("Thinking..."):
                srcs = search(st.session_state.index, question, top_k)
                reply = answer_question(question, srcs, model)
            st.session_state.history.append((question, reply, srcs))
            st.rerun()
        except Exception as e:
            st.error(f"Could not answer: {e}")
