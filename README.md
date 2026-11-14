# Document Q&A Chatbot (RAG)

Ask questions about your own PDF/TXT files. Answers come from a Llama model on Groq, using **only** text retrieved from your documents, which reduces hallucination.

**Stack:** Python · LangChain · FAISS · Groq API (Llama) · FastEmbed · Streamlit

## How it works
| Step | File | What happens |
|---|---|---|
| 1. Load | `rag/loader.py` | Uploaded PDF/TXT files → text, with file name and page kept |
| 2. Chunk | `rag/indexer.py` | `RecursiveCharacterTextSplitter` cuts the text into overlapping chunks |
| 3. Embed + store | `rag/indexer.py` | A local FastEmbed model (`bge-small-en-v1.5`) turns chunks into vectors stored in **FAISS** |
| 4. Retrieve | `rag/indexer.py` | The top-k chunks most similar to the question |
| 5. Generate | `rag/qa.py` | The LLM is prompted to answer only from those chunks, or say it can't find the answer |
| 6. UI | `app.py` | Streamlit upload, chat, and a "Sources" panel under each answer |

## Run (Windows)
```bash
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Paste a free Groq API key (https://console.groq.com/keys) into the sidebar. Embeddings run locally, so they need no key.

## Ideas to extend
- Save the FAISS index to disk
- Add conversation memory
- Hybrid search (BM25 + vectors)
