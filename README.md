# Document Q&A Chatbot (RAG)

Ask questions about your own PDF/TXT files. Answers come from a Llama model on Groq, using **only** text retrieved from your documents, with numbered citations, which reduces hallucination.

**Stack:** Python · LangChain · FAISS · BM25 · Groq API (Llama) · FastEmbed · Streamlit

## Features
- **Hybrid search**: FAISS vector search (meaning) + BM25 (exact keywords), merged with Reciprocal Rank Fusion
- **Grounded answers with citations** like `[1]`, plus a Sources panel showing file and page
- **Chat memory**: follow-up questions ("what about on Sundays?") are rewritten into standalone questions before searching
- **Streaming**: answers appear word by word
- **Fast re-use**: the same files and settings aren't re-embedded (cached index)
- **Free local embeddings**: no key needed for the embedding step

## How it works
| Step | File | What happens |
|---|---|---|
| 1. Load | `rag/loader.py` | PDF/TXT → text, with file name and page kept |
| 2. Chunk | `rag/indexer.py` | `RecursiveCharacterTextSplitter` cuts the text into overlapping chunks |
| 3. Embed + index | `rag/indexer.py` | FastEmbed (`bge-small-en-v1.5`) vectors in **FAISS**, plus a **BM25** keyword index |
| 4. Retrieve | `rag/indexer.py` | Top-k chunks from both indexes, fused with RRF |
| 5. Generate | `rag/qa.py` | The LLM answers only from those chunks and cites them, or says it can't find the answer |
| 6. UI | `app.py` | Streamlit upload, streaming chat, Sources panel |

## Run (Windows)
```bash
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Get a free Groq key at https://console.groq.com/keys and paste it into the sidebar.
To skip pasting it every time, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put your key in it. That file is git-ignored.

## Ideas to extend
- Save the FAISS index to disk
- OCR for scanned PDFs
- Evaluate answer quality on a test question set
