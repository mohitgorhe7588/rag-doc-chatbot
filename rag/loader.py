"""Step A - Load uploaded PDF/TXT files into LangChain Documents."""
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(uploaded_files):
    """Turn uploaded files into a list of Documents (one per PDF page / TXT file)."""
    docs = []
    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1].lower()
        # Loaders need a real file path, so write the upload to a temp file first.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8")
            for d in loader.load():
                d.metadata["source"] = f.name  # remember which file each page came from
                docs.append(d)
        finally:
            os.remove(path)
    return docs
