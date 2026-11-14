"""Step A - Load PDF/TXT files into LangChain Documents."""
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(files):
    """files: list of (file_name, file_bytes). Returns one Document per PDF page / TXT file."""
    docs = []
    for name, data in files:
        suffix = os.path.splitext(name)[1].lower()
        # Loaders need a real file path, so write the bytes to a temp file first.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8")
            for d in loader.load():
                d.metadata["source"] = name  # remember which file each page came from
                if d.page_content.strip():  # skip empty / image-only pages
                    docs.append(d)
        finally:
            os.remove(path)
    return docs
