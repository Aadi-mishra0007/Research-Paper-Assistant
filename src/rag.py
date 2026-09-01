"""
RAG (Retrieval-Augmented Generation) pipeline.

Flow:
  Uploaded PDF -> Text extraction -> Chunking -> Embeddings -> FAISS vector store -> Retriever
"""

import os
import tempfile
from dataclasses import dataclass
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
# Groq does not offer an embeddings API (it only serves chat/completion models),
# so embeddings run locally and free via sentence-transformers. This has no
# dependency on which LLM provider (Groq/OpenAI/etc.) is used for generation.
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings_singleton = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily create and cache the embeddings model (avoids reloading per call)."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    return _embeddings_singleton


@dataclass
class PaperIndex:
    """Holds everything needed to query one paper."""
    name: str
    vectorstore: FAISS
    raw_docs: List[Document]


def load_pdf_to_documents(file_bytes: bytes, filename: str) -> List[Document]:
    """Write uploaded bytes to a temp file and load it with PyPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
        return docs
    finally:
        os.remove(tmp_path)


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """Split long documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(file_bytes: bytes, filename: str) -> PaperIndex:
    """Full pipeline: PDF bytes -> chunked, embedded FAISS index for one paper."""
    raw_docs = load_pdf_to_documents(file_bytes, filename)
    chunks = chunk_documents(raw_docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return PaperIndex(name=filename, vectorstore=vectorstore, raw_docs=raw_docs)


def retrieve_context(index: PaperIndex, query: str, k: int = 6) -> str:
    """Retrieve top-k relevant chunks for a query and format them as context text."""
    retriever = index.vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    formatted = []
    for i, doc in enumerate(results, start=1):
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Chunk {i} | page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def retrieve_context_for_summary(index: PaperIndex, max_chars: int = 12000) -> str:
    """
    For whole-document tasks (summary, key contributions, etc.) we pull a broad
    sample of chunks across the paper rather than a query-specific top-k,
    since there's no single 'query' representing the whole document.
    """
    retriever = index.vectorstore.as_retriever(search_kwargs={"k": 20})
    results = retriever.invoke(
        "overview, contributions, methodology, results, limitations, future work"
    )
    text = ""
    for doc in results:
        addition = doc.page_content + "\n\n"
        if len(text) + len(addition) > max_chars:
            break
        text += addition
    return text
