from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import settings
from app.llm import get_embedding_model


DATA_DIR = Path("data")


def load_documents():
    docs = []

    for path in DATA_DIR.glob("**/*"):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())

    return docs


def build_vector_store():
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in ./data")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=settings.vector_db_path,
    )
    return vectorstore, len(docs), len(chunks)


if __name__ == "__main__":
    vs, doc_count, chunk_count = build_vector_store()
    print(f"Ingested {doc_count} docs into {chunk_count} chunks.")