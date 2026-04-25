from langchain_chroma import Chroma

from app.config import settings
from app.llm import get_embedding_model


def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=settings.vector_db_path,
        embedding_function=get_embedding_model(),
    )

def retrieve_context(query: str, k: int | None = None):
    vectorstore = get_vector_store()
    docs = vectorstore.similarity_search(query, k=k or settings.top_k)
    return docs

def format_context(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        formatted.append(
            f"[Chunk {i}]\nSource: {source}\nPage: {page}\nContent:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

def list_available_sources() -> list[str]:
    vectorstore = get_vector_store()
    data = vectorstore.get()

    metadatas = data.get("metadatas", [])
    unique_sources = sorted(
        {
            metadata.get("source", "unknown")
            for metadata in metadatas
            if metadata and metadata.get("source")
        }
    )
    return unique_sources