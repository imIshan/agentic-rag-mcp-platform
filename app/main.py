from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model
from app.retrieval import retrieve_context, format_context
from app.schemas import AskRequest, AskResponse

app = FastAPI(title="Agentic RAG MCP Platform")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = retrieve_context(question)
    if not docs:
        return AskResponse(
            answer="I could not find relevant context in the indexed documents.",
            sources=[],
        )

    context = format_context(docs)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant answering only from the provided context.

        Rules:
        - Use only the context below.
        - If the answer is not in the context, say you do not know.
        - Be concise and accurate.
        - Mention uncertainty when context is incomplete.

        Question:
        {question}

        Context:
        {context}
        """
    )

    chain = prompt | get_chat_model()
    result = chain.invoke({"question": question, "context": context})

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
        }
        for doc in docs
    ]

    return AskResponse(
        answer=result.content,
        sources=sources,
    )