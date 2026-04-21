from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
import json
from app.graph import rag_graph
from app.llm import get_chat_model, get_structured_chat_model
from app.retrieval import retrieve_context, format_context
from app.schemas import AskRequest, AskResponse, QueryPlan

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

@app.post("/plan", response_model=QueryPlan)
def plan_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    prompt = ChatPromptTemplate.from_template(
        """
        You are a query planner for a RAG system.

        Classify the user question and return JSON with exactly these keys:
        - question_type: one of ["incident", "api_docs", "architecture", "general"]
        - should_retrieve: boolean
        - reason: short string

        User question:
        {question}
        """
    )

    chain = prompt | get_structured_chat_model()
    result = chain.invoke({"question": question})

    data = json.loads(result.content)
    return QueryPlan(**data)

@app.post("/ask-stream")
def ask_stream(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = retrieve_context(question)
    if not docs:
        def empty_stream():
            yield "I could not find relevant context in the indexed documents."
        return StreamingResponse(empty_stream(), media_type="text/plain")

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

    def generate():
        for chunk in chain.stream({"question": question, "context": context}):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/ask-rag-graph", response_model=AskResponse)
def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_graph.invoke({"question": question})

    return AskResponse(
        answer=result.get("answer", "No answer generated."),
        sources=result.get("sources", []),
    )