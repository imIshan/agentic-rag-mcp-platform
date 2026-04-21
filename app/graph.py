from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model, get_structured_chat_model
from app.retrieval import retrieve_context, format_context

import json


class RAGState(TypedDict, total=False):
    question: str
    question_type: str
    should_retrieve: bool
    reason: str
    retrieved_docs: list
    context: str
    answer: str
    sources: list[dict]


def plan_node(state: RAGState) -> RAGState:
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
    result = chain.invoke({"question": state["question"]})

    data = json.loads(result.content)

    return {
        "question_type": data["question_type"],
        "should_retrieve": data["should_retrieve"],
        "reason": data["reason"],
    }


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"]
    docs = retrieve_context(question)
    context = format_context(docs)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
        }
        for doc in docs
    ]

    return {
        "retrieved_docs": docs,
        "context": context,
        "sources": sources,
    }


def answer_node(state: RAGState) -> RAGState:
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
    result = chain.invoke(
        {
            "question": state["question"],
            "context": state.get("context", ""),
        }
    )

    return {
        "answer": result.content
    }


def direct_answer_node(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question directly.
        Question:
        {question}
        """
    )

    chain = prompt | get_chat_model()
    result = chain.invoke({"question": state["question"]})

    return {
        "answer": result.content,
        "sources": [],
    }


def route_after_plan(state: RAGState) -> str:
    if state.get("should_retrieve"):
        return "retrieve"
    return "direct_answer"


def build_graph():
    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("plan", plan_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("answer", answer_node)
    graph_builder.add_node("direct_answer", direct_answer_node)

    graph_builder.add_edge(START, "plan")
    graph_builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer",
        },
    )
    graph_builder.add_edge("retrieve", "answer")
    graph_builder.add_edge("answer", END)
    graph_builder.add_edge("direct_answer", END)

    return graph_builder.compile()

rag_graph = build_graph()