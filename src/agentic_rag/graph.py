"""LangGraph self-correcting RAG loop.

    retrieve -> grade --good--> generate
                  |
                  +--weak & retries left--> rewrite -> retrieve
                  |
                  +--weak & no retries----> generate (with caveat)

The grading + rewrite loop is the "agentic" part: the system inspects its own
retrieval quality and acts on it instead of blindly generating.
"""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agentic_rag.config import settings
from agentic_rag.retriever import HybridRetriever


class RAGState(TypedDict):
    question: str
    query: str            # current search query (rewritten on retries)
    context: list[dict]   # retrieved chunks: {text, source}
    grade: str            # "good" | "weak"
    retries: int
    answer: str


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)


def build_graph(retriever: HybridRetriever | None = None):
    retriever = retriever or HybridRetriever()
    llm = get_llm()

    def retrieve(state: RAGState) -> dict:
        chunks = retriever.retrieve(state["query"] or state["question"])
        return {"context": chunks}

    def grade(state: RAGState) -> dict:
        # TODO(day2): replace the length heuristic with an LLM grader that
        # judges whether the context can actually answer the question.
        context_text = "\n".join(c["text"] for c in state["context"])
        verdict = "good" if len(context_text) > 200 else "weak"
        return {"grade": verdict}

    def rewrite(state: RAGState) -> dict:
        prompt = (
            "Rewrite this question as a better search query for a paper corpus. "
            "Keep key technical terms, add likely synonyms. Output only the query.\n\n"
            f"Question: {state['question']}\nPrevious query: {state['query']}"
        )
        new_query = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"query": new_query, "retries": state["retries"] + 1}

    def generate(state: RAGState) -> dict:
        context = "\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in state["context"]
        )
        caveat = (
            "\nIf the context is insufficient, say what is missing instead of guessing."
            if state["grade"] == "weak"
            else ""
        )
        messages = [
            SystemMessage(content=(
                "Answer using only the context below. Cite sources by filename. "
                "Be concise and specific." + caveat
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
        ]
        return {"answer": llm.invoke(messages).content}

    def route(state: RAGState) -> str:
        if state["grade"] == "good" or state["retries"] >= settings.max_retries:
            return "generate"
        return "rewrite"

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(question: str) -> dict:
    """Run the full loop; returns {answer, sources, retries}."""
    app = build_graph()
    result = app.invoke({
        "question": question, "query": question, "context": [],
        "grade": "", "retries": 0, "answer": "",
    })
    return {
        "answer": result["answer"],
        "sources": sorted({c["source"] for c in result["context"]}),
        "retries": result["retries"],
    }
