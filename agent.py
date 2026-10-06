from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    query: str
    context: str
    relevant: bool
    answer: str
    attempts: int

def build_agent(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def retrieve(state: AgentState):
        docs = vectorstore.similarity_search(state["query"], k=4)
        context = "\n\n".join(
            f"[Source page {d.metadata.get('page', '?') + 1}] {d.page_content}"
            for d in docs
        )
        return {"context": context, "attempts": state.get("attempts", 0) + 1}

    def grade(state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a retrieval grader. Decide whether the context contains "
             "enough information to answer the question. Reply only YES or NO."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])
        result = llm.invoke(prompt.format_messages(
            question=state["question"], context=state["context"]
        ))
        return {"relevant": result.content.strip().upper().startswith("YES")}

    def rewrite_query(state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Rewrite the user's question into a broader search query for a "
             "study-notes vector database. Keep the meaning but add useful "
             "keywords. Return only the query."),
            ("human", "{question}")
        ])
        result = llm.invoke(prompt.format_messages(question=state["question"]))
        return {"query": result.content.strip()}

    def answer(state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using only the supplied study context. "
             "If the context is insufficient, say that the notes do not "
             "contain enough information. Keep the answer clear and concise. "
             "Mention source page numbers when possible."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])
        result = llm.invoke(prompt.format_messages(
            question=state["question"], context=state["context"]
        ))
        return {"answer": result.content}

    def route_after_grade(state: AgentState):
        if state["relevant"] or state.get("attempts", 0) >= 2:
            return "answer"
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("answer", answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"rewrite": "rewrite", "answer": "answer"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)

    return graph.compile()
