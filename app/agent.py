"""Two flows.

1. RAG graph (LangGraph): retrieve -> grounded? -> answer | decline.
   'decline' makes no LLM call: cheap, and prevents hallucination.
2. Tool-use agent (Bedrock Converse tool loop): model decides when to call search_docs / calc.
   Capped by max_agent_steps so a confused model can't burn tokens.
"""
import ast
import operator
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .llm import converse
from .retrieval import hybrid_search

DECLINE = "I don't know based on the indexed documents."


class State(TypedDict, total=False):
    question: str
    context: list[dict]
    answer: str
    usage: dict


async def retrieve_node(state: State) -> State:
    return {"context": await hybrid_search(state["question"])}


def is_grounded(state: State) -> str:
    ctx = state.get("context") or []
    ok = any(c["cosine"] >= get_settings().min_cosine for c in ctx)
    return "answer" if ok else "decline"


def system_prompt(state: State) -> str:
    ctx = state.get("context") or []
    joined = "\n\n".join(f"[{c['source']}] {c['content']}" for c in ctx)
    return f"Answer ONLY from the context; cite sources in [brackets].\n\n{joined}"


def build_messages(state: State) -> list[dict]:
    return [{"role": "user", "content": [{"text": state["question"]}]}]


async def answer_node(state: State) -> State:
    resp = await converse(build_messages(state), system_prompt(state))
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    return {"answer": text, "usage": resp["usage"]}


async def decline_node(state: State) -> State:
    return {"answer": DECLINE}


def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_node("decline", decline_node)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", is_grounded, {"answer": "answer", "decline": "decline"})
    g.add_edge("answer", END)
    g.add_edge("decline", END)
    return g.compile()


# ---------------------------------------------------------------- tool-use agent

TOOLS = [
    {"toolSpec": {
        "name": "search_docs",
        "description": "Search the company handbook. Use for any policy question.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"query": {"type": "string"}},
                                 "required": ["query"]}},
    }},
    {"toolSpec": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression, e.g. '60*5 + 800'.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"expression": {"type": "string"}},
                                 "required": ["expression"]}},
    }},
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_eval(expr: str) -> float:
    """Arithmetic only. Never eval() model output."""
    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](walk(n.operand))
        raise ValueError("unsupported expression")
    return walk(ast.parse(expr, mode="eval").body)


async def run_tool(name: str, args: dict) -> str:
    try:
        if name == "search_docs":
            hits = await hybrid_search(args["query"])
            return "\n\n".join(f"[{h['source']}] {h['content']}" for h in hits) or "no results"
        if name == "calc":
            return str(safe_eval(args["expression"]))
    except Exception as e:  # tool errors go back to the model, not the caller
        return f"error: {e}"
    return f"error: unknown tool {name}"


async def run_agent(question: str) -> dict:
    s = get_settings()
    system = "Use tools to answer. Cite sources in [brackets]. Be concise."
    messages = [{"role": "user", "content": [{"text": question}]}]
    calls: list[dict] = []
    tokens = {"inputTokens": 0, "outputTokens": 0}
    for _ in range(s.max_agent_steps):
        resp = await converse(messages, system, TOOLS)
        for k in tokens:
            tokens[k] += resp["usage"].get(k, 0)
        msg = resp["output"]["message"]
        messages.append(msg)
        if resp["stopReason"] != "tool_use":
            text = "".join(b.get("text", "") for b in msg["content"])
            return {"answer": text, "tool_calls": calls, "usage": tokens}
        results = []
        for block in msg["content"]:
            if "toolUse" in block:
                tu = block["toolUse"]
                out = await run_tool(tu["name"], tu["input"])
                calls.append({"tool": tu["name"], "input": tu["input"]})
                results.append({"toolResult": {"toolUseId": tu["toolUseId"],
                                               "content": [{"text": out}]}})
        messages.append({"role": "user", "content": results})
    return {"answer": "Stopped: step limit reached.", "tool_calls": calls, "usage": tokens}
