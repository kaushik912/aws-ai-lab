import json
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import (
    DECLINE,
    State,
    build_graph,
    build_messages,
    is_grounded,
    retrieve_node,
    run_agent,
    system_prompt,
)
from .llm import converse_stream

log = logging.getLogger("aws-ai-lab")
logging.basicConfig(level=logging.INFO)

# No app-level auth: on Lambda the Function URL uses AWS_IAM (SigV4), so AWS rejects
# unsigned callers before this code runs. Locally it's just localhost.
app = FastAPI(title="aws-ai-lab")
graph = build_graph()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    """RAG graph: retrieve -> answer | decline."""
    out = await graph.ainvoke({"question": req.question})
    log.info("ask usage=%s", out.get("usage"))  # tokens = your bill; watch this in CloudWatch
    return {"answer": out["answer"], "sources": out.get("context", []), "usage": out.get("usage")}


@app.post("/agent")
async def agent(req: AskRequest):
    """Tool-use loop (search_docs, calc)."""
    out = await run_agent(req.question)
    log.info("agent calls=%d usage=%s", len(out["tool_calls"]), out["usage"])
    return out


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE token stream, same grounding gate. On Lambda needs a Function URL in RESPONSE_STREAM mode."""
    state: State = {"question": req.question}
    state.update(await retrieve_node(state))

    async def gen():
        if is_grounded(state) == "decline":
            yield f"data: {json.dumps(DECLINE)}\n\n"
        else:
            async for ev in converse_stream(build_messages(state), system_prompt(state)):
                if "text" in ev:
                    yield f"data: {json.dumps(ev['text'])}\n\n"
                else:
                    log.info("stream usage=%s", ev["usage"])
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
