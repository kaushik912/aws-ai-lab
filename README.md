# aws-ai-lab

Cost-optimized RAG + tool-using agent on AWS. Zero idle cost.

Stack: FastAPI (async, SSE) · Bedrock Converse (Claude) + Titan Embed v2 · FAISS + BM25 (RRF) in-process · LangGraph · Lambda (container, Web Adapter, Function URL streaming, IAM auth) · Terraform. No secrets to manage.

Port of `../azure-ai-lab`. See `GUIDE.md` for the 4-week walkthrough.

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'
pytest                      # no AWS needed
cp .env.example .env
python scripts/ingest.py    # real Bedrock call, < $0.01
uvicorn app.main:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"How many PTO days?"}'
```

Endpoints: `GET /healthz`, `POST /ask` (RAG graph), `POST /ask/stream` (SSE), `POST /agent` (tool loop). No app-level auth: deployed, the Function URL uses `AWS_IAM` (SigV4), so AWS rejects unsigned callers.
