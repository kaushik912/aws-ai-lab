# aws-ai-lab guide: 4 weeks, ~$0

Everything runs locally first. AWS spend only where noted. Code for all weeks already exists in this repo; each week = read it, run it, break it, extend it.

## Azure -> AWS mapping (from azure-ai-lab)

| Azure | AWS here | File |
|---|---|---|
| Azure OpenAI chat | Bedrock Converse (Claude Haiku 5.5) | `app/llm.py` |
| AOAI embeddings | Titan Embed v2 (512 dims) | `app/llm.py` |
| AI Search hybrid + semantic rerank | FAISS + BM25 + RRF, in-process | `app/retrieval.py` |
| rerank score gate | best dense cosine gate (`min_cosine`) | `app/agent.py` |
| LangGraph router | same graph + a Converse tool loop | `app/agent.py` |
| DefaultAzureCredential / Workload Identity | boto3 default chain / Lambda exec role | `app/clients.py` |
| AKS + ACR | Lambda container + ECR | `Dockerfile`, `infra/` |
| Key Vault / keys | none needed: IAM role to Bedrock, IAM-signed Function URL | `infra/main.tf` |
| Terraform | Terraform | `infra/` |

## Prereqs (one-time)
1. Bedrock console -> Model access: enable Claude Haiku + Titan Text Embeddings V2 (Anthropic may ask a use-case form).
2. Check: `aws bedrock list-inference-profiles --region us-east-1 | grep haiku`. Claude needs an **inference profile id** (`us.anthropic...`), not the bare model id.
3. `aws sts get-caller-identity` works (SSO or keys).

## Week 1: Bedrock from Python
Goal: call Claude + Titan with boto3, see tokens and cost.

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]' && cp .env.example .env
python -c "
from app.clients import bedrock; from app.config import get_settings as s
r = bedrock().converse(modelId=s().bedrock_chat_model,
    messages=[{'role':'user','content':[{'text':'Say hi in 5 words'}]}],
    inferenceConfig={'maxTokens':50})
print(r['output']['message']['content'][0]['text'], r['usage'])"
```
Learn:
- Converse API is model-agnostic (swap model id, same code). `invoke_model` is model-specific (used for Titan).
- `usage.inputTokens/outputTokens` = your bill. Log it everywhere (`main.py` does).
- boto3 is sync -> `asyncio.to_thread` (`llm.py`). Alternative: `aioboto3`.
- Always set `maxTokens`. It is the hard cost cap.
Exercise: vary system prompt and `temperature`, compare `usage`; call the same prompt with a different model id to see Converse stay unchanged.

## Week 2: local hybrid search
Goal: FAISS + BM25 + RRF, understand why hybrid beats either alone.

```bash
python scripts/ingest.py        # writes ./index/{faiss.index,chunks.json}
pytest tests/test_retrieval.py
```
Learn:
- Dense misses exact tokens (IDs, names); BM25 misses paraphrase. RRF fuses *ranks*, no score calibration needed.
- RRF scores are rank-based -> useless as an absolute "is this relevant" signal. Hence cosine gate (`min_cosine=0.30`, tune it!).
- `IndexFlatIP` + normalized vectors = exact cosine. Fine to ~100K chunks; beyond that use `IndexHNSWFlat`.
- Index ships with the artifact -> no vector DB, no idle cost. Trade-off: update = rebuild + redeploy.
Exercises: (1) add 5-10 of your own docs to `data/`, re-ingest. (2) write 10 Q->expected-source pairs, compute hit@k for dense-only vs BM25-only vs hybrid. (3) tune `min_cosine` on out-of-scope questions. (4) swap in ChromaDB behind the same `Index.search` signature.

## Week 3: async FastAPI + RAG graph + agent
```bash
uvicorn app.main:app --reload
H='content-type: application/json'
curl -s localhost:8000/ask -H "$H" -d '{"question":"How many PTO days do I get?"}' | jq
curl -s localhost:8000/ask -H "$H" -d '{"question":"What is the CEO favorite color?"}' | jq   # declines, 0 LLM tokens
curl -N localhost:8000/ask/stream -H "$H" -d '{"question":"Summarize the security policy"}'
curl -s localhost:8000/agent -H "$H" -d '{"question":"5 days of travel meals, how much can I claim?"}' | jq  # search_docs + calc
```
Learn:
- Graph: retrieve -> grounded? -> answer | decline. Decline path is free.
- Agent: Converse `toolConfig`; loop while `stopReason == "tool_use"`; append `toolResult`; step cap.
- `safe_eval` uses `ast`, never `eval()` on model output (prompt injection -> RCE).
- Streaming: `converse_stream` events (`contentBlockDelta`, `metadata.usage`).
Exercises: retry-with-rewritten-query node when ungrounded; conversation memory; per-request token budget; a `/ask` eval script.

## Week 4: ship to Lambda (needs Docker running)
Why Lambda Web Adapter: run the same uvicorn app unchanged; no Mangum/handler code. Function URL with `RESPONSE_STREAM` gives SSE (API Gateway REST can't stream).

```bash
cd infra && cp terraform.tfvars.example terraform.tfvars   # set budget_email
terraform init && terraform apply -target=aws_ecr_repository.app   # step 1: repo only
cd .. && python scripts/ingest.py                                  # index must exist before build
REPO=$(terraform -chdir=infra output -raw ecr_repo_url)
aws ecr get-login-password | docker login --username AWS --password-stdin ${REPO%%/*}
docker build -t $REPO:v1 . && docker push $REPO:v1
terraform -chdir=infra apply -var image_tag=v1                     # step 2: everything
```
Call it:
```bash
URL=$(terraform -chdir=infra output -raw function_url)
# Function URL uses AWS_IAM: requests must be SigV4-signed with your AWS credentials
curl -s --aws-sigv4 "aws:amz:us-east-1:lambda" --user "$(aws configure get aws_access_key_id):$(aws configure get aws_secret_access_key)" \
  ${URL}ask -H 'content-type: application/json' -d '{"question":"PTO days?"}'
curl -s ${URL}healthz   # unsigned -> 403 from AWS (proves the lock works)
```
(Temporary/SSO credentials also need `-H "x-amz-security-token: $AWS_SESSION_TOKEN"`.)

Security map:
- **IAM role (Lambda -> Bedrock)**: exec role can only `InvokeModel*` on Anthropic + Titan embed ARNs and write logs. Claude via inference profile needs both profile ARN and foundation-model ARNs.
- **IAM auth (caller -> Lambda)**: Function URL `authorization_type = "AWS_IAM"`. AWS rejects unsigned requests before your code runs, so strangers can't spend your tokens. No API key, no Secrets Manager.
- No secrets anywhere in this design. To let others call it: grant their IAM user `lambda:InvokeFunctionUrl`, or add CloudFront/WAF later.
- Cold start: FAISS+numpy import + index load ~1-2s. Memory 1024MB. Check CloudWatch `Init Duration`.
Teardown: `terraform -chdir=infra destroy` (ECR has `force_delete`).

## Keeping the bill at $0

| Item | Cost | Notes |
|---|---|---|
| Lambda | free tier 1M req + 400K GB-s/mo | idle = $0 |
| Function URL | free | vs API Gateway ($1/M, streaming limits) |
| Bedrock Haiku | per token | `maxTokens` 400, decline path, small top_k |
| Titan embed | ~$0.02/M tokens | ingest once, embed 1 query per request |
| ECR | $0.10/GB-mo | lifecycle keeps 3 images (~$0.05) |
| CloudWatch Logs | 5GB free | 7-day retention |
| Budgets | first 2 free | alarm at 50% actual / 100% forecast of `budget_usd` |

Rules:
1. Never use OpenSearch Serverless / Knowledge Bases (OCU minimums ~$170+/mo), NAT gateway, EKS, provisioned throughput.
2. Haiku for dev; switch model by env var only to compare. Avoid Opus-class in loops.
3. The Function URL is IAM-protected, so only your AWS identity can spend tokens. Still keep `maxTokens` low; the Budgets alarm only emails, it doesn't stop spend. For a hard stop, destroy the stack.
4. Don't loop agent tests against real Bedrock; unit tests mock Converse (`tests/conftest.py`).
5. Enable Cost Explorer + a tag filter `project=aws-ai-lab` (default_tags set in `versions.tf`).
6. `terraform destroy` at the end of every session you don't need the URL.

## Differences from Azure lab worth remembering
- No semantic reranker equivalent; to add one: Bedrock Rerank API (`bedrock-agent-runtime.rerank`, Cohere/Amazon rerank models, per-query cost) or a cross-encoder locally.
- IAM-only auth to Bedrock means no key rotation story; Azure lab had a key shortcut.
- Terraform here is two-step apply because Lambda can't exist before its image does (Azure's AKS pulled lazily).
