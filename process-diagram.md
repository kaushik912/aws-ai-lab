# aws-ai-lab: build, deploy, runtime

Solid boxes = built in this lab. Dashed = what a production team adds.
Bedrock only (Claude Haiku 4.5 + Titan Embed v2). No secrets: IAM role to Bedrock, IAM-signed Function URL for callers.

## 1. Build -> test -> deploy

```mermaid
flowchart TD
    subgraph DEV["1. Develop (local, WSL)"]
        A["Write code<br/>FastAPI + LangGraph"] --> B["pytest (offline, mocked Bedrock)"]
        B --> C["Run locally: uvicorn<br/>uses your AWS login"]
    end

    subgraph DATA["2. Build the index (local)"]
        D1["data/*.md"] --> D2["scripts/ingest.py<br/>chunk -> Titan embed"]
        D2 --> D3[("index/<br/>faiss.index + chunks.json")]
    end

    subgraph INFRA["3. Provision (Terraform, 2 applies)"]
        T1["apply -target ECR repo"] --> T2["apply -var image_tag=v1"]
        T2 --> R["Lambda + IAM Function URL (streaming),<br/>exec role (Bedrock only),<br/>log group, budget alarm"]
    end

    subgraph SHIP["4. Package and ship (Docker on)"]
        S1["docker build<br/>(index baked in)"] --> S2["docker push"]
        S2 --> ECR[("ECR")]
        ECR -. "Lambda pulls image" .-> FN["Lambda function<br/>FastAPI via Web Adapter"]
    end

    subgraph VERIFY["5. Verify"]
        V1["curl --aws-sigv4 Function URL"] --> V2["/ask grounded + decline,<br/>/ask/stream, /agent"]
        V3["unsigned curl -> 403"]
        V4["CloudWatch logs: token usage"]
    end

    C --> D1
    D3 --> S1
    T1 --> S2
    R --> FN
    FN --> V1
    FN --> V4
    V2 --> CLEAN["terraform destroy"]

    subgraph PROD["Production extras (not built)"]
        P1["GitHub Actions CI/CD"]
        P2["RAG eval set as CI gate"]
        P3["Alarms on cost / errors"]
        P4["CloudFront + WAF for non-AWS callers"]
    end
    V2 -.-> P1
    P1 -.-> P2
    P1 -.-> P3
    P1 -.-> P4

    classDef later stroke-dasharray: 5 5,fill:#f5f5f5,color:#333
    class P1,P2,P3,P4 later
```

## 2. Infra at rest (what exists in AWS, and idle cost)

```mermaid
flowchart LR
    CLIENT["Caller<br/>(SigV4-signed)"] -->|"HTTPS"| URL["Lambda Function URL<br/>AWS_IAM, RESPONSE_STREAM<br/>(free)"]
    URL --> FN["Lambda (container image)<br/>FastAPI + FAISS + BM25<br/>index baked in"]
    ECR[("ECR<br/>~$0.05/mo")] -. image .-> FN
    FN -->|"InvokeModel (exec role)"| BR["Bedrock<br/>Claude Haiku 4.5 + Titan Embed v2<br/>pay per token"]
    FN --> CW[("CloudWatch Logs<br/>7-day retention")]
    BUD["AWS Budgets<br/>email alarm"] -.-> ACC["Account bill"]
```

Idle cost: ECR pennies. Lambda, Function URL and logs (within free tier): $0.
No VPC, no NAT gateway, no vector database, no secrets.

## 3. One request (runtime)

```mermaid
sequenceDiagram
    autonumber
    participant U as Caller (curl --aws-sigv4)
    participant A as AWS (Function URL auth)
    participant L as Lambda (FastAPI)
    participant B as Bedrock

    U->>A: POST /ask {question} (signed)
    alt unsigned or not allowed
        A-->>U: 403 (Lambda never runs, nothing billed)
    else signed by allowed IAM identity
        A->>L: forward request
        L->>B: Titan embed(question)
        B-->>L: vector
        L->>L: FAISS dense + BM25 -> RRF fuse (in process)
        alt best cosine >= min_cosine
            L->>B: Converse(context + question) [Claude Haiku 4.5]
            B-->>L: grounded answer + usage
            L-->>U: answer + sources + usage
        else not grounded
            L-->>U: "I don't know" (no Claude call, no tokens)
        end
    end
```

## 4. Who can do what (identity map)

```mermaid
flowchart LR
    ME["You (terraform-admin)"] -->|"terraform apply, ecr push,<br/>lambda:InvokeFunctionUrl"| AWS[("AWS account")]
    ROLE["Lambda execution role"] -->|"bedrock:InvokeModel* on Anthropic + Titan ARNs<br/>write logs"| AWS
    OTHER["Anyone else"] -->|"no IAM permission -> 403"| URL["Function URL"]
```

## Cheat sheet
| Step | File / command |
| --- | --- |
| Infra | `infra/` -> `terraform apply` (two-step: ECR first, then image) |
| Index | `scripts/ingest.py` (re-run when data changes) |
| Image | `Dockerfile` -> `docker build` / `push` (Docker on) |
| Test | `pytest` (no network), then signed `curl` to the Function URL |
| Teardown | `terraform destroy` |

## Differences from the Azure lab
| Azure | Here |
| --- | --- |
| AOAI + AI Search (idle cost) | Bedrock (pay per token) + local FAISS/BM25 ($0 idle) |
| AKS pod + Workload Identity | Lambda + execution role |
| Key Vault / keyless | no secrets at all (IAM everywhere) |
| Semantic reranker | none (cosine gate instead) |
