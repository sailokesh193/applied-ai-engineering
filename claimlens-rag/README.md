# ClaimLens RAG

A production RAG pipeline built on LangGraph: **document parsing → chunking → embeddings → hybrid vector search → reranking → cited LLM generation**. It runs on **AWS** or **Azure**, picked with one setting, and runs locally with no cloud account.

Each cloud can use either your own pipeline or its managed RAG service (Bedrock Knowledge Bases or Foundry IQ). Both sit behind the same interface, so you can compare them on one eval set.

All sample data is synthetic.

---

## Architecture

```
                 INGEST (LangGraph)                                QUERY (LangGraph)
  files ─► parse ─► chunk ─► embed ─► index          question ─► retrieve ─► rerank ─► gate ─┬─► generate ─► verify citations ─► answer
                                                                                              └─► refuse (nothing relevant)
```

| Stage | Local (offline) | AWS | Azure |
| --- | --- | --- | --- |
| Parsing | Markdown/text, Docling for PDF (optional) | Amazon Textract (async, LAYOUT + TABLES) | Document Intelligence `prebuilt-layout` → markdown |
| Chunking | Section-aware, sentence-split, overlap (shared) | same | same |
| Embeddings | Feature hashing, or BGE via sentence-transformers | Bedrock Titan Text Embeddings V2 | Azure OpenAI `text-embedding-3-large` |
| Vector store | JSON file, BM25 + cosine with RRF | OpenSearch Serverless (vector collection), BM25 + k-NN with RRF | Azure AI Search, hybrid (BM25 + HNSW, RRF) |
| Reranker | Lexical, or `bge-reranker` cross-encoder | Bedrock Rerank API (Cohere Rerank 3.5 / Amazon Rerank) | AI Search semantic ranker |
| LLM | Extractive stand-in, or any OpenAI-compatible endpoint (vLLM, NIM) | Bedrock Converse API | Azure OpenAI chat |
| Managed RAG (`RETRIEVAL_MODE=managed`) | not available | Bedrock Knowledge Bases (managed or S3 Vectors) | Foundry IQ knowledge base (agentic retrieval, API `2026-04-01`) |
| Hosting | `uvicorn` / Docker | ECS Fargate behind an ALB | Azure Container Apps |

Generation is the same everywhere:
- Sources are numbered, and every sentence has to cite one.
- Citations to sources that weren't provided are removed.
- If the best reranked source scores below `MIN_RELEVANCE`, the pipeline refuses instead of guessing.

## Quickstart (local, no cloud)

```bash
pip install -e ".[dev]"
cp .env.example .env            # CLOUD=local by default
claimlens ingest data/sample
claimlens ask "What CPT code was billed on claim CLM-1001?"
claimlens ask "What was billed?" --filter claim_id=CLM-1002
claimlens eval evals/golden.jsonl --min-hit-rate 0.8 --min-refusal-accuracy 1.0
claimlens serve                 # http://localhost:8000/docs
```

The local stack uses an extractive stand-in instead of a real LLM so tests are deterministic. To use a real model, set `OPENAI_API_KEY`, or point `OPENAI_BASE_URL` at vLLM or NVIDIA NIM. For real local models (BGE, bge-reranker, Docling), install `.[local-models]` and set `EMBEDDER=bge RERANKER=cross-encoder`.

## API

| Method | Path | Body |
| --- | --- | --- |
| GET | `/health` | Shows the resolved config and components |
| POST | `/ingest` | multipart `files` (pdf, md, txt, docx, images) |
| POST | `/query` | `{"question": "...", "filters": {"claim_id": "CLM-1001"}}` |

The `/query` response contains `answer`, `grounded`, `citations` (source, section, page, quote) and a `trace` with candidate scores and per-step timings.

## Switching clouds and modes

```bash
CLOUD=aws   RETRIEVAL_MODE=pipeline   # Textract -> Titan -> OpenSearch -> Bedrock rerank -> Bedrock LLM
CLOUD=aws   RETRIEVAL_MODE=managed    # Bedrock Knowledge Base retrieve -> Bedrock LLM
CLOUD=azure RETRIEVAL_MODE=pipeline   # Doc Intelligence -> Azure OpenAI -> AI Search + semantic ranker -> Azure OpenAI
CLOUD=azure RETRIEVAL_MODE=managed    # Foundry IQ knowledge base -> Azure OpenAI
```

You can override any single component and mix clouds, for example `CLOUD=azure RERANKER=bedrock` or `CLOUD=aws LLM=openai`.

---

## Deploy on AWS

1. **Infrastructure.** This creates the S3 bucket, OpenSearch Serverless collection, ECR, ECS Fargate, ALB and IAM roles.
   ```bash
   cd infra/aws
   terraform init
   terraform apply -target=aws_ecr_repository.api \
     -var bedrock_llm_model=<model-or-inference-profile-id> \
     -var bedrock_rerank_model_arn=<rerank-model-arn> \
     -var 'admin_principal_arns=["arn:aws:iam::<acct>:role/<your-role>"]'
   ```
2. **Image:** `make aws-push AWS_REGION=us-east-1 ECR=$(terraform output -raw ecr_repository_url)`
3. **Everything else:** run `terraform apply` with the same `-var`s.
4. **Ingest from your machine or CI.** The vector store is shared with the API.
   ```bash
   export CLOUD=aws AWS_DOCS_BUCKET=$(terraform output -raw docs_bucket) AOSS_ENDPOINT=$(terraform output -raw aoss_endpoint) \
          BEDROCK_LLM_MODEL=... BEDROCK_RERANK_MODEL_ARN=...
   claimlens ingest data/sample
   curl -s $(terraform output -raw api_url)/query -H 'content-type: application/json' -d '{"question":"What CPT code was billed on CLM-1001?"}'
   ```
5. **Managed mode (optional):**
   ```bash
   python scripts/aws_create_knowledge_base.py --type managed --role-arn $(terraform output -raw kb_role_arn) --bucket <docs_bucket>
   # export the printed BEDROCK_KB_* values, then:
   RETRIEVAL_MODE=managed claimlens ingest data/sample   # uploads to S3 and starts the KB sync job
   RETRIEVAL_MODE=managed claimlens eval evals/golden.jsonl
   ```
   Use `--type s3vectors` for a customer-managed knowledge base on S3 Vectors. It's the cheapest option but supports semantic search only.

Model access: enable the embedding, rerank and generation models in the Bedrock console for your Region. Rerank models are offered in fewer Regions, so check before deploying.

## Deploy on Azure

1. **Infrastructure.** This creates AI Search (Basic, with semantic ranker), Document Intelligence, Azure OpenAI deployments, Blob storage, ACR, Container Apps and a user-assigned identity with RBAC. The app uses no keys.
   ```bash
   cd infra/azure
   terraform init
   terraform apply -var subscription_id=<sub-id>
   ```
2. **Image:** `make azure-push ACR=$(terraform output -raw acr_login_server)`
3. **Point the app at the image:** `terraform apply -var subscription_id=<sub-id> -var container_image=<acr>/claimlens:latest`
4. **Ingest.** Terraform gives your own identity the same data roles as the app.
   ```bash
   export CLOUD=azure AZURE_SEARCH_ENDPOINT=$(terraform output -raw search_endpoint) \
          AZURE_OPENAI_ENDPOINT=$(terraform output -raw openai_endpoint) AZURE_DOCINTEL_ENDPOINT=$(terraform output -raw docintel_endpoint)
   claimlens ingest data/sample
   ```
5. **Managed mode (Foundry IQ)** puts a knowledge base over the same index, so the two modes are directly comparable:
   ```bash
   python scripts/azure_create_knowledge_base.py --endpoint $AZURE_SEARCH_ENDPOINT
   RETRIEVAL_MODE=managed claimlens eval evals/golden.jsonl
   ```

Role assignments can take a few minutes to apply. If the first calls fail with 403, wait and retry.

## Evaluation

`evals/golden.jsonl` holds answerable and unanswerable questions. `claimlens eval` reports:
- retrieval hit rate
- citation precision
- answer term coverage
- grounded rate
- refusal accuracy
- p50 latency

With thresholds set, it exits non-zero, which makes it the CI gate in `.github/workflows/ci.yml`. To compare modes, run the same golden file with `RETRIEVAL_MODE=pipeline` and then `managed`.

## Project layout

```
src/claimlens/
  graphs.py          LangGraph ingest + query graphs
  chunking.py        section-aware chunker
  factory.py         builds the pipeline from settings
  providers/         base.py (interfaces), local.py, aws.py, azure.py
  api.py, cli.py, evals.py, prompts.py, config.py
infra/aws, infra/azure   Terraform
scripts/             managed knowledge base setup (Bedrock KB, Foundry IQ)
tests/               offline tests (botocore Stubber, httpx mocks, local end to end)
```

## Production notes

- **Security:** both stacks expose a public endpoint with no authentication. Before using real PHI, put the API behind your identity provider (ALB + Cognito/OIDC, or Container Apps authentication), add HTTPS, and restrict `allowed_cidrs`. Use private endpoints for OpenSearch and AI Search.
- **Re-ingesting a document** replaces its chunks, because `doc_id` comes from the file name.
- **OpenSearch Serverless vector collections** don't accept custom document IDs. Chunks are stored without them, and deletes run through a search on `doc_id`.
- **Model IDs, versions and Regions change often.** They are variables, not hard-coded. Check the Bedrock console and Foundry portal before deploying.
- **Foundry IQ on the stable `2026-04-01` API** returns extractive grounding data only. LLM query planning and answer synthesis are preview features, so generation stays in this pipeline's LLM step.
